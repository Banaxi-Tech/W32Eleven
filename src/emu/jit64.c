/* x86-64 guest -> i386 host JIT.
 *
 * Translation unit: a straight-line block of 64-bit-mode guest code inside one
 * 4 KiB page, ending at a control transfer, an instruction that can change
 * interrupt or machine state (I/O, STI/CLI/POPF/IRET, CR/MSR writes, HLT, ...),
 * a page boundary, or 64 instructions.
 *
 * Guest state stays in memory (struct Cpu) between instructions, so generated
 * code and the interpreter interoperate freely:
 *   - hot instructions (MOV/LEA/ALU/PUSH/POP/MOVZX/MOVSX/CALL/RET/JMP/Jcc) are
 *     translated to native i386 code, with lazy flags stored exactly as the
 *     interpreter stores them;
 *   - memory operands go through an inline lookup of the interpreter's TLB;
 *     on a miss the whole instruction is executed by the interpreter instead;
 *   - every other instruction is a call into the interpreter.
 * Blocks are entered only when the run loop's remaining step budget covers
 * the whole block, and they never contain an instruction that can make an
 * interrupt deliverable before the block's last instruction, so execution is
 * step-for-step identical to the interpreter (same device timing, same state).
 *
 * Self-modifying code: a page holding translated code is removed from the
 * write TLB; the next write to it goes through the slow path, which
 * invalidates the page's blocks before the write happens. Every interpreted
 * instruction inside a block is followed by a validity check, so a block that
 * modifies its own page leaves right after the write.
 *
 * Register use in generated code: EBP = Cpu *, everything else is scratch.
 */
#include "cpu_int.h"

#if defined(__i386__)

void cpu_step_insn(Cpu *c);
int cpu_cond(Cpu *c, int cc);
u32 cpu_lf_cf(Cpu *c);
u64 get_flags(Cpu *c);
u8 *cpu_tlb_fill(Cpu *c, u32 lo, u32 hi, u32 wsz);
u8 *cpu_exec_probe(Cpu *c, u64 la);
extern u32 jit_off;
void cpu_bt_flags(Cpu *c, u32 cf);
void cpu_set_cf_of(Cpu *c, u32 bits);

/* generated code keeps EBP = Cpu + CPU_BIAS, so the hot fields (registers,
 * rip, flags, lazy flags, JIT scratch: the first 256 bytes) take 8-bit offsets */
#define CPU_BIAS 128
#define OFF(f) ((u32)__builtin_offsetof(Cpu, f) - CPU_BIAS)
#define R_LO(n) (OFF(r) + 8u * (u32)(n))
#define R_HI(n) (OFF(r) + 8u * (u32)(n) + 4u)
enum { EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI };

#define HASH_SIZE (1u << 20)
#define JC_SIZE (1u << 16)
#define MAX_BLOCK_INSNS 64
#define CODE_RESERVE 8192u

typedef struct JitBlock {
    u64 pa, rip;        /* keyed by both: the block embeds its guest RIP */
    void (*fn)(Cpu *);
    u8 *entry;          /* chain entry (after the prologue) */
    u8 *host_page;      /* host address of the block's physical page */
    u8 *inval;          /* exit stub its entry is redirected to when the block is invalidated */
    u32 in_links;       /* incoming direct links (Jit.links index + 1, 0 = none) */
    u32 m32;            /* translated for 32-bit flat protected/compatibility mode */
    u32 ninsns;
    u32 valid;
    struct JitBlock *hnext, *pnext;
    struct JitBlock *slow;   /* same code without the stack region (its check failed) */
    u32 hits, cold_len;      /* placeholder for code not run often enough yet: executions so far, instructions to interpret */
    u32 heat, hot;           /* executions left before promotion; translated into the promoted (dense hot) arena */
} JitBlock;

/* jump cache: RIP -> block. An entry is used only while the exec TLB maps the
 * RIP's page to the block's physical page, so guest TLB flushes need no work here. */
typedef struct { u64 rip; JitBlock *b; u32 pad; } JcEntry;
_Static_assert(sizeof(JcEntry) == 16, "generated code indexes JcEntry with <<4");
_Static_assert(sizeof(TlbEntry) == 16, "generated code indexes TlbEntry with <<4");

typedef struct Jit {
    u8 *code, *code_end, *code_ptr;
    u8 *cold, *cold_end, *cold_ptr;   /* out-of-line code: C entry prologues, block tails */
    u8 *pcode, *pcode_end, *pcode_ptr;   /* promoted blocks: the hot code, densely */
    JitBlock *blocks;
    u32 nblocks, maxblocks;
    JitBlock **hash;
    JitBlock **page_head;
    u8 *code_bits;
    u32 npages;
    JcEntry *jc;
    u32 flushes, invalidations;
    u64 jit_insns;          /* instructions executed inside blocks */
    u64 translations;
    u8 *pending_from;   /* exit jump to patch once its target block is known */
    u64 pending_rip;
    u32 pending_gen;
    u32 gen;            /* bumped by every flush and every guest TLB flush */
    Machine *m;
    u8 *interp_stub;    /* shared routine behind every interpreted instruction (see emit_interp) */
    u8 *exit_stub;      /* shared return to C (see emit_exit) */
    u8 *tlb_fn[2][5];   /* shared TLB lookups [write][log2 size] (see emit_tlb) */
    u8 *disp_fn[2];     /* shared indirect dispatch [m32] (see emit_dispatch) */
    u8 *lookup_fn[2], *miss_fn;   /* dispatch with a per-site indirect jump: lookup (EDX = entry or 0), leave to C */
    u8 *lfcf_fn;        /* shared: EAX = CF of the lazy state (see emit_lfcf_routine) */
    u8 *lfzf_fn;        /* shared: host ZF = guest ZF (see emit_lfzf_routine) */
    struct Link { u8 *site, *stub; u32 next; } *links;   /* patched jmp rel32 fields (and their unlinked target), per target block */
    u32 nlinks, maxlinks;
} Jit;

/* ================= emitter ================= */

typedef struct { u8 *p, *end; int over; } E;

static inline __attribute__((always_inline)) void e8(E *e, u32 v) { if (e->p < e->end) *e->p++ = (u8)v; else e->over = 1; }
static inline __attribute__((always_inline)) void e32(E *e, u32 v) {
    if (e->p + 4 <= e->end) { __builtin_memcpy(e->p, &v, 4); e->p += 4; } else e->over = 1;
}

/* ModRM + displacement for [ebp + disp]: disp8 when it fits */
static void ebp_disp(E *e, int reg, u32 disp) {
    if ((i32)disp >= -128 && (i32)disp <= 127) { e8(e, 0x45 | reg << 3); e8(e, disp); }
    else { e8(e, 0x85 | reg << 3); e32(e, disp); }
}
static void mov_r_m(E *e, int r, u32 disp) { e8(e, 0x8b); ebp_disp(e, r, disp); }
static void mov_m_r(E *e, u32 disp, int r) { e8(e, 0x89); ebp_disp(e, r, disp); }
static void mov_m_i(E *e, u32 disp, u32 imm) { e8(e, 0xc7); ebp_disp(e, 0, disp); e32(e, imm); }
/* set while the host CF must survive register loads (ADC/SBB carry-in) */
static int g_keepflags;
static void mov_r_i(E *e, int r, u32 imm) {
    if (!imm && !g_keepflags) { e8(e, 0x31); e8(e, 0xc0 | r << 3 | r); return; }   /* xor r, r */
    e8(e, 0xb8 + r); e32(e, imm);
}
static void mov_r_r(E *e, int d, int s) { e8(e, 0x89); e8(e, 0xc0 | s << 3 | d); }
/* op r/m32(dst), r32(src) — opcode is the "/r, r/m first" form (01, 09, 11, ...) */
static void alu_r_r(E *e, u8 op, int d, int s) { e8(e, op); e8(e, 0xc0 | s << 3 | d); }
/* op r32, [ebp+disp] — opcode is the "r, r/m" form (03, 0B, 13, ...) */
static void alu_r_m(E *e, u8 op, int r, u32 disp) { e8(e, op); ebp_disp(e, r, disp); }
static void alu_r_i(E *e, int ext, int r, u32 imm) {
    if ((i32)imm >= -128 && (i32)imm <= 127) { e8(e, 0x83); e8(e, 0xc0 | ext << 3 | r); e8(e, imm); return; }
    e8(e, 0x81); e8(e, 0xc0 | ext << 3 | r); e32(e, imm);
}
static void alu_m_i(E *e, int ext, u32 disp, u32 imm) {
    if ((i32)imm >= -128 && (i32)imm <= 127) { e8(e, 0x83); ebp_disp(e, ext, disp); e8(e, imm); return; }
    e8(e, 0x81); ebp_disp(e, ext, disp); e32(e, imm);
}
static void shift_r_i(E *e, int ext, int r, u8 n) { e8(e, 0xc1); e8(e, 0xc0 | ext << 3 | r); e8(e, n); }
static void shld_r_r_i(E *e, int d, int s, u8 n) { e8(e, 0x0f); e8(e, 0xa4); e8(e, 0xc0 | s << 3 | d); e8(e, n); }
static void shrd_r_r_i(E *e, int d, int s, u8 n) { e8(e, 0x0f); e8(e, 0xac); e8(e, 0xc0 | s << 3 | d); e8(e, n); }
/* [base+disp8] forms; base must not be ESP */
static void ld_b(E *e, int r, int base, u8 disp) { e8(e, 0x8b); e8(e, 0x40 | r << 3 | base); e8(e, disp); }
static void st_b(E *e, int base, u8 disp, int r) { e8(e, 0x89); e8(e, 0x40 | r << 3 | base); e8(e, disp); }
static void st_b_i(E *e, int base, u8 disp, u32 imm) { e8(e, 0xc7); e8(e, 0x40 | base); e8(e, disp); e32(e, imm); }
static void movx_b(E *e, u8 op2, int r, int base) { e8(e, 0x0f); e8(e, op2); e8(e, 0x40 | r << 3 | base); e8(e, 0); }
static void movx_m(E *e, u8 op2, int r, u32 disp) { e8(e, 0x0f); e8(e, op2); ebp_disp(e, r, disp); }
static void push_r(E *e, int r) { e8(e, 0x50 + r); }
static void pop_r(E *e, int r) { e8(e, 0x58 + r); }
static void push_i(E *e, u32 v) { e8(e, 0x68); e32(e, v); }
static void call_abs(E *e, const void *fn) {
    e8(e, 0xe8);
    e32(e, (u32)(uintptr_t)fn - (u32)(uintptr_t)(e->p + 4));
}
static u8 *jcc32(E *e, int cc) { e8(e, 0x0f); e8(e, 0x80 | cc); u8 *at = e->p; e32(e, 0); return at; }
static u8 *jmp32(E *e) { e8(e, 0xe9); u8 *at = e->p; e32(e, 0); return at; }
static void patch(E *e, u8 *at, u8 *target) {
    if (!at || e->over || at + 4 > e->end) return;
    u32 rel = (u32)(uintptr_t)target - (u32)(uintptr_t)(at + 4);
    at[0] = (u8)rel; at[1] = (u8)(rel >> 8); at[2] = (u8)(rel >> 16); at[3] = (u8)(rel >> 24);
}
static void add_esp(E *e, u8 n) { e8(e, 0x83); e8(e, 0xc4); e8(e, n); }
static void test_r_r(E *e, int a, int b) { e8(e, 0x85); e8(e, 0xc0 | b << 3 | a); }

enum { CC_E = 4, CC_NE = 5, CC_A = 7 };

/* ================= instruction decoding (64-bit mode) ================= */

typedef struct {
    int len;
    int p66, a67, rep, lock, seg;
    int rex, w, r, x, b;
    int op;                 /* one-byte opcode, or 0x100 | second byte for 0F xx */
    int has_modrm, mod, reg, rm;
    int index, base, scale; /* -1 = none */
    int riprel;
    i32 disp;
    i64 imm;
    int osz;
} Ins;

static int modrm_1(int b) {
    if (b < 0x40) return (b & 7) < 4;
    if (b >= 0x80 && b <= 0x8f) return 1;
    switch (b) {
    case 0x63: case 0x69: case 0x6b: case 0xc0: case 0xc1: case 0xc6: case 0xc7:
    case 0xd0: case 0xd1: case 0xd2: case 0xd3: case 0xf6: case 0xf7: case 0xfe: case 0xff:
        return 1;
    }
    return b >= 0xd8 && b <= 0xdf;
}

static int modrm_2(int b) {
    switch (b) {
    case 0x05: case 0x06: case 0x07: case 0x08: case 0x09: case 0x0b: case 0x0e: case 0x77:
    case 0xa0: case 0xa1: case 0xa2: case 0xa8: case 0xa9: case 0xaa:
        return 0;
    }
    if (b >= 0x30 && b <= 0x37) return 0;
    if (b >= 0x80 && b <= 0x8f) return 0;
    if (b >= 0xc8 && b <= 0xcf) return 0;
    return 1;
}

/* returns 1 and fills `in`, or 0 if the bytes are not decodable here
 * (unknown map, or the instruction does not fit in the `avail` bytes) */
static int dec_m32;     /* decoding 32-bit code (set by translate) */
static int decode(const u8 *p, u32 avail, Ins *in) {
    u32 n = 0;
    e_memset(in, 0, sizeof *in);
    in->seg = -1; in->index = -1; in->base = -1;
    u8 b;
    for (;;) {
        if (n >= avail || n >= 14) return 0;
        b = p[n++];
        if (b == 0x66) in->p66 = 1;
        else if (b == 0x67) in->a67 = 1;
        else if (b == 0xf2 || b == 0xf3) in->rep = b;
        else if (b == 0xf0) in->lock = 1;
        else if (b == 0x64) in->seg = FS;
        else if (b == 0x65) in->seg = GS;
        else if (b == 0x26 || b == 0x2e || b == 0x36 || b == 0x3e) {}
        else break;
    }
    if ((b & 0xf0) == 0x40 && dec_m32) {
        /* INC/DEC r16/r32: decoded as FF /0, /1 with a register operand */
        in->op = 0xff; in->has_modrm = 1; in->mod = 3;
        in->reg = (b >> 3) & 1; in->rm = b & 7;
        in->osz = in->p66 ? 2 : 4;
        in->len = (int)n;
        return 1;
    }
    if ((b & 0xf0) == 0x40) {
        in->rex = b; in->w = (b >> 3) & 1; in->r = (b >> 2) & 1; in->x = (b >> 1) & 1; in->b = b & 1;
        if (n >= avail) return 0;
        b = p[n++];
    }
    if (dec_m32 && in->a67) return 0;   /* 16-bit addressing: not translated */
    in->osz = in->w ? 8 : in->p66 ? 2 : 4;
    int immsz = 0;
    if (b == 0x0f) {
        if (n >= avail) return 0;
        b = p[n++];
        if (b == 0x0f) return 0;
        if ((b == 0x38 || b == 0x3a) && (jit_off & 256)) return 0;
        if (b == 0x38 || b == 0x3a) {
            /* three-byte maps: never translated, but decoded so they stay inside blocks */
            if (n >= avail) return 0;
            in->op = (b == 0x38 ? 0x200 : 0x300) | p[n++];
            in->has_modrm = 1;
            if (b == 0x3a) immsz = 1;
        } else {
        in->op = 0x100 | b;
        in->has_modrm = modrm_2(b);
        if ((b >= 0x70 && b <= 0x73) || b == 0xa4 || b == 0xac || b == 0xba || b == 0xc2 ||
            (b >= 0xc4 && b <= 0xc6)) immsz = 1;
        if (b >= 0x80 && b <= 0x8f) immsz = (dec_m32 && in->p66) ? 2 : 4;
        }
    } else {
        in->op = b;
        in->has_modrm = modrm_1(b);
        int z = in->osz == 2 ? 2 : 4;
        if (b < 0x40 && (b & 7) == 4) immsz = 1;
        else if (b < 0x40 && (b & 7) == 5) immsz = z;
        else switch (b) {
        case 0x68: case 0x69: case 0x81: case 0xa9: case 0xc7: immsz = z; break;
        case 0x6a: case 0x6b: case 0x80: case 0x83: case 0xa8: case 0xc0: case 0xc1: case 0xc6:
        case 0xcd: case 0xeb: immsz = 1; break;
        case 0xc2: case 0xca: immsz = 2; break;
        case 0xc8: immsz = 3; break;
        case 0xe8: case 0xe9: immsz = (dec_m32 && in->p66) ? 2 : 4; break;
        case 0xa0: case 0xa1: case 0xa2: case 0xa3: immsz = dec_m32 ? 4 : in->a67 ? 4 : 8; break;
        case 0x82: immsz = dec_m32 ? 1 : 0; break;
        case 0x9a: case 0xea: immsz = dec_m32 ? (in->p66 ? 4 : 6) : 0; break;
        default:
            if (b >= 0x70 && b <= 0x7f) immsz = 1;
            else if (b >= 0xb0 && b <= 0xb7) immsz = 1;
            else if (b >= 0xb8 && b <= 0xbf) immsz = in->w ? 8 : z;
            else if (b >= 0xe0 && b <= 0xe7) immsz = 1;
            break;
        }
    }
    if (in->has_modrm) {
        if (n >= avail) return 0;
        u8 m = p[n++];
        in->mod = m >> 6;
        in->reg = ((m >> 3) & 7) | (in->r << 3);
        int rm = m & 7;
        if (in->mod == 3) {
            in->rm = rm | (in->b << 3);
        } else {
            int dsz = in->mod == 1 ? 1 : in->mod == 2 ? 4 : 0;
            if (rm == 4) {
                if (n >= avail) return 0;
                u8 sib = p[n++];
                int idx = ((sib >> 3) & 7) | (in->x << 3);
                if (idx != 4) { in->index = idx; in->scale = sib >> 6; }
                int base = sib & 7;
                if (base == 5 && in->mod == 0) dsz = 4;
                else in->base = base | (in->b << 3);
            } else if (rm == 5 && in->mod == 0) {
                in->riprel = !dec_m32;          /* 32-bit code: absolute disp32 */
                dsz = 4;
            } else {
                in->base = rm | (in->b << 3);
            }
            if (n + (u32)dsz > avail) return 0;
            if (dsz == 1) in->disp = (i8)p[n];
            else if (dsz == 4) in->disp = (i32)((u32)p[n] | (u32)p[n + 1] << 8 | (u32)p[n + 2] << 16 | (u32)p[n + 3] << 24);
            n += (u32)dsz;
        }
        /* F6/F7 /0,/1 carry an immediate */
        if (in->op == 0xf6 && (in->reg & 7) < 2) immsz = 1;
        if (in->op == 0xf7 && (in->reg & 7) < 2) immsz = in->osz == 2 ? 2 : 4;
    }
    if (n + (u32)immsz > avail) return 0;
    u64 imm = 0;
    for (int i = 0; i < immsz; i++) imm |= (u64)p[n + i] << (8 * i);
    if (immsz == 1) in->imm = (i8)imm;
    else if (immsz == 2) in->imm = (i16)imm;
    else if (immsz == 4) in->imm = (i32)imm;
    else in->imm = (i64)imm;
    n += (u32)immsz;
    in->len = (int)n;
    return 1;
}

/* Instructions after which the block must end (control transfer, or a change
 * of interrupt/machine state that the run loop must observe). */
static int ends_block(const Ins *in) {
    int op = in->op;
    if (op >= 0x200) return 0;
    if (op >= 0x100) {
        int b = op & 0xff;
        if (b == 0x01 && in->mod == 3 && (in->reg & 7) == 7 && (in->rm & 7) == 0) return 0;   /* SWAPGS */
        if (b <= 0x0b) return b != 0x0d;                 /* 0F 00-0B system */
        if (b == 0x22 || b == 0x23) return 1;            /* MOV to CR/DR (reads change nothing) */
        if (b == 0x30 || (b >= 0x32 && b <= 0x37)) return 1;
        if (b >= 0x80 && b <= 0x8f) return 1;
        if (b == 0xa1 || b == 0xa9) return 1;            /* POP FS/GS */
        return 0;
    }
    if (op >= 0x70 && op <= 0x7f) return 1;
    if (in->rep && op >= 0xa4 && op <= 0xaf) return 1;     /* REP string ops may re-execute */
    if (op >= 0xe0 && op <= 0xef) return 1;              /* LOOP/JCXZ, IN/OUT, CALL/JMP */
    switch (op) {
    case 0x06: case 0x07: case 0x0e: case 0x16: case 0x17: case 0x1e: case 0x1f: case 0x27: case 0x2f:
    case 0x37: case 0x3f: case 0x60: case 0x61: case 0x62: case 0x6c: case 0x6d: case 0x6e: case 0x6f:
    case 0x8e: case 0x9a: case 0x9d: case 0xc2: case 0xc3: case 0xc4: case 0xc5: case 0xca: case 0xcb:
    case 0xcc: case 0xcd: case 0xce: case 0xcf: case 0xd4: case 0xd5: case 0xd6:
    case 0xf4:                                       /* (CLI cannot make an interrupt deliverable; STI: see translate) */
        return 1;
    case 0xff: { int r = in->reg & 7; return r >= 2 && r <= 5; }
    }
    return 0;
}

/* ================= profiling (W32_JITPROF) ================= */

/* Dynamic counts, keyed by (opcode << 3 | modrm.reg), of instructions that the
 * interpreter runs: inside blocks (fallback), on a block's slow path (stub),
 * and outside blocks. Off by default; generated code only counts when on. */
#define PKEYS 0x2000u
int jit_prof_on;
extern u32 legacy_hist_pc[4096], legacy_hist_n[4096];
u64 jit_dump_rip;
u32 jit_off;     /* debug: bit mask of translators to disable (W32_JITOFF) */
u32 jit_off2 = 82; /* more of them (W32_JITOFF2): 1 deferred Jcc flags, 2 base reuse, 4 page-crossing insns, 8 packed SSE helper, 16 hotness threshold, 32 per-site dispatch, 64 hot-block promotion (2, 16, 64 measured slower: off) */
u32 jit_hot = 2;   /* translate code on this execution, interpret it before (W32_JITHOT) */
static u32 prof_fb[PKEYS], prof_stub[PKEYS], prof_out[PKEYS];
static u32 prof_out_why[4];
static u32 prof_blk_exec, prof_c_entries;
static u32 px_link, px_tag, px_disp, px_budget, px_inval, px_interp, px_nolink, px_rethit, px_retmiss, px_xpage;
/* code range -> guest instruction kind (W32_HOSTPROF + W32_JITPROF): starts are increasing */
#define CMAP_MAX (6u << 20)
static u32 *cmap_off; static u16 *cmap_key; static u32 cmap_n;
static u32 cmap_hits[PKEYS + 4];
static u32 cmap_off8b[160], cmap_offentry[64];
static u64 pb_bytes[PKEYS + 2], pb_cnt[PKEYS + 2];
static u32 ph_cond[16 * 8], ph_lfcf[8], ph_shift[8], ph_incdec, ph_adc, ph_pause, ph_sr_chk, ph_sr_fail, ph_sr_acc, ph_defer, ph_jcc_side;
static u32 *pexec;
static u32 *bexec; static u8 *bend, *bninsn, *bmem, *breuse, *breuse2;
void *jit_prof_alloc(u32 n);
__attribute__((weak)) void jit_dump_bytes(const u8 *p, u32 n, u32 key) { (void)p; (void)n; (void)key; }
__attribute__((weak)) void *jit_prof_alloc(u32 n) { (void)n; return 0; }
static void cmap_mark(Jit *j, u8 *p, u32 key) {
    if (!cmap_off || cmap_n >= CMAP_MAX || p >= j->code_end) return;
    cmap_off[cmap_n] = (u32)(p - j->code); cmap_key[cmap_n++] = (u16)key;
}
void jit_prof_sample(Machine *m, uintptr_t pc);
void jit_prof_sample(Machine *m, uintptr_t pc) {
    Jit *j = m->jit;
    if (!j || !cmap_n || pc < (uintptr_t)j->code || pc >= (uintptr_t)j->code_ptr) { cmap_hits[PKEYS + 3]++; return; }
    u32 off = (u32)(pc - (uintptr_t)j->code), lo = 0, hi = cmap_n;
    while (hi - lo > 1) { u32 mid = (lo + hi) / 2; if (cmap_off[mid] <= off) lo = mid; else hi = mid; }
    u32 k = cmap_off[lo] <= off ? cmap_key[lo] : PKEYS + 3;
    cmap_hits[k]++;
    if (k == (0x8bu << 3)) { u32 d = off - cmap_off[lo]; if (d < 160) cmap_off8b[d]++; }
    if (k == PKEYS) { u32 d = off - cmap_off[lo]; if (d < 64) cmap_offentry[d]++; }
}
static void bytes_report(Machine *m) {
    u64 tot = 0;
    for (u32 i = 0; i < PKEYS + 2; i++) tot += pb_bytes[i];
    if (!tot) return;
    emu_log(m, "JIT code bytes by instruction kind (static, %llu bytes):", (unsigned long long)tot);
    static u64 keep[PKEYS + 2];
    for (u32 i = 0; i < PKEYS + 2; i++) keep[i] = pb_bytes[i];
    for (int r = 0; r < 24; r++) {
        u32 best = 0;
        for (u32 i = 1; i < PKEYS + 2; i++) if (pb_bytes[i] > pb_bytes[best]) best = i;
        if (!pb_bytes[best]) break;
        u32 pm = (u32)(pb_bytes[best] * 1000 / tot), avg = (u32)(pb_bytes[best] / (pb_cnt[best] ? pb_cnt[best] : 1));
        if (best == PKEYS + 1) emu_log(m, "  block tail (stubs, cold exits)  %u.%u%%  avg %u B", pm / 10, pm % 10, avg);
        else { u32 op = best >> 3; emu_log(m, "  %s%02x /%u  %u.%u%%  avg %u B", op >= 0x300 ? "0f3a " : op >= 0x200 ? "0f38 " : op >= 0x100 ? "0f " : "", op & 0xff, best & 7, pm / 10, pm % 10, avg); }
        pb_bytes[best] = 0;
    }
    for (u32 i = 0; i < PKEYS + 2; i++) pb_bytes[i] = keep[i];
}
static void cmap_report(Machine *m) {
    bytes_report(m);
    if (bexec) {
        static const char *nm[9] = { "jcc", "jmp same page", "jmp other page", "call", "ret", "indirect", "page end", "64 insns", "other" };
        u64 cnt[9] = { 0 }, ins[9] = { 0 }, tot = 0;
        for (u32 k = 0; k < m->jit->nblocks; k++) { u32 w = bend[k] < 9 ? bend[k] : 8; cnt[w] += bexec[k]; ins[w] += (u64)bexec[k] * bninsn[k]; tot += bexec[k]; }
        for (u32 w = 0; w < 9; w++) if (cnt[w]) emu_log(m, "  blocks ending in %s: %u.%u%% (avg %u insns)", nm[w], (u32)(cnt[w] * 1000 / tot) / 10, (u32)(cnt[w] * 1000 / tot) % 10, (u32)(ins[w] / cnt[w]));
        {
            Jit *jj = m->jit;
            for (int r = 0; r < 12; r++) {
                u32 best = 0; u64 bv = 0;
                for (u32 k = 0; k < jj->nblocks; k++) { u64 v = (u64)bexec[k] * bninsn[k]; if (v > bv) { bv = v; best = k; } }
                if (!bv) break;
                JitBlock *hb = &jj->blocks[best];
                const u8 *hp = hb->host_page + (hb->rip & 0xfff);
                emu_log(m, "  hot block %llx: %u insns x %u = %llu  bytes %02x %02x %02x %02x %02x %02x %02x %02x", (unsigned long long)hb->rip, bninsn[best], bexec[best],
                        (unsigned long long)bv, hp[0], hp[1], hp[2], hp[3], hp[4], hp[5], hp[6], hp[7]);
                bexec[best] = 0;
            }
        }
        {
            u64 nb[5] = { 0 }, ni[5] = { 0 }, ei[5] = { 0 };   /* executed 0, 1, 2-3, 4-15, 16+ times */
            for (u32 k = 0; k < m->jit->nblocks; k++) {
                u32 x = bexec[k], c2 = x == 0 ? 0 : x == 1 ? 1 : x < 4 ? 2 : x < 16 ? 3 : 4;
                nb[c2]++; ni[c2] += bninsn[k]; ei[c2] += (u64)x * bninsn[k];
            }
            static const char *bn[5] = { "0", "1", "2-3", "4-15", "16+" };
            for (int q = 0; q < 5; q++) emu_log(m, "  blocks executed %s times: %llu blocks, %llu insns translated, %llu insns executed", bn[q],
                (unsigned long long)nb[q], (unsigned long long)ni[q], (unsigned long long)ei[q]);
        }
        if (pexec) {
            u64 tot = 0; for (u32 i = 0; i < PKEYS; i++) tot += pexec[i];
            emu_log(m, "  executed instructions by kind (%llu), with static bytes per instance:", (unsigned long long)tot);
            for (int r = 0; r < 30; r++) {
                u32 best = 0; for (u32 i = 1; i < PKEYS; i++) if (pexec[i] > pexec[best]) best = i;
                if (!pexec[best]) break;
                u32 op = best >> 3;
                u32 pm = (u32)((u64)pexec[best] * 10000 / tot);
                emu_log(m, "    %s%02x /%u  %u.%02u%%  %u B", op >= 0x300 ? "0f3a " : op >= 0x200 ? "0f38 " : op >= 0x100 ? "0f " : "", op & 0xff, best & 7,
                        pm / 100, pm % 100, (u32)(pb_bytes[best] / (pb_cnt[best] ? pb_cnt[best] : 1)));
                pexec[best] = 0;
            }
        }
        if (bmem) {
            u64 mm = 0, ru = 0, ru2 = 0;
            for (u32 k = 0; k < m->jit->nblocks; k++) { mm += (u64)bexec[k] * bmem[k]; ru += (u64)bexec[k] * breuse[k]; ru2 += (u64)bexec[k] * breuse2[k]; }
            emu_log(m, "  non-stack memory accesses: %llu, same base as an earlier access: %llu, of which within +-2 KiB: %llu",
                    (unsigned long long)mm, (unsigned long long)ru, (unsigned long long)ru2);
        }
    }
    for (u32 i = 0; i < 128; i++) if (ph_cond[i] > 100000) emu_log(m, "  cpu_cond cc %x lfkind %u: %u", i / 8, i % 8, ph_cond[i]);
    for (u32 i = 0; i < 8; i++) if (ph_lfcf[i]) emu_log(m, "  cpu_lf_cf lfkind %u: %u", i, ph_lfcf[i]);
    for (u32 i = 0; i < 8; i++) if (ph_shift[i]) emu_log(m, "  cpu_shift /%u: %u", i, ph_shift[i]);
    emu_log(m, "  carry-in fetches: inc/dec %u, adc/sbb %u, pause %u", ph_incdec, ph_adc, ph_pause);
    emu_log(m, "  stack region: %u checks, %u failed, %u accesses", ph_sr_chk, ph_sr_fail, ph_sr_acc);
    emu_log(m, "  deferred Jcc flags: %u translated; %u superblock Jcc executed", ph_defer, ph_jcc_side);
    { extern u32 lfcf_hist[16]; for (u32 i = 0; i < 16; i++) if (lfcf_hist[i]) emu_log(m, "  cpu_lf_cf with lf_op %u: %u", i, lfcf_hist[i]); }
    for (u32 d = 0; d < 160; d++) if (cmap_off8b[d]) emu_log(m, "  8b/0 +%u: %u", d, cmap_off8b[d]);
    for (u32 d = 0; d < 64; d++) if (cmap_offentry[d]) emu_log(m, "  entry +%u: %u", d, cmap_offentry[d]);
    u64 tot = 0;
    for (u32 i = 0; i < PKEYS + 4; i++) tot += cmap_hits[i];
    if (!tot) return;
    emu_log(m, "JIT code samples by guest instruction kind (%llu):", (unsigned long long)tot);
    for (int r = 0; r < 30; r++) {
        u32 best = 0;
        for (u32 i = 1; i < PKEYS + 4; i++) if (cmap_hits[i] > cmap_hits[best]) best = i;
        if (!cmap_hits[best]) break;
        const char *nm = best == PKEYS ? "block entry" : best == PKEYS + 1 ? "exits+stubs" : best == PKEYS + 2 ? "shared routines" : best == PKEYS + 3 ? "unmapped" : 0;
        u32 pm = (u32)(cmap_hits[best] * 1000ull / tot);
        if (nm) emu_log(m, "  %s  %u.%u%%", nm, pm / 10, pm % 10);
        else { u32 op = best >> 3; emu_log(m, "  %s%02x /%u  %u.%u%%", op >= 0x300 ? "0f3a " : op >= 0x200 ? "0f38 " : op >= 0x100 ? "0f " : "", op & 0xff, best & 7, pm / 10, pm % 10); }
        cmap_hits[best] = 0;
    }
}
static u64 prof_tr_cycles, prof_tr_bytes, prof_tr_insns, prof_tr_stubs;
static inline u64 rdtsc64(void) { u32 lo, hi; __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi)); return (u64)hi << 32 | lo; }

static u32 raw_key(const u8 *p, u32 avail) {
    u32 n = 0, op;
    while (n < avail && n < 14 && (p[n] == 0x66 || p[n] == 0x67 || p[n] == 0xf2 || p[n] == 0xf3 || p[n] == 0xf0 ||
           p[n] == 0x64 || p[n] == 0x65 || p[n] == 0x26 || p[n] == 0x2e || p[n] == 0x36 || p[n] == 0x3e)) n++;
    if (n < avail && (p[n] & 0xf0) == 0x40) n++;
    if (n >= avail) return 0;
    op = p[n++];
    int mr;
    if (op == 0x0f && n < avail) {
        op = p[n++];
        if ((op == 0x38 || op == 0x3a) && n < avail) { op = (op == 0x38 ? 0x200u : 0x300u) | p[n++]; mr = 1; }
        else { mr = modrm_2((int)op); op |= 0x100; }
    } else mr = modrm_1((int)op);
    u32 reg = (mr && n < avail) ? (p[n] >> 3) & 7 : 0;
    return (op << 3 | reg) & (PKEYS - 1);
}

static void prof_inc_e(E *e, u32 *ctr) {
    if (!jit_prof_on) return;
    e8(e, 0x9c);                                             /* pushfd: the guest's lazy/host flags may be live */
    e8(e, 0xff); e8(e, 0x05); e32(e, (u32)(uintptr_t)ctr);   /* inc dword [ctr] */
    e8(e, 0x9d);                                             /* popfd */
}

void jit_prof_reset(void);
void jit_prof_reset(void) {
    prof_blk_exec = prof_c_entries = 0;
    px_link = px_tag = px_disp = px_budget = px_inval = px_interp = px_nolink = 0;
    { extern u32 stat_tlb_flush, stat_invlpg, stat_walks, stat_cr3; stat_tlb_flush = stat_invlpg = stat_walks = stat_cr3 = 0; }
    e_memset(prof_fb, 0, sizeof prof_fb); e_memset(prof_stub, 0, sizeof prof_stub); e_memset(prof_out, 0, sizeof prof_out);
    e_memset(prof_out_why, 0, sizeof prof_out_why);
}

static void prof_top(Machine *m, const char *what, u32 *a) {
    u64 tot = 0;
    for (u32 i = 0; i < PKEYS; i++) tot += a[i];
    emu_log(m, "JIT prof %s: %llu", what, (unsigned long long)tot);
    if (!tot) return;
    for (int r = 0; r < 25; r++) {
        u32 best = 0;
        for (u32 i = 1; i < PKEYS; i++) if (a[i] > a[best]) best = i;
        if (!a[best]) break;
        u32 op = best >> 3;
        emu_log(m, "  %s%02x /%u  %10u  %2u.%u%%", op >= 0x300 ? "0f3a " : op >= 0x200 ? "0f38 " : op >= 0x100 ? "0f " : "",
                op & 0xff, best & 7, a[best], (u32)(a[best] * 100ull / tot), (u32)(a[best] * 1000ull / tot % 10));
        a[best] = 0;   /* destructive: counts restart after a report */
    }
    e_memset(a, 0, PKEYS * 4);
}

/* ================= translation ================= */

typedef struct {
    E e;
    Jit *j;
    JitBlock *b;            /* block being translated */
    u64 rip, next;
    u64 page;           /* virtual page of the block start */
    u32 pending;            /* instructions of this block before the current one */
    struct Cold { u8 *rel; u64 target; } colds[64]; int ncold;   /* exits whose cold part goes to the block tail */
    struct Side { u8 *jcc_rel; u64 target; u32 executed; int dk, dsz; } sides[64]; int nside;   /* taken Jcc of a superblock */
    u8 *nfix[160]; u32 nfix_k[160]; int nnfix;   /* imm32 fields = N - k (N known at the end) */
    int fuse_sz, fuse_now;  /* previous insn left valid host flags for a register ALU op of this size */
    int flags_dead;         /* this insn's flags are overwritten before anything can observe them */
    u32 pkey;               /* profiling key of the current instruction */
    int lf_known;           /* lazy flags in memory come from a native ALU op: kind << 8 | size (0 = unknown) */
    int lf_prev;            /* lf_known before the current instruction */
    int m32;                /* 32-bit flat code: 32-bit addresses, 4-byte stack slots */
    int nolink;             /* exits go back to C (the dispatcher checks interrupts after an STI shadow) */
    /* register forwarding: guest register fw_reg is also in host fw_lo:fw_hi (fw_hi -1 = zero)
     * at the start of this instruction; fws_* is what this instruction leaves for the next */
    int fw_reg, fw_lo, fw_hi;
    int fws_reg, fws_lo, fws_hi;
    u8 *fw_pos;             /* code position at the start of this instruction: fw_* usable only before anything else is emitted */
    struct Stub { u8 *jumps[8]; int n, end; u64 rip; u32 pending; u8 *ret; u32 pkey; int fw_reg, fw_lo, fw_hi; u8 *tjumps[2]; int tn; u8 *tlb_after; u32 wsz; int br_slot; } stubs[96];
    int nstubs;
    u8 *epilogue;
    /* stack region: accesses at RSP_entry + delta go through the host pointer jit_sr */
    int sr_cur;             /* this instruction's RSP-based access uses the region */
    i32 sr_delta;           /* RSP - RSP_entry before this instruction */
    int sr_pending; i32 sr_off;   /* emit_ea left EDI = jit_sr; emit_tlb adds sr_off */
    /* lazy flags of a register ALU op read only by the next Jcc: stored on its side exit only */
    int can_defer;          /* this instruction may defer (next is a superblock Jcc, flags dead after it) */
    int defer_k, defer_sz;  /* pending: kind (1 ADD, 2 SUB, 3 LOGIC) and size; result EAX:EDX, b ECX:EBX */
    /* base reuse: [B + d] after an earlier access through the unchanged base B in this block */
    int br_ok[16], br_w[16]; i32 br_disp[16];
    int br_mode, br_reg; i32 br_d; int br_emitted;   /* this instruction: 1 first access (record), 2 reuse */
} T;
static int tr_nosr;         /* translating the conservative variant of a block */
static int tr_hot;          /* translating a promoted block into the dense hot arena */

static void set_rip(T *t, u64 v) { mov_m_i(&t->e, OFF(rip), (u32)v); mov_m_i(&t->e, OFF(rip) + 4, (u32)(v >> 32)); }

/* host stack may hold call frames (see t_branch): reset it to the chain entry */
static void emit_epilogue(E *e) {
    mov_m_r(e, OFF(chain_budget), ESI);           /* the chain keeps the budget in ESI */
    mov_r_m(e, ESP, OFF(jit_esp)); pop_r(e, EDI); pop_r(e, ESI); pop_r(e, EBX); pop_r(e, EBP); e8(e, 0xc3);
}
static void epilogue(T *t) { emit_epilogue(&t->e); }

/* push the Cpu * argument of a C helper (EBP is biased) */
static void push_cpu(E *e) { push_r(e, EBP); e8(e, 0x83); e8(e, 0x04); e8(e, 0x24); e8(e, (u8)-CPU_BIAS); }   /* add dword [esp], -128 */
static void call_step(T *t) { push_cpu(&t->e); call_abs(&t->e, (const void *)cpu_step_insn); add_esp(&t->e, 4); }

/* After an interpreted instruction: if it wrote to this block's code (which
 * invalidates the block), leave now. rip and icount are already exact. */
/* Instruction counting: a block claims its N instructions from chain_budget
 * on entry; C derives icount = icount_base - chain_budget when the chain
 * returns. An exit to C before the block's end refunds the instructions it did
 * not execute; interpreter calls first make c->icount exact. */
static void e32_nk(T *t, u32 k) {
    if (t->nnfix >= (int)(sizeof t->nfix / sizeof t->nfix[0])) { t->e.over = 1; return; }
    t->nfix[t->nnfix] = t->e.p; t->nfix_k[t->nnfix++] = k;
    e32(&t->e, 0);
}
static void emit_refund(T *t, u32 executed) {
    E *e = &t->e;
    e8(e, 0x81); e8(e, 0xc6); e32_nk(t, executed);                    /* add esi, N - executed */
}
/* c->icount = exact count before this block's instruction #done (clobbers EAX, EDX) */
static void emit_icount_exact(T *t, u32 done) {
    E *e = &t->e;
    mov_r_m(e, EAX, OFF(icount_base)); mov_r_m(e, EDX, OFF(icount_base) + 4);
    alu_r_r(e, 0x29, EAX, ESI); alu_r_i(e, 3, EDX, 0);                          /* - budget (ESI) */
    e8(e, 0x81); e8(e, 0xe8); e32_nk(t, done);                                   /* sub eax, N - done */
    alu_r_i(e, 3, EDX, 0);
    mov_m_r(e, OFF(icount), EAX); mov_m_r(e, OFF(icount) + 4, EDX);
}


/* Interpreting one instruction from generated code: a call to the shared
 * routine, followed by its record (28 bytes, never executed):
 *   +0 rip (8)  +8 N - done (4)  +12 &block->valid (4)
 *   +16 continuation (4, 0 = leave after it)  +20 N - done - 1 (4)  +24 counter (4, 0 = none)
 * The routine sets rip and the exact icount, counts, calls cpu_step_insn, then
 * continues, or leaves (refunding the rest) if the block was invalidated. */
static void emit_interp(T *t, u64 rip, u32 done, u8 *cont, int leave, u32 *counter) {
    E *e = &t->e;
    e8(e, 0xe8);
    u8 *rel = e->p; e32(e, 0);
    if (!e->over) { u32 r = (u32)(uintptr_t)t->j->interp_stub - (u32)(uintptr_t)(rel + 4); rel[0] = (u8)r; rel[1] = (u8)(r >> 8); rel[2] = (u8)(r >> 16); rel[3] = (u8)(r >> 24); }
    e32(e, (u32)rip); e32(e, (u32)(rip >> 32));
    e32_nk(t, done);
    e32(e, (u32)(uintptr_t)&t->b->valid);
    u8 *c = e->p;
    e32(e, leave ? 0u : (u32)(uintptr_t)cont);
    e32_nk(t, done + 1);
    e32(e, jit_prof_on ? (u32)(uintptr_t)counter : 0u);
    if (!leave && !cont && !e->over) { u32 v = (u32)(uintptr_t)e->p; c[0] = (u8)v; c[1] = (u8)(v >> 8); c[2] = (u8)(v >> 16); c[3] = (u8)(v >> 24); }
}

/* Return to C at rip: call to the shared routine + record {rip (8), chain_from (4)} */
static void emit_exit(T *t, u64 rip, u8 *chain) {
    E *e = &t->e;
    e8(e, 0xe8);
    u8 *rel = e->p; e32(e, 0);
    if (!e->over) { u32 r = (u32)(uintptr_t)t->j->exit_stub - (u32)(uintptr_t)(rel + 4); rel[0] = (u8)r; rel[1] = (u8)(r >> 8); rel[2] = (u8)(r >> 16); rel[3] = (u8)(r >> 24); }
    e32(e, (u32)rip); e32(e, (u32)(rip >> 32));
    e32(e, (u32)(uintptr_t)chain);
}

static void emit_exit_routine(E *e) {
    pop_r(e, EDI);
    ld_b(e, EAX, EDI, 0); mov_m_r(e, OFF(rip), EAX);
    ld_b(e, EAX, EDI, 4); mov_m_r(e, OFF(rip) + 4, EAX);
    ld_b(e, EAX, EDI, 8); mov_m_r(e, OFF(chain_from), EAX);
    emit_epilogue(e);
}

/* EAX = current CF from the lazy flags, natively for the common producers,
 * else via cpu_lf_cf. Clobbers ECX, EDX (like a C call). */
static void emit_lfcf_routine(E *e) {
    u8 *to_slow[8]; int ns = 0;
    movx_m(e, 0xb7, EAX, OFF(lf_op));                                  /* movzx eax, word [lf_op] */
    alu_r_i(e, 7, EAX, LF_LOGIC); u8 *j_zero = jcc32(e, CC_E);
    alu_r_i(e, 7, EAX, LF_INC); u8 *j_c1 = jcc32(e, CC_E);
    alu_r_i(e, 7, EAX, LF_DEC); u8 *j_c2 = jcc32(e, CC_E);
    alu_r_i(e, 7, EAX, LF_MUL); u8 *j_c3 = jcc32(e, CC_E);
    movx_m(e, 0xb7, ECX, OFF(lf_size));
    alu_r_i(e, 7, EAX, LF_SUB); u8 *j_sub = jcc32(e, CC_E);
    alu_r_i(e, 7, EAX, LF_ADD); to_slow[ns++] = jcc32(e, CC_NE);
    /* ADD: CF = res < a */
    alu_r_i(e, 7, ECX, 4); u8 *j_a4 = jcc32(e, CC_E);
    alu_r_i(e, 7, ECX, 8); to_slow[ns++] = jcc32(e, CC_NE);
    mov_r_m(e, EAX, OFF(lf_res) + 4); alu_r_m(e, 0x3b, EAX, OFF(lf_a) + 4);
    u8 *j_ah = jcc32(e, CC_NE);
    patch(e, j_a4, e->p);
    mov_r_m(e, EAX, OFF(lf_res)); alu_r_m(e, 0x3b, EAX, OFF(lf_a));
    patch(e, j_ah, e->p);
    u8 *j_setb1 = jmp32(e);
    /* SUB: CF = a < b */
    patch(e, j_sub, e->p);
    alu_r_i(e, 7, ECX, 4); u8 *j_s4 = jcc32(e, CC_E);
    alu_r_i(e, 7, ECX, 8); to_slow[ns++] = jcc32(e, CC_NE);
    mov_r_m(e, EAX, OFF(lf_a) + 4); alu_r_m(e, 0x3b, EAX, OFF(lf_b) + 4);
    u8 *j_sh = jcc32(e, CC_NE);
    patch(e, j_s4, e->p);
    mov_r_m(e, EAX, OFF(lf_a)); alu_r_m(e, 0x3b, EAX, OFF(lf_b));
    patch(e, j_sh, e->p);
    patch(e, j_setb1, e->p);
    e8(e, 0x0f); e8(e, 0x92); e8(e, 0xc0);                             /* setb al */
    e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc0);                             /* movzx eax, al */
    e8(e, 0xc3);
    patch(e, j_zero, e->p);
    alu_r_r(e, 0x31, EAX, EAX); e8(e, 0xc3);
    patch(e, j_c1, e->p); patch(e, j_c2, e->p); patch(e, j_c3, e->p);
    mov_r_m(e, EAX, OFF(lf_c)); e8(e, 0xc3);
    for (int k = 0; k < ns; k++) patch(e, to_slow[k], e->p);
    push_cpu(e); call_abs(e, (const void *)cpu_lf_cf); add_esp(e, 4);
    e8(e, 0xc3);
}

/* host ZF = guest ZF from the lazy state (result masked to its size, or rflags
 * when nothing is pending). Clobbers EAX, ECX. */
static void emit_lfzf_routine(E *e) {
    movx_m(e, 0xb7, EAX, OFF(lf_op));
    test_r_r(e, EAX, EAX);
    u8 *none = jcc32(e, CC_E);
    movx_m(e, 0xb7, ECX, OFF(lf_size));
    alu_r_i(e, 7, ECX, 8); u8 *s8 = jcc32(e, CC_E);
    alu_r_i(e, 7, ECX, 4); u8 *s4 = jcc32(e, CC_E);
    alu_r_i(e, 7, ECX, 2); u8 *s2 = jcc32(e, CC_E);
    e8(e, 0x80); ebp_disp(e, 7, OFF(lf_res)); e8(e, 0); e8(e, 0xc3);        /* cmp byte [res], 0; ret */
    patch(e, s2, e->p);
    e8(e, 0x66); e8(e, 0x83); ebp_disp(e, 7, OFF(lf_res)); e8(e, 0); e8(e, 0xc3);   /* cmp word [res], 0; ret */
    patch(e, s4, e->p);
    e8(e, 0x83); ebp_disp(e, 7, OFF(lf_res)); e8(e, 0); e8(e, 0xc3);        /* cmp dword [res], 0; ret */
    patch(e, s8, e->p);
    mov_r_m(e, EAX, OFF(lf_res)); alu_r_m(e, 0x0b, EAX, OFF(lf_res) + 4); e8(e, 0xc3);
    patch(e, none, e->p);
    e8(e, 0xf6); ebp_disp(e, 0, OFF(rflags)); e8(e, F_ZF);                   /* test byte [rflags], ZF */
    e8(e, 0x0f); e8(e, 0x94); e8(e, 0xc0);                                   /* sete al: al = !ZFguest */
    test_r_r(e, EAX, EAX);                                                   /* ZF = !al = ZFguest... */
    e8(e, 0xc3);
}

static void emit_interp_routine(E *e) {
    pop_r(e, EDI);                                                     /* record */
    ld_b(e, EAX, EDI, 0); mov_m_r(e, OFF(rip), EAX);
    ld_b(e, EAX, EDI, 4); mov_m_r(e, OFF(rip) + 4, EAX);
    mov_r_m(e, EAX, OFF(icount_base)); mov_r_m(e, EDX, OFF(icount_base) + 4);
    alu_r_r(e, 0x29, EAX, ESI); alu_r_i(e, 3, EDX, 0);                 /* - budget (ESI) */
    e8(e, 0x2b); e8(e, 0x47); e8(e, 8);                                /* sub eax, [edi+8] */
    alu_r_i(e, 3, EDX, 0);
    mov_m_r(e, OFF(icount), EAX); mov_m_r(e, OFF(icount) + 4, EDX);
    ld_b(e, EAX, EDI, 24); test_r_r(e, EAX, EAX);
    e8(e, 0x74); e8(e, 2); e8(e, 0xff); e8(e, 0x00);                   /* jz +2; inc dword [eax] */
    push_cpu(e);
    call_abs(e, (const void *)cpu_step_insn);                          /* ESI, EDI are callee-saved */
    add_esp(e, 4);
    ld_b(e, EAX, EDI, 16); test_r_r(e, EAX, EAX);
    e8(e, 0x74); u8 *leave = e->p; e8(e, 0);                           /* jz leave */
    ld_b(e, ECX, EDI, 12);
    e8(e, 0x80); e8(e, 0x39); e8(e, 0);                                /* cmp byte [ecx], 0 */
    e8(e, 0x74); u8 *inval = e->p; e8(e, 0);                           /* je invalidated */
    e8(e, 0xff); e8(e, 0xe0);                                          /* jmp eax */
    *inval = (u8)(e->p - inval - 1);
    ld_b(e, EAX, EDI, 20);
    alu_r_r(e, 0x01, ESI, EAX);                                        /* refund */
    *leave = (u8)(e->p - leave - 1);
    emit_epilogue(e);
}

static void emit_dispatch(T *t);
static void emit_irq_pending_jumps(T *t, u8 **j1, u8 **j2, u8 **j3);
/* After an interpreted block-ending instruction (rip already set): continue
 * through the jump cache unless the dispatcher must look at the machine
 * (interrupt shadow, halt, fatal error, interrupt request). */
static void emit_post_interp_dispatch(T *t) {
    E *e = &t->e;
    e8(e, 0x83); ebp_disp(e, 7, OFF(shadow)); e8(e, 0); u8 *x1 = jcc32(e, CC_NE);
    e8(e, 0x83); ebp_disp(e, 7, OFF(halted)); e8(e, 0); u8 *x2 = jcc32(e, CC_NE);
    e8(e, 0x83); ebp_disp(e, 7, OFF(fatal)); e8(e, 0); u8 *x3 = jcc32(e, CC_NE);
    e8(e, 0x83); ebp_disp(e, 7, OFF(yield_req)); e8(e, 0); u8 *x9 = jcc32(e, CC_NE);   /* PAUSE yield */
    e8(e, 0x83); ebp_disp(e, 7, OFF(mode64)); e8(e, 1); u8 *x7 = jcc32(e, CC_NE);   /* still 64-bit code */
    u8 *x4, *x5, *x6;
    emit_irq_pending_jumps(t, &x4, &x5, &x6);
    mov_r_m(e, ECX, OFF(rip)); mov_r_m(e, EBX, OFF(rip) + 4);
    int nl = t->nolink; t->nolink = 0;
    emit_dispatch(t);
    t->nolink = nl;
    patch(e, x1, e->p); patch(e, x2, e->p); patch(e, x3, e->p);
    patch(e, x4, e->p); patch(e, x5, e->p); patch(e, x6, e->p); patch(e, x7, e->p); patch(e, x9, e->p);
    epilogue(t);
}

/* Interpret the current instruction (state in memory is current). */
static void fallback(T *t, int end) {
    if (end) prof_inc_e(&t->e, &px_interp);
    if (end && !t->m32 && !(jit_off & 0x800)) {
        emit_interp(t, t->rip, t->pending, 0, 0, &prof_fb[t->pkey]);
        emit_post_interp_dispatch(t);
        return;
    }
    emit_interp(t, t->rip, t->pending, 0, end, &prof_fb[t->pkey]);
}

/* jumps taken when the APIC or a PIC has a request pending (the dispatcher decides) */
static void emit_irq_pending_jumps(T *t, u8 **j1, u8 **j2, u8 **j3) {
    E *e = &t->e;
    Machine *m = t->j->m;
    e8(e, 0x83); ebp_disp(e, 7, OFF(apic_pending)); e8(e, 0);                      /* cmp dword [cpu->apic_pending], 0 */
    *j1 = jcc32(e, CC_NE);
    /* PIC: only unmasked requests matter (Windows masks the legacy PIC; IRR bits linger) */
    for (int k = 0; k < 2; k++) {
        e8(e, 0xa0); e32(e, (u32)(uintptr_t)&m->pic[k].imr);                      /* mov al, [imr] */
        e8(e, 0xf6); e8(e, 0xd0);                                                 /* not al */
        e8(e, 0x84); e8(e, 0x05); e32(e, (u32)(uintptr_t)&m->pic[k].irr);         /* test [irr], al */
        if (k == 0) *j2 = jcc32(e, CC_NE); else *j3 = jcc32(e, CC_NE);
    }
}

static void exit_const_cold(T *t, u64 target, u8 *inline_rel);
/* Exit at the end of the block to a constant target. A direct jmp starts out
 * falling through to a link stub that returns to C; the dispatcher then
 * patches it to the target block's entry (and can reset it to 0 to unlink). */
static void exit_const(T *t, u64 target) {
    E *e = &t->e;
    /* Same-page targets: this block was entered through the current mapping of its
     * page, so a target on that page is the same physical code under any page
     * tables. Other pages: guarded by the exec TLB entry of the target page. */
    if (t->nolink) {
        /* after an STI shadow: back to C only if an interrupt is requested at all */
        u8 *j1, *j2, *j3;
        emit_irq_pending_jumps(t, &j1, &j2, &j3);
        t->nolink = 0;
        exit_const(t, target);
        t->nolink = 1;
        patch(e, j1, e->p); patch(e, j2, e->p); patch(e, j3, e->p);
        prof_inc_e(e, &px_nolink);
        emit_exit(t, target, 0);
        return;
    }
    if (!(jit_off & 0x4000000) && t->ncold < (int)(sizeof t->colds / sizeof t->colds[0])) {
        /* hot path: one jmp; the link stub / page guard is emitted at the block's tail */
        e8(e, 0xe9);
        t->colds[t->ncold].rel = e->p;
        t->colds[t->ncold++].target = target;
        e32(e, 0);
        return;
    }
    exit_const_cold(t, target, 0);
}

/* cold part of an exit: inline_rel = the hot jmp's rel32 (0: emitted in line) */
static void exit_const_cold(T *t, u64 target, u8 *inline_rel) {
    E *e = &t->e;
    if ((target & ~0xfffull) == t->page) {
        u8 *rel = inline_rel;
        if (!rel) { e8(e, 0xe9); rel = e->p; e32(e, 0); }
        prof_inc_e(e, &px_link);
        emit_exit(t, target, rel);
        return;
    } else if (!(jit_off & 131072)) {
        u32 tx = OFF(tlb_x) + (((u32)(target >> 12) & (TLB_SIZE - 1)) << 4);
        mov_r_i(e, EAX, (u32)target & 0xfffff000u);
        alu_r_m(e, 0x0b, EAX, OFF(tlb_mode));
        alu_r_m(e, 0x3b, EAX, tx);                                     /* cmp eax, [tag lo] */
        u8 *o1 = jcc32(e, CC_NE);
        e8(e, 0x81); e8(e, 0xbd); e32(e, tx + 4); e32(e, (u32)(target >> 32));   /* cmp [tag hi], hi */
        u8 *o2 = jcc32(e, CC_NE);
        e8(e, 0x81); e8(e, 0xbd); e32(e, tx + 8);                      /* cmp [host], linked host page */
        u8 *himm = e->p;
        e32(e, 0);
        u8 *o3 = jcc32(e, CC_NE);
        e8(e, 0xe9);                                                   /* himm + 10 */
        e32(e, 0);                                                     /* rel32 at himm + 11: 0 = fall through */
        patch(e, o3, e->p);
        prof_inc_e(e, &px_link);
        emit_exit(t, target, himm);                                    /* (re)link */
        patch(e, o1, e->p); patch(e, o2, e->p);
    }
    prof_inc_e(e, &px_tag);
    emit_exit(t, target, 0);
}

/* the lazy-flag stores a deferring ALU op skipped (EAX:EDX = result, ECX:EBX = b) */
static void emit_deferred_lf(T *t, int k, int sz) {
    E *e = &t->e;
    mov_m_r(e, OFF(lf_res), EAX); if (sz == 8) mov_m_r(e, OFF(lf_res) + 4, EDX);
    if (k != 3) {
        mov_m_r(e, OFF(lf_b), ECX); if (sz == 8) mov_m_r(e, OFF(lf_b) + 4, EBX);
        alu_r_r(e, k == 2 ? 0x01 : 0x29, EAX, ECX);                    /* a = res + b (SUB) or res - b (ADD) */
        if (sz == 8) alu_r_r(e, k == 2 ? 0x11 : 0x19, EDX, EBX);
        mov_m_r(e, OFF(lf_a), EAX); if (sz == 8) mov_m_r(e, OFF(lf_a) + 4, EDX);
    }
    mov_m_i(e, OFF(lf_op), (u32)(k == 1 ? LF_ADD : k == 2 ? LF_SUB : LF_LOGIC) | (u32)sz << 16);
}

static void emit_colds(T *t) {
    E *e = &t->e;
    for (int k = 0; k < t->nside; k++) {
        /* taken side exit: give back the instructions not executed, then leave */
        patch(e, t->sides[k].jcc_rel, e->p);
        if (t->sides[k].dk) emit_deferred_lf(t, t->sides[k].dk, t->sides[k].dsz);
        emit_refund(t, t->sides[k].executed);
        exit_const_cold(t, t->sides[k].target, 0);
    }
    t->nside = 0;
    for (int k = 0; k < t->ncold; k++) {
        patch(e, t->colds[k].rel, e->p);
        if ((t->colds[k].target & ~0xfffull) == t->page) exit_const_cold(t, t->colds[k].target, t->colds[k].rel);
        else exit_const_cold(t, t->colds[k].target, 0);
    }
    t->ncold = 0;
}

/* Slow-path stubs for native memory instructions: run the instruction in the
 * interpreter (it fills the TLB), then continue after it, or leave the block
 * for control transfers. Jumps to the stub are recorded and patched. */
/* jumps: to the interpreter; tjumps: TLB misses of the instruction's one memory
 * access (tlb_after = where the fast path continues), which try a refill first */
typedef struct { u8 *jumps[8]; int n; u8 *tjumps[2]; int tn, ntlb; u8 *tlb_after; u32 wsz; int br_slot; } Slow;

/* Slow paths are emitted out of line at the end of the block, so the hot
 * path of a memory instruction is straight-line code. */
static void add_stub(T *t, Slow *s, int end) {
    if (!s->n && !s->tn) return;
    if (s->n + s->tn > 8) { t->e.over = 1; return; }
    if (t->nstubs >= (int)(sizeof t->stubs / sizeof t->stubs[0])) { t->e.over = 1; return; }
    struct Stub *st = &t->stubs[t->nstubs++];
    for (int i = 0; i < s->n; i++) st->jumps[i] = s->jumps[i];
    st->n = s->n;
    st->end = end;
    st->rip = t->rip;
    st->pending = t->pending;
    st->ret = t->e.p;
    st->pkey = t->pkey;
    st->fw_reg = t->fws_reg; st->fw_lo = t->fws_lo; st->fw_hi = t->fws_hi;
    st->tn = 0;
    st->br_slot = s->br_slot;
    if (s->ntlb == 1 && !(jit_off & 0x8000000)) {
        for (int i = 0; i < s->tn; i++) st->tjumps[i] = s->tjumps[i];
        st->tn = s->tn; st->tlb_after = s->tlb_after; st->wsz = s->wsz;
    } else for (int i = 0; i < s->tn; i++) st->jumps[st->n++] = s->tjumps[i];
}

static void slow_continue(T *t, Slow *s) { add_stub(t, s, 0); }
static void slow_end(T *t, Slow *s) { add_stub(t, s, 1); }

static void emit_stubs(T *t) {
    E *e = &t->e;
    for (int k = 0; k < t->nstubs; k++) {
        struct Stub *st = &t->stubs[k];
        if (st->tn) {
            /* TLB miss: refill and resume the fast path; the interpreter only if that fails */
            for (int i = 0; i < st->tn; i++) patch(e, st->tjumps[i], e->p);
            push_r(e, EDX); push_r(e, EAX);
            push_i(e, st->wsz); push_r(e, EDX); push_r(e, EAX); push_cpu(e);
            call_abs(e, (const void *)cpu_tlb_fill);
            add_esp(e, 16);
            mov_r_r(e, EDI, EAX);
            pop_r(e, EAX); pop_r(e, EDX);
            test_r_r(e, EDI, EDI);
            u8 *fail = jcc32(e, CC_E);
            u8 *fail2 = 0;
            if (st->wsz >> 8) {   /* the write may have invalidated this block (its own page) */
                e8(e, 0x80); e8(e, 0x3d); e32(e, (u32)(uintptr_t)&t->b->valid); e8(e, 0);
                fail2 = jcc32(e, CC_E);
            }
            u8 *j = jmp32(e);
            patch(e, j, st->tlb_after);
            patch(e, fail, e->p);
            if (fail2) patch(e, fail2, e->p);
        }
        for (int i = 0; i < st->n; i++) patch(e, st->jumps[i], e->p);
        if (st->br_slot) {   /* interpreted: the base-reuse slot was not recorded */
            e8(e, 0xc7); ebp_disp(e, 0, OFF(jit_br) + 8u * (u32)(st->br_slot - 1) + 4); e32(e, 0x40000000u);   /* fails every check (|d| <= 1 MiB) */
        }
        if (st->fw_reg >= 0 && !st->end) {
            /* the fast path left a forwarded copy in host registers: reload it */
            emit_interp(t, st->rip, st->pending, 0, 0, &prof_stub[st->pkey]);
            mov_r_m(e, st->fw_lo, R_LO(st->fw_reg));
            if (st->fw_hi >= 0) mov_r_m(e, st->fw_hi, R_HI(st->fw_reg));
            u8 *j = jmp32(e);
            patch(e, j, st->ret);
        } else emit_interp(t, st->rip, st->pending, st->ret, st->end, &prof_stub[st->pkey]);
    }
}

static void ld_reg_fw2(T *t, int n, int lo, int hi, int ok);
/* guest register n -> lo (and hi if hi >= 0): forwarded copy if nothing was emitted yet */
static void ld_reg_fw(T *t, int n, int lo, int hi) { ld_reg_fw2(t, n, lo, hi, t->e.p == t->fw_pos); }
static void ld_reg_fw2(T *t, int n, int lo, int hi, int ok) {
    E *e = &t->e;
    if (ok && n == t->fw_reg && lo != t->fw_hi && (hi < 0 || hi == t->fw_hi || hi != t->fw_lo)) {
        if (lo != t->fw_lo) mov_r_r(e, lo, t->fw_lo);
        if (hi >= 0 && hi != t->fw_hi) { if (t->fw_hi < 0) mov_r_i(e, hi, 0); else mov_r_r(e, hi, t->fw_hi); }
        return;
    }
    mov_r_m(e, lo, R_LO(n));
    if (hi >= 0) mov_r_m(e, hi, R_HI(n));
}
static void fw_set(T *t, int n, int lo, int hi) { t->fws_reg = n; t->fws_lo = lo; t->fws_hi = hi; }

/* effective address of `in` -> EDX:EAX (clobbers ECX, EBX) */
static void emit_ea(T *t, const Ins *in, int with_seg) {
    E *e = &t->e;
    if (t->m32) {
        /* 32-bit address arithmetic; CS/DS/ES/SS are flat (checked at entry) */
        if (in->base >= 0) mov_r_m(e, EAX, R_LO(in->base)); else mov_r_i(e, EAX, 0);
        if (in->index >= 0) {
            mov_r_m(e, ECX, R_LO(in->index));
            if (in->scale) shift_r_i(e, 4, ECX, (u8)in->scale);
            alu_r_r(e, 0x01, EAX, ECX);
        }
        if (in->disp) alu_r_i(e, 0, EAX, (u32)in->disp);
        if (with_seg && (in->seg == FS || in->seg == GS))
            alu_r_m(e, 0x03, EAX, OFF(seg) + (u32)in->seg * (u32)sizeof(Seg) + (u32)__builtin_offsetof(Seg, base));
        mov_r_i(e, EDX, 0);
        return;
    }
    if (t->br_mode == 2) { t->sr_pending = 2; return; }   /* base reuse: the address is not needed */
    if (t->sr_cur && in->base == 4 && in->index < 0 && !in->riprel && !(with_seg && (in->seg == FS || in->seg == GS))) {
        mov_r_m(e, EDI, OFF(jit_sr));
        t->sr_pending = 1;
        prof_inc_e(e, &ph_sr_acc);
        t->sr_off = t->sr_delta + (i32)in->disp;
        return;
    }
    if (in->riprel) {
        u64 a = t->next + (u64)(i64)in->disp;
        mov_r_i(e, EAX, (u32)a);
        mov_r_i(e, EDX, (u32)(a >> 32));
    } else {
        /* load order keeps a forwarded base/index from being overwritten first */
        int ok = e->p == t->fw_pos;
        int base_first = in->base >= 0 && in->base == t->fw_reg && (t->fw_lo == ECX || t->fw_lo == EBX || t->fw_hi == ECX || t->fw_hi == EBX);
        if (in->index >= 0 && !base_first) {
            ld_reg_fw2(t, in->index, ECX, EBX, ok);
            if (in->scale) { shld_r_r_i(e, EBX, ECX, (u8)in->scale); shift_r_i(e, 4, ECX, (u8)in->scale); }
        }
        if (in->base >= 0) ld_reg_fw2(t, in->base, EAX, EDX, ok);
        else { mov_r_i(e, EAX, 0); mov_r_i(e, EDX, 0); }
        if (in->index >= 0 && base_first) {
            ld_reg_fw2(t, in->index, ECX, EBX, ok);
            if (in->scale) { shld_r_r_i(e, EBX, ECX, (u8)in->scale); shift_r_i(e, 4, ECX, (u8)in->scale); }
        }
        if (in->index >= 0) {
            alu_r_r(e, 0x01, EAX, ECX);
            alu_r_r(e, 0x11, EDX, EBX);
        }
        if (in->disp) {
            alu_r_i(e, 0, EAX, (u32)in->disp);
            alu_r_i(e, 2, EDX, in->disp < 0 ? 0xffffffffu : 0);
        }
    }
    if (with_seg && (in->seg == FS || in->seg == GS)) {
        u32 base = OFF(seg) + (u32)in->seg * (u32)sizeof(Seg) + (u32)__builtin_offsetof(Seg, base);
        alu_r_m(e, 0x03, EAX, base);
        alu_r_m(e, 0x13, EDX, base + 4);
    }
}

/* TLB lookup of the linear address in EDX:EAX for an sz-byte access.
 * Hit: EDI = host pointer, EAX/EDX preserved. Miss: jumps recorded in s. */
static int log2sz(int sz) { return sz == 1 ? 0 : sz == 2 ? 1 : sz == 4 ? 2 : sz == 8 ? 3 : 4; }
static void emit_tlb_inline(E *e, int write, int sz, u8 **miss1, u8 **miss2);
/* shared lookup: EDX:EAX = linear address; hit: ZF = 1, EDI = host pointer;
 * miss: ZF = 0. EAX/EDX kept, ECX clobbered. */
static void emit_tlb_routine(E *e, int write, int sz) {
    u8 *m1, *m2;
    emit_tlb_inline(e, write, sz, &m1, &m2);
    e8(e, 0xc3);                                                 /* ret (ZF = 1 from the tag compare) */
    patch(e, m1, e->p); patch(e, m2, e->p);
    e8(e, 0xc3);                                                 /* ret (ZF = 0) */
}

static void emit_tlb(T *t, Slow *s, int write, int sz) {
    E *e = &t->e;
    if (t->sr_pending == 2) {   /* base reuse: same page as the first access through this base? */
        t->sr_pending = 0;
        u32 so = OFF(jit_br) + 8u * (u32)t->br_reg;
        mov_r_m(e, ECX, so + 4);
        if (t->br_d) alu_r_i(e, 0, ECX, (u32)t->br_d);
        alu_r_i(e, 7, ECX, (u32)(4096 - sz));
        s->jumps[s->n++] = jcc32(e, 7);                              /* ja: other page -> interpreter */
        mov_r_m(e, EDI, so);
        if (t->br_d) { e8(e, 0x8d); if (t->br_d >= -128 && t->br_d < 128) { e8(e, 0x7f); e8(e, (u8)t->br_d); } else { e8(e, 0xbf); e32(e, (u32)t->br_d); } }
        t->br_emitted = 1;
        (void)write;
        return;
    }
    if (t->sr_pending) {   /* stack region: EDI = jit_sr, checked at block entry */
        t->sr_pending = 0;
        if (t->sr_off) { e8(e, 0x8d); if (t->sr_off >= -128 && t->sr_off < 128) { e8(e, 0x7f); e8(e, (u8)t->sr_off); } else { e8(e, 0xbf); e32(e, (u32)t->sr_off); } }
        (void)s; (void)write; (void)sz;
        return;
    }
    if (jit_off & 0x2000000) {   /* opt-in: shared lookup routine (smaller code, measured ~2% slower) */
        e8(e, 0xe8);
        u8 *rel = e->p; e32(e, 0);
        patch(e, rel, t->j->tlb_fn[write][log2sz(sz)]);
        s->tjumps[s->tn++] = jcc32(e, CC_NE);
    } else {
        u8 *m1, *m2;
        emit_tlb_inline(e, write, sz, &m1, &m2);
        if (s->tn + 2 > 2) { s->jumps[s->n++] = m1; s->jumps[s->n++] = m2; s->ntlb = 99; return; }
        s->tjumps[s->tn++] = m1;
        s->tjumps[s->tn++] = m2;
    }
    s->ntlb++;
    s->tlb_after = e->p;
    s->wsz = (u32)write << 8 | (u32)sz;
    if (t->br_mode == 1 && s->ntlb == 1) {   /* first access through this base: remember the page */
        u32 so = OFF(jit_br) + 8u * (u32)t->br_reg;
        mov_r_r(e, ECX, EAX); alu_r_i(e, 4, ECX, 0xfff);
        mov_m_r(e, so + 4, ECX); mov_m_r(e, so, EDI);
        s->br_slot = t->br_reg + 1;
        t->br_emitted = 1;
    }
}

static void emit_tlb_inline(E *e, int write, int sz, u8 **miss1, u8 **miss2) {
    u32 tlb = write ? OFF(tlb_w) : OFF(tlb_r);
    _Static_assert((TLB_SIZE - 1) << 4 <= 0xffff0, "index mask below");
    mov_r_r(e, ECX, EAX);
    shift_r_i(e, 5, ECX, 8);
    alu_r_i(e, 4, ECX, (TLB_SIZE - 1) << 4);                     /* ecx = page index * 16 */
    /* tag of the page holding the LAST byte: an access crossing the page misses */
    if (sz == 1) mov_r_r(e, EDI, EAX);
    else if (sz <= 128) { e8(e, 0x8d); e8(e, 0x78); e8(e, (u8)(sz - 1)); }      /* lea edi, [eax+sz-1] */
    else { e8(e, 0x8d); e8(e, 0xb8); e32(e, (u32)(sz - 1)); }
    alu_r_i(e, 4, EDI, 0xfffff000u);
    alu_r_m(e, 0x0b, EDI, OFF(tlb_mode));                        /* or edi, [tlb_mode] */
    e8(e, 0x3b); e8(e, 0xbc); e8(e, 0x0d); e32(e, tlb);          /* cmp edi, [ebp+ecx+tlb] */
    *miss1 = jcc32(e, CC_NE);
    e8(e, 0x3b); e8(e, 0x94); e8(e, 0x0d); e32(e, tlb + 4);      /* cmp edx, [ebp+ecx+tlb+4] */
    *miss2 = jcc32(e, CC_NE);
    e8(e, 0x8b); e8(e, 0xbc); e8(e, 0x0d); e32(e, tlb + 12);     /* mov edi, [ebp+ecx+tlb+12] (addend) */
    e8(e, 0x8d); e8(e, 0x3c); e8(e, 0x07);                       /* lea edi, [edi+eax] (flags kept) */
}


/* guest register -> ECX:EBX (sz 4: EBX = 0) */
static void load_reg_cb(T *t, int n, int sz) {
    if (sz == 8) ld_reg_fw(t, n, ECX, EBX);
    else { ld_reg_fw(t, n, ECX, -1); mov_r_i(&t->e, EBX, 0); }
}

/* lo:hi registers -> guest register (sz 4: high half zero) */
static void store_reg(T *t, int n, int sz, int lo, int hi) {
    mov_m_r(&t->e, R_LO(n), lo);
    if (sz == 8) mov_m_r(&t->e, R_HI(n), hi); else mov_m_i(&t->e, R_HI(n), 0);
}
/* same, when the caller guarantees hi == 0 for sz 4 (shorter: register store) */
static void store_reg_hz(T *t, int n, int sz, int lo, int hi) {
    mov_m_r(&t->e, R_LO(n), lo);
    mov_m_r(&t->e, R_HI(n), hi);
    (void)sz;
}

static int mem_ok(const Ins *in) { return !in->a67 && !in->lock; }

/* MOV 89/8B/C7/B8+r, LEA */
static int t_mov16(T *t, const Ins *in) {
    E *e = &t->e;
    Slow s = {0};
    if (in->op == 0xc7 && (in->reg & 7) != 0) return 0;
    if (in->mod == 3) {
        if (in->op == 0xc7) { e8(e, 0x66); e8(e, 0xc7); e8(e, 0x85); e32(e, R_LO(in->rm)); e8(e, (u32)in->imm); e8(e, (u32)in->imm >> 8); return 1; }
        int src = in->op == 0x89 ? in->reg : in->rm, dst = in->op == 0x89 ? in->rm : in->reg;
        e8(e, 0x66); mov_r_m(e, EAX, R_LO(src));
        e8(e, 0x66); mov_m_r(e, R_LO(dst), EAX);
        return 1;
    }
    if (!mem_ok(in)) return 0;
    emit_ea(t, in, 1);
    emit_tlb(t, &s, in->op != 0x8b, 2);
    if (in->op == 0xc7) { e8(e, 0x66); e8(e, 0xc7); e8(e, 0x47); e8(e, 0); e8(e, (u32)in->imm); e8(e, (u32)in->imm >> 8); }
    else if (in->op == 0x89) { e8(e, 0x66); mov_r_m(e, ECX, R_LO(in->reg)); e8(e, 0x66); st_b(e, EDI, 0, ECX); }
    else { e8(e, 0x66); ld_b(e, ECX, EDI, 0); e8(e, 0x66); mov_m_r(e, R_LO(in->reg), ECX); }
    slow_continue(t, &s);
    return 1;
}

static int t_mov(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (sz == 2 && !in->rep && in->op != 0x8d && (in->op < 0xb8 || in->op == 0xc7)) return (jit_off & 16) ? 0 : t_mov16(t, in);
    if (sz == 2 && in->op == 0x8d && in->mod != 3 && !in->rep) {   /* 16-bit LEA: the low word */
        emit_ea(t, in, 0);
        e8(e, 0x66); mov_m_r(e, R_LO(in->reg), EAX);
        return 1;
    }
    if (sz < 4 || in->rep) return 0;
    Slow s = {0};
    switch (in->op) {
    case 0x89: case 0x8b: {
        int store = in->op == 0x89;
        if (in->mod == 3) {
            int src = store ? in->reg : in->rm, dst = store ? in->rm : in->reg;
            load_reg_cb(t, src, sz);                            /* sz 4: EBX = 0 */
            store_reg_hz(t, dst, sz, ECX, EBX);
            fw_set(t, dst, ECX, EBX);
            return 1;
        }
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, store, sz);
        if (store) {
            mov_r_m(e, ECX, R_LO(in->reg));
            st_b(e, EDI, 0, ECX);
            if (sz == 8) { mov_r_m(e, ECX, R_HI(in->reg)); st_b(e, EDI, 4, ECX); }
        } else {
            ld_b(e, ECX, EDI, 0);
            if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
            store_reg_hz(t, in->reg, sz, ECX, EBX);
            fw_set(t, in->reg, ECX, EBX);
        }
        slow_continue(t, &s);
        return 1;
    }
    case 0xc7: {
        if ((in->reg & 7) != 0) return 0;
        u32 lo = (u32)in->imm, hi = sz == 8 ? (u32)((u64)in->imm >> 32) : 0;
        if (in->mod == 3) {
            mov_m_i(e, R_LO(in->rm), lo);
            mov_m_i(e, R_HI(in->rm), hi);
            return 1;
        }
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        st_b_i(e, EDI, 0, lo);
        if (sz == 8) st_b_i(e, EDI, 4, hi);
        slow_continue(t, &s);
        return 1;
    }
    case 0x8d:
        if (in->mod == 3) return 0;
        emit_ea(t, in, 0);
        if (in->a67) mov_r_i(e, EDX, 0);                       /* 32-bit address: the low half */
        store_reg(t, in->reg, sz, EAX, EDX);
        fw_set(t, in->reg, EAX, sz == 8 ? EDX : -1);
        return 1;
    default: {  /* B8+r */
        int n = (in->op & 7) | (in->b << 3);
        mov_m_i(e, R_LO(n), (u32)in->imm);
        mov_m_i(e, R_HI(n), sz == 8 ? (u32)((u64)in->imm >> 32) : 0);
        return 1;
    }
    }
}

/* ALU: ADD OR AND SUB XOR CMP (and TEST) on 32/64-bit operands */
/* byte register forms on [ebp+disp32] */
static void st8_m_r(E *e, u32 disp, int r) { e8(e, 0x88); ebp_disp(e, r, disp); }
static void st8_m_i(E *e, u32 disp, u8 v) { e8(e, 0xc6); e8(e, 0x85); e32(e, disp); e8(e, v); }

/* guest 8-bit register n -> [ebp+disp] (AH..BH without REX) */
static u32 r8_off(const Ins *in, int n) {
    if (!in->rex && n >= 4 && n < 8) return R_LO(n - 4) + 1;
    return R_LO(n);
}

/* sized guest register / memory accessors (low half only for 64-bit) */
static void ld_gpr(T *t, const Ins *in, int dst, int n, int sz) {
    if (sz == 1) movx_m(&t->e, 0xb6, dst, r8_off(in, n));
    else if (sz == 2) movx_m(&t->e, 0xb7, dst, R_LO(n));
    else mov_r_m(&t->e, dst, R_LO(n));
}
static void ld_hmem(T *t, int dst, int sz) {
    if (sz == 1) movx_b(&t->e, 0xb6, dst, EDI);
    else if (sz == 2) movx_b(&t->e, 0xb7, dst, EDI);
    else ld_b(&t->e, dst, EDI, 0);
}
static void st_gpr(T *t, const Ins *in, int n, int sz, int lo, int hi) {
    E *e = &t->e;
    if (sz == 1) st8_m_r(e, r8_off(in, n), lo);
    else if (sz == 2) { e8(e, 0x66); mov_m_r(e, R_LO(n), lo); }
    else store_reg(t, n, sz, lo, hi);
}
static void st_hmem(T *t, int sz, int lo, int hi) {
    E *e = &t->e;
    if (sz == 1) { e8(e, 0x88); e8(e, 0x40 | lo << 3 | EDI); e8(e, 0); }
    else if (sz == 2) { e8(e, 0x66); st_b(e, EDI, 0, lo); }
    else { st_b(e, EDI, 0, lo); if (sz == 8) st_b(e, EDI, 4, hi); }
}

static void emit_cf_eax(T *t);

/* ALU: ADD OR AND SUB XOR CMP TEST on 8/16/32/64-bit operands */
static int t_alu(T *t, const Ins *in) {
    E *e = &t->e;
    int op = in->op, ext, form;   /* form: 0 r/m,reg  1 reg,r/m  2 r/m,imm  3 rAX,imm */
    int sz = in->osz;
    if (in->rep) return 0;
    if (op < 0x40) {
        ext = op >> 3;
        int f = op & 7;
        form = (f == 0 || f == 1) ? 0 : (f == 2 || f == 3) ? 1 : 3;
        if (!(f & 1)) sz = 1;
    } else if (op == 0x80 || op == 0x81 || op == 0x83) { ext = in->reg & 7; form = 2; if (op == 0x80) sz = 1; }
    else if (op == 0xf6 || op == 0xf7) { ext = 4; form = 2; if (op == 0xf6) sz = 1; }   /* TEST r/m, imm */
    else if (op == 0xa8 || op == 0xa9) { ext = 4; form = 3; if (op == 0xa8) sz = 1; }   /* TEST rAX, imm */
    else { ext = 4; form = 0; if (op == 0x84) sz = 1; }   /* 0x84/0x85 TEST */
    int is_test = op == 0x84 || op == 0x85 || op == 0xf6 || op == 0xf7 || op == 0xa8 || op == 0xa9;
    int writes = ext != 7 && !is_test;
    int dst_mem = (form == 0 || form == 2) && in->mod != 3;
    int src_mem = form == 1 && in->mod != 3;
    /* LOCK: a single CPU makes every read-modify-write atomic (as the interpreter does) */
    if (in->lock && (!(dst_mem && writes) || (jit_off & 8))) return 0;
    if ((dst_mem || src_mem) && in->a67) return 0;
    int carry = ext == 2 || ext == 3;   /* ADC/SBB: CF of the current lazy state, via the helper */
    if (carry && (jit_off & 4)) return 0;
    /* right after a native ALU op on registers, the host CF is the guest CF: keep it
     * (jit_tmp = CF; everything up to the adc/sbb below is flag-neutral moves) */
    int hostcf = carry && t->fuse_now && !dst_mem && !src_mem && !(jit_off & 0x400000);
    if (hostcf) { mov_m_i(e, OFF(jit_tmp), 0); e8(e, 0x0f); e8(e, 0x92); e8(e, 0x85); e32(e, OFF(jit_tmp)); g_keepflags = 1; }   /* setc byte [jit_tmp] (CF kept) */
    else if (carry) {
        prof_inc_e(e, &ph_adc);
        emit_cf_eax(t);
        mov_m_r(e, OFF(jit_tmp), EAX);
    }
    Slow s = {0};
    if (dst_mem || src_mem) {
        emit_ea(t, in, 1);
        emit_tlb(t, &s, dst_mem && writes, sz);
    }
    int dn = form == 1 ? in->reg : form == 3 ? 0 : in->rm;
    /* a -> EAX:EDX, b -> ECX:EBX */
    if (dst_mem) { ld_hmem(t, EAX, sz); if (sz == 8) ld_b(e, EDX, EDI, 4); }
    else { ld_gpr(t, in, EAX, dn, sz); if (sz == 8) mov_r_m(e, EDX, R_HI(dn)); }
    if (form >= 2) {
        u64 imm = (u64)in->imm & MASK[sz];
        mov_r_i(e, ECX, (u32)imm);
        if (sz == 8) mov_r_i(e, EBX, (u32)(imm >> 32));
    } else if (src_mem) {
        ld_hmem(t, ECX, sz); if (sz == 8) ld_b(e, EBX, EDI, 4);
    } else {
        int n = form == 1 ? in->rm : in->reg;
        ld_gpr(t, in, ECX, n, sz); if (sz == 8) mov_r_m(e, EBX, R_HI(n));
    }
    if (sz != 8) { mov_r_i(e, EDX, 0); mov_r_i(e, EBX, 0); }
    int logic = ext == 1 || ext == 4 || ext == 6;
    int defer = 0;
    if (t->can_defer && !t->flags_dead && !dst_mem && !src_mem && !carry && !(jit_off2 & 1)) {
        int cc = t->can_defer & 15;
        defer = sz != 8 || cc <= 3 || cc == 8 || cc == 9 || cc == 12 || cc == 13;   /* the Jcc tests host flags directly */
    }
    if (!t->flags_dead && !logic && !defer) {   /* LOGIC flags depend on the result only */
        mov_m_r(e, OFF(lf_a), EAX); if (sz == 8) mov_m_r(e, OFF(lf_a) + 4, EDX);
        mov_m_r(e, OFF(lf_b), ECX); if (sz == 8) mov_m_r(e, OFF(lf_b) + 4, EBX);
    }
    if (carry && !hostcf) {
        e8(e, 0x0f); e8(e, 0xba); e8(e, 0xa5); e32(e, OFF(jit_tmp)); e8(e, 0);   /* bt dword [jit_tmp], 0: CF = carry in */
    }
    g_keepflags = 0;
    static const u8 lo_op[8] = { 0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x29 };
    static const u8 hi_op[8] = { 0x11, 0x09, 0x11, 0x19, 0x21, 0x19, 0x31, 0x19 };
    if (sz == 1) alu_r_r(e, (u8)(lo_op[ext] - 1), EAX, ECX);           /* 8-bit form: op al, cl */
    else {
        if (sz == 2) e8(e, 0x66);
        alu_r_r(e, lo_op[ext], EAX, ECX);
        if (sz == 8) alu_r_r(e, hi_op[ext], EDX, EBX);
    }
    if (sz == 1) { e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc0); }            /* movzx eax, al (flags kept) */
    if (sz == 2) { e8(e, 0x0f); e8(e, 0xb7); e8(e, 0xc0); }            /* movzx eax, ax */
    if (defer) {
        ph_defer++;                                    /* static count (a counter increment here would change the host flags) */
        t->defer_k = ext == 0 ? 1 : (ext == 5 || ext == 7) ? 2 : 3;
        t->defer_sz = sz;
    } else if (!t->flags_dead) {
        mov_m_r(e, OFF(lf_res), EAX); if (sz == 8) mov_m_r(e, OFF(lf_res) + 4, EDX);
        int kind = ext == 0 ? 1 : (ext == 5 || ext == 7) ? 2 : (ext == 2 || ext == 3) ? 0 : 3;
        if (!(kind && t->lf_prev == (kind << 8 | sz) && !(jit_off & 0x10000000)))   /* lf_op/lf_size already hold this */
            mov_m_i(e, OFF(lf_op), (u32)(ext == 0 ? LF_ADD : ext == 2 ? LF_ADC : ext == 3 ? LF_SBB : (ext == 5 || ext == 7) ? LF_SUB : LF_LOGIC) | (u32)((u32)sz) << 16);
        if (carry) { mov_r_m(e, ECX, OFF(jit_tmp)); mov_m_r(e, OFF(lf_c), ECX); }
    }
    if (writes) {
        if (dst_mem) st_hmem(t, sz, EAX, EDX);
        else if (sz >= 4) store_reg_hz(t, dn, sz, EAX, EDX);   /* sz 4: EDX = 0 */
        else st_gpr(t, in, dn, sz, EAX, EDX);
        if (!dst_mem && sz >= 4) fw_set(t, dn, EAX, EDX);
    }
    /* only MOVs since the ALU op: host flags (and EAX:EDX = result) are still live */
    if (!s.n && !s.tn) t->fuse_sz = sz;
    if (!t->flags_dead && !carry && !defer) t->lf_known = (ext == 0 ? 1 : (ext == 5 || ext == 7) ? 2 : 3) << 8 | sz;
    slow_continue(t, &s);
    return 1;
}

u32 cpu_lf_cf(Cpu *c);

/* INC/DEC r/m (FE/FF /0 /1): CF is preserved, so the current CF is fetched
 * from the lazy state first unless the flags are dead */
static int t_incdec(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->op == 0xfe ? 1 : in->osz, dec = (in->reg & 7) == 1;
    if (in->rep || (in->lock && in->mod == 3)) return 0;
    int mem = in->mod != 3;
    if (mem && in->a67) return 0;
    int live = !t->flags_dead;
    if (live) {
        prof_inc_e(e, &ph_incdec);
        emit_cf_eax(t);
        mov_m_r(e, OFF(jit_tmp), EAX);
    }
    Slow s = {0};
    if (mem) {
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_hmem(t, EAX, sz);
        if (sz == 8) ld_b(e, EDX, EDI, 4);
    } else {
        ld_gpr(t, in, EAX, in->rm, sz);
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->rm));
    }
    if (sz != 8) mov_r_i(e, EDX, 0);
    if (live) {
        mov_m_r(e, OFF(lf_a), EAX); if (sz == 8) mov_m_r(e, OFF(lf_a) + 4, EDX);
        mov_m_i(e, OFF(lf_b), 1); if (sz == 8) mov_m_i(e, OFF(lf_b) + 4, 0);
    }
    if (sz == 1) { e8(e, 0xfe); e8(e, dec ? 0xc8 : 0xc0); e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc0); }   /* inc/dec al; movzx */
    else if (sz == 2) { e8(e, 0x66); e8(e, dec ? 0x48 : 0x40); e8(e, 0x0f); e8(e, 0xb7); e8(e, 0xc0); }
    else if (sz == 4) { e8(e, dec ? 0x48 : 0x40); }                                               /* inc/dec eax */
    else { alu_r_i(e, dec ? 5 : 0, EAX, 1); alu_r_i(e, dec ? 3 : 2, EDX, 0); }
    if (live) {
        mov_m_r(e, OFF(lf_res), EAX); if (sz == 8) mov_m_r(e, OFF(lf_res) + 4, EDX);
        mov_m_i(e, OFF(lf_op), (u32)(dec ? LF_DEC : LF_INC) | (u32)((u32)sz) << 16);
        mov_r_m(e, ECX, OFF(jit_tmp)); mov_m_r(e, OFF(lf_c), ECX);
        t->lf_known = 6 << 8 | sz;
    }
    if (mem) st_hmem(t, sz, EAX, EDX);
    else st_gpr(t, in, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* BT/BTS/BTR/BTC: 0F BA /4-7 imm8 (register or memory), 0F A3/AB/B3/BB with a
 * register destination. CF = the bit; the other flags are kept (helper). */
static int t_bt(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, op, var = in->op != 0x1ba;
    if (sz < 4 || in->rep) return 0;
    if (var) op = in->op == 0x1a3 ? 0 : in->op == 0x1ab ? 1 : in->op == 0x1b3 ? 2 : 3;
    else { if ((in->reg & 7) < 4) return 0; op = (in->reg & 7) - 4; }
    int mem = in->mod != 3;
    if (in->lock && (!mem || !op)) return 0;
    if (mem && in->a67) return 0;
    Slow s = {0};
    if (mem) {
        emit_ea(t, in, 1);
        if (var) {
            /* a register bit offset addresses a bit string: EA += (offset >> log2(bits)) * size */
            mov_r_m(e, ECX, R_LO(in->reg));
            if (sz == 8) {
                mov_r_m(e, EBX, R_HI(in->reg));
                shrd_r_r_i(e, ECX, EBX, 6); shift_r_i(e, 7, EBX, 6);
                shld_r_r_i(e, EBX, ECX, 3); shift_r_i(e, 4, ECX, 3);
            } else {
                shift_r_i(e, 7, ECX, 5); shift_r_i(e, 4, ECX, 2);
                mov_r_r(e, EBX, ECX); shift_r_i(e, 7, EBX, 31);
            }
            alu_r_r(e, 0x01, EAX, ECX); alu_r_r(e, 0x11, EDX, EBX);
        }
        emit_tlb(t, &s, op != 0, sz);
        ld_b(e, EAX, EDI, 0);
        if (sz == 8) ld_b(e, EDX, EDI, 4);
    } else {
        mov_r_m(e, EAX, R_LO(in->rm));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->rm));
    }
    static const u8 bt_rr[4] = { 0xa3, 0xab, 0xb3, 0xbb };
    if (!var) {
        u32 n = (u32)in->imm & (u32)(sz * 8 - 1);
        int r = n >= 32 ? EDX : EAX;
        e8(e, 0x0f); e8(e, 0xba); e8(e, 0xc0 | (4 + op) << 3 | r); e8(e, n & 31);   /* bt* r, n */
        e8(e, 0x0f); e8(e, 0x92); e8(e, 0xc3);                                       /* setc bl */
    } else {
        mov_r_m(e, ECX, R_LO(in->reg));
        alu_r_i(e, 4, ECX, (u32)(sz * 8 - 1));
        if (sz == 8) {
            alu_r_i(e, 4, ECX, 31);
            e8(e, 0xf6); e8(e, 0x85); e32(e, R_LO(in->reg)); e8(e, 32); e8(e, 0x74); e8(e, 1); e8(e, 0x92);   /* test byte [reg],32; jz +1; xchg eax,edx */
        }
        e8(e, 0x0f); e8(e, bt_rr[op]); e8(e, 0xc0 | ECX << 3 | EAX);                 /* bt* eax, ecx */
        e8(e, 0x0f); e8(e, 0x92); e8(e, 0xc3);                                       /* setc bl */
        if (sz == 8) { e8(e, 0xf6); e8(e, 0x85); e32(e, R_LO(in->reg)); e8(e, 32); e8(e, 0x74); e8(e, 1); e8(e, 0x92); }
    }
    if (op) {
        if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
        else store_reg(t, in->rm, sz, EAX, EDX);
    }
    if (!t->flags_dead) {
        e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xdb);                                       /* movzx ebx, bl */
        push_r(e, EBX); push_cpu(e);
        call_abs(e, (const void *)cpu_bt_flags);
        add_esp(e, 8);
        t->lf_known = 5 << 8;                                                        /* flags now in rflags */
    }
    slow_continue(t, &s);
    return 1;
}

/* CMPXCHG r/m, r (0F B1) and XADD r/m, r (0F C1), 32/64-bit, LOCK allowed on memory.
 * Mirrors the interpreter: CMPXCHG writes memory back even on mismatch. */
static int t_cmpxchg_xadd(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, xadd = in->op == 0x1c1, mem = in->mod != 3;
    if (sz < 4 || in->rep || (in->lock && !mem) || (mem && in->a67)) return 0;
    Slow s = {0};
    if (mem) {
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_b(e, ECX, EDI, 0);
        if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    } else load_reg_cb(t, in->rm, sz);                      /* ECX:EBX = destination */
    int live = !t->flags_dead;
    if (!xadd) {
        mov_r_m(e, EAX, R_LO(0));
        if (sz == 8) mov_r_m(e, EDX, R_HI(0)); else mov_r_i(e, EDX, 0);   /* EAX:EDX = accumulator */
        if (live) {
            mov_m_r(e, OFF(lf_a), EAX); mov_m_r(e, OFF(lf_a) + 4, EDX);
            mov_m_r(e, OFF(lf_b), ECX); mov_m_r(e, OFF(lf_b) + 4, EBX);
            push_r(e, EAX); push_r(e, EDX);
            alu_r_r(e, 0x29, EAX, ECX); mov_m_r(e, OFF(lf_res), EAX);
            if (sz == 8) { alu_r_r(e, 0x19, EDX, EBX); mov_m_r(e, OFF(lf_res) + 4, EDX); }
            else mov_m_i(e, OFF(lf_res) + 4, 0);
            pop_r(e, EDX); pop_r(e, EAX);
            mov_m_i(e, OFF(lf_op), (u32)(LF_SUB) | (u32)((u32)sz) << 16);
            t->lf_known = 2 << 8 | sz;
        }
        alu_r_r(e, 0x39, EAX, ECX);                          /* cmp eax, ecx */
        u8 *ne1 = jcc32(e, CC_NE), *ne2 = 0;
        if (sz == 8) { alu_r_r(e, 0x39, EDX, EBX); ne2 = jcc32(e, CC_NE); }
        /* equal: destination = source register */
        mov_r_m(e, EAX, R_LO(in->reg));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->reg)); else mov_r_i(e, EDX, 0);
        if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
        else store_reg(t, in->rm, sz, EAX, EDX);
        u8 *done = jmp32(e);
        patch(e, ne1, e->p); if (ne2) patch(e, ne2, e->p);
        /* not equal: memory rewritten with its value, accumulator = destination */
        if (mem) { st_b(e, EDI, 0, ECX); if (sz == 8) st_b(e, EDI, 4, EBX); }
        store_reg(t, 0, sz, ECX, EBX);
        patch(e, done, e->p);
    } else {
        mov_r_r(e, EAX, ECX); mov_r_r(e, EDX, EBX);          /* EAX:EDX = destination */
        mov_r_m(e, ECX, R_LO(in->reg));
        if (sz == 8) mov_r_m(e, EBX, R_HI(in->reg)); else mov_r_i(e, EBX, 0);
        if (live) {
            mov_m_r(e, OFF(lf_a), EAX); mov_m_r(e, OFF(lf_a) + 4, EDX);
            mov_m_r(e, OFF(lf_b), ECX); mov_m_r(e, OFF(lf_b) + 4, EBX);
        }
        push_r(e, EDX); push_r(e, EAX);                      /* old destination */
        alu_r_r(e, 0x01, EAX, ECX);
        if (sz == 8) alu_r_r(e, 0x11, EDX, EBX); else mov_r_i(e, EDX, 0);
        if (live) {
            mov_m_r(e, OFF(lf_res), EAX); mov_m_r(e, OFF(lf_res) + 4, EDX);
            mov_m_i(e, OFF(lf_op), (u32)(LF_ADD) | (u32)((u32)sz) << 16);
            t->lf_known = 1 << 8 | sz;
        }
        pop_r(e, ECX); pop_r(e, EBX);                        /* ECX:EBX = old destination */
        if (mem) {
            st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX);
            store_reg(t, in->reg, sz, ECX, EBX);
        } else {
            store_reg(t, in->reg, sz, ECX, EBX);             /* source <- destination, then the sum */
            store_reg(t, in->rm, sz, EAX, EDX);
        }
    }
    slow_continue(t, &s);
    return 1;
}

static void emit_bswap386(E *e, int r) {
    e8(e, 0x66); e8(e, 0xc1); e8(e, 0xc8 | r); e8(e, 8);    /* ror r16, 8 */
    e8(e, 0xc1); e8(e, 0xc8 | r); e8(e, 16);                 /* ror r32, 16 */
    e8(e, 0x66); e8(e, 0xc1); e8(e, 0xc8 | r); e8(e, 8);    /* ror r16, 8 */
}

/* BSWAP r32/r64 */
static int t_bswap(T *t, const Ins *in) {
    E *e = &t->e;
    int n = (in->op & 7) | (in->b << 3);
    if (in->osz < 4) return 0;
    /* (the host BSWAP instruction is 486+: byte-swap with i386 rotates) */
    mov_r_m(e, EAX, R_LO(n));
    emit_bswap386(e, EAX);
    if (in->osz == 4) { mov_m_r(e, R_LO(n), EAX); mov_m_i(e, R_HI(n), 0); return 1; }
    mov_r_m(e, EDX, R_HI(n));
    emit_bswap386(e, EDX);
    mov_m_r(e, R_LO(n), EDX); mov_m_r(e, R_HI(n), EAX);
    return 1;
}

/* BSF/BSR (and TZCNT/LZCNT, which decode as BSF/BSR here, like the interpreter):
 * zero source: destination unchanged, ZF = 1; else all arithmetic flags 0 */
static int t_bsf(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, bsr = in->op == 0x1bd, mem = in->mod != 3;
    if (sz < 4 || in->lock || (mem && in->a67)) return 0;
    Slow s = {0};
    if (mem) {
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 0, sz);
        ld_b(e, EAX, EDI, 0);
        if (sz == 8) ld_b(e, EDX, EDI, 4); else mov_r_i(e, EDX, 0);
    } else {
        mov_r_m(e, EAX, R_LO(in->rm));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->rm)); else mov_r_i(e, EDX, 0);
    }
    mov_r_r(e, ECX, EAX); alu_r_r(e, 0x09, ECX, EDX);
    u8 *zero = jcc32(e, CC_E);
    int first = bsr ? EDX : EAX, second = bsr ? EAX : EDX;
    test_r_r(e, first, first);
    u8 *use2 = jcc32(e, CC_E);
    e8(e, 0x0f); e8(e, bsr ? 0xbd : 0xbc); e8(e, 0xc0 | ECX << 3 | first);   /* bs? ecx, first */
    if (bsr) alu_r_i(e, 0, ECX, 32);
    u8 *have = jmp32(e);
    patch(e, use2, e->p);
    e8(e, 0x0f); e8(e, bsr ? 0xbd : 0xbc); e8(e, 0xc0 | ECX << 3 | second);
    if (!bsr) alu_r_i(e, 0, ECX, 32);
    patch(e, have, e->p);
    mov_m_r(e, R_LO(in->reg), ECX); mov_m_i(e, R_HI(in->reg), 0);
    if (!t->flags_dead) { alu_m_i(e, 4, OFF(rflags), ~(u32)F_ARITH); alu_m_i(e, 1, OFF(rflags), 2); mov_m_i(e, OFF(lf_op), LF_NONE); }
    u8 *done = jmp32(e);
    patch(e, zero, e->p);
    if (!t->flags_dead) { alu_m_i(e, 4, OFF(rflags), ~(u32)F_ARITH); alu_m_i(e, 1, OFF(rflags), 2 | F_ZF); mov_m_i(e, OFF(lf_op), LF_NONE); }
    patch(e, done, e->p);
    if (!t->flags_dead) t->lf_known = 5 << 8;
    slow_continue(t, &s);
    return 1;
}

/* Ring-0 system instructions that change no flags and cannot make an interrupt
 * deliverable: CLI, MOV r64 <- CR0/2/3/4/8, SWAPGS. CPL != 0 takes the slow
 * path (the interpreter raises #GP). */
void cpu_jit_set_tpr(Cpu *c);
static int t_sys(T *t, const Ins *in) {
    E *e = &t->e;
    int op = in->op;
    if (in->rep || in->lock || (t->m32 && op != 0xfa && op != 0xfb)) return 0;
    Slow s = {0};
    if (op == 0x120) {
        if (in->mod != 3) return 0;
        u32 off;
        switch (in->reg) {
        case 0: off = OFF(cr0); break;
        case 2: off = OFF(cr2); break;
        case 3: off = OFF(cr3); break;
        case 4: off = OFF(cr4); break;
        case 8: off = OFF(cr8); break;
        default: return 0;
        }
        e8(e, 0x83); e8(e, 0xbd); e32(e, OFF(cpl)); e8(e, 0);          /* cmp dword [cpl], 0 */
        s.jumps[s.n++] = jcc32(e, CC_NE);
        mov_r_m(e, EAX, off); mov_r_m(e, EDX, off + 4);
        store_reg(t, in->rm, 8, EAX, EDX);
        slow_continue(t, &s);
        return 1;
    }
    if (op == 0x122) {   /* MOV CR8, reg: TPR; ends the block, back to C if an interrupt is requested */
        e8(e, 0x83); e8(e, 0xbd); e32(e, OFF(cpl)); e8(e, 0);
        s.jumps[s.n++] = jcc32(e, CC_NE);
        mov_r_m(e, EAX, R_LO(in->rm)); alu_r_i(e, 4, EAX, 15);
        mov_m_r(e, OFF(cr8), EAX); mov_m_i(e, OFF(cr8) + 4, 0);
        push_cpu(e); call_abs(e, (const void *)cpu_jit_set_tpr); add_esp(e, 4);
        u32 pend = t->pending;
        t->pending = pend + 1;
        int nl = t->nolink; t->nolink = 1;
        exit_const(t, t->next);
        t->nolink = nl;
        t->pending = pend;
        slow_end(t, &s);
        return 2;
    }
    e8(e, 0x83); e8(e, 0xbd); e32(e, OFF(cpl)); e8(e, 0);
    s.jumps[s.n++] = jcc32(e, CC_NE);
    if (op == 0xfa) alu_m_i(e, 4, OFF(rflags), ~(u32)F_IF);           /* and [rflags], ~IF */
    else if (op == 0xfb) alu_m_i(e, 1, OFF(rflags), F_IF);            /* or [rflags], IF */
    else {                                                              /* SWAPGS */
        u32 gb = OFF(seg) + (u32)GS * (u32)sizeof(Seg) + (u32)__builtin_offsetof(Seg, base);
        mov_r_m(e, EAX, gb); mov_r_m(e, EDX, gb + 4);
        mov_r_m(e, ECX, OFF(gs_kernel)); mov_r_m(e, EBX, OFF(gs_kernel) + 4);
        mov_m_r(e, gb, ECX); mov_m_r(e, gb + 4, EBX);
        mov_m_r(e, OFF(gs_kernel), EAX); mov_m_r(e, OFF(gs_kernel) + 4, EDX);
    }
    slow_continue(t, &s);
    return 1;
}

/* CF/OF of a rotate whose result is in lo (low dword) / hi (top dword): ROL: CF =
 * bit 0, OF = top bit ^ CF; ROR: CF = top bit, OF = top bit ^ the bit below.
 * Other flags are kept: inline when they are already materialized in rflags. */
static void emit_rot_flags(T *t, int ror, int lo, int hi) {
    E *e = &t->e;
    /* ECX = CF, EBX = OF (0/1) */
    mov_r_r(e, EBX, hi); shift_r_i(e, 5, EBX, 31);                 /* top bit */
    if (ror) {
        mov_r_r(e, ECX, EBX);                                       /* CF = top */
        mov_r_r(e, EDX, hi); shift_r_i(e, 5, EDX, 30); alu_r_i(e, 4, EDX, 1);
        alu_r_r(e, 0x31, EBX, EDX);                                 /* OF = top ^ next */
    } else {
        mov_r_r(e, ECX, lo); alu_r_i(e, 4, ECX, 1);                 /* CF = bit 0 */
        alu_r_r(e, 0x31, EBX, ECX);                                 /* OF = top ^ CF */
    }
    shift_r_i(e, 4, EBX, 11); alu_r_r(e, 0x09, ECX, EBX);           /* ecx = CF | OF << 11 */
    if ((t->lf_prev >> 8) == 5) {
        alu_m_i(e, 4, OFF(rflags), ~(u32)(F_CF | F_OF));
        alu_r_m(e, 0x0b, ECX, OFF(rflags)); mov_m_r(e, OFF(rflags), ECX);   /* ecx |= rflags; store */
    } else {
        push_r(e, ECX); push_cpu(e);
        call_abs(e, (const void *)cpu_set_cf_of);
        add_esp(e, 8);
    }
    t->lf_known = 5 << 8;                                           /* flags now in rflags */
}

/* ROL/ROR reg, imm with dead flags (only CF/OF change, and nobody reads them) */
static int t_rot(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, ror = (in->reg & 7) == 1;
    if (sz < 4 || in->rep || in->lock || in->mod != 3) return 0;
    u32 n = in->op == 0xd1 ? 1u : (u32)in->imm & (sz == 8 ? 63u : 31u);
    if (!n) return 0;
    int r = in->rm;
    if (sz == 4) {
        mov_r_m(e, EAX, R_LO(r));
        shift_r_i(e, ror, EAX, (u8)n);
        mov_m_r(e, R_LO(r), EAX);
        mov_m_i(e, R_HI(r), 0);
        if (!t->flags_dead) emit_rot_flags(t, ror, EAX, EAX);
        return 1;
    }
    /* 64-bit: halves swap for n >= 32, then a double-shift pair */
    int lo = EAX, hi = EDX;
    mov_r_m(e, EAX, R_LO(r)); mov_r_m(e, EDX, R_HI(r));
    if (n >= 32) { lo = EDX; hi = EAX; n -= 32; }   /* rotating by 32 swaps the halves */
    if (n) {
        if (!ror) { mov_r_r(e, ECX, hi); shld_r_r_i(e, hi, lo, (u8)n); shld_r_r_i(e, lo, ECX, (u8)n); }
        else { mov_r_r(e, ECX, lo); shrd_r_r_i(e, lo, hi, (u8)n); shrd_r_r_i(e, hi, ECX, (u8)n); }
    }
    mov_m_r(e, R_LO(r), lo); mov_m_r(e, R_HI(r), hi);
    if (!t->flags_dead) emit_rot_flags(t, ror, lo, hi);
    return 1;
}

/* MOVZX / MOVSX / MOVSXD into 32/64-bit registers */
static int t_movx(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, ssz, sgn;
    if (sz < 4 || in->rep) return 0;
    switch (in->op) {
    case 0x1b6: ssz = 1; sgn = 0; break;
    case 0x1b7: ssz = 2; sgn = 0; break;
    case 0x1be: ssz = 1; sgn = 1; break;
    case 0x1bf: ssz = 2; sgn = 1; break;
    default: if (sz != 8) return 0; ssz = 4; sgn = 1; break;   /* 63 MOVSXD */
    }
    Slow s = {0};
    u8 op2 = ssz == 1 ? (sgn ? 0xbe : 0xb6) : (sgn ? 0xbf : 0xb7);
    if (in->mod == 3) {
        if (ssz == 4) mov_r_m(e, EAX, R_LO(in->rm));
        else if (ssz == 1 && !in->rex && in->rm >= 4 && in->rm < 8) movx_m(e, op2, EAX, R_LO(in->rm - 4) + 1);
        else movx_m(e, op2, EAX, R_LO(in->rm));
    } else {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 0, ssz);
        if (ssz == 4) ld_b(e, EAX, EDI, 0);
        else movx_b(e, op2, EAX, EDI);
    }
    if (sgn && sz == 8) { mov_r_r(e, EDX, EAX); shift_r_i(e, 7, EDX, 31); }
    else mov_r_i(e, EDX, 0);
    store_reg(t, in->reg, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* push guest register `reg` (or the constant cval if reg < 0); RSP is updated
 * only after the store succeeded, so a TLB miss leaves the state untouched */
/* 32-bit code: ESP -= 4 (wrapping), store the low half of reg / cval */
static void emit_push32(T *t, Slow *s, int reg, u32 cval) {
    E *e = &t->e;
    mov_r_m(e, EAX, R_LO(4)); mov_r_i(e, EDX, 0);
    alu_r_i(e, 5, EAX, 4);
    emit_tlb(t, s, 1, 4);
    if (reg >= 0) mov_r_m(e, ECX, R_LO(reg)); else mov_r_i(e, ECX, cval);
    st_b(e, EDI, 0, ECX);
    mov_m_r(e, R_LO(4), EAX); mov_m_i(e, R_HI(4), 0);
}
/* 32-bit code: ECX = [ESP], EBX = 0, EAX = ESP + 4 (not stored yet) */
static void emit_pop32(T *t, Slow *s) {
    E *e = &t->e;
    mov_r_m(e, EAX, R_LO(4)); mov_r_i(e, EDX, 0);
    emit_tlb(t, s, 0, 4);
    ld_b(e, ECX, EDI, 0); mov_r_i(e, EBX, 0);
    alu_r_i(e, 0, EAX, 4);
}

static void emit_push(T *t, Slow *s, int reg, u64 cval) {
    E *e = &t->e;
    if (t->m32) { emit_push32(t, s, reg, (u32)cval); return; }
    mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
    alu_r_i(e, 5, EAX, 8); alu_r_i(e, 3, EDX, 0);
    emit_tlb(t, s, 1, 8);
    if (reg >= 0) { mov_r_m(e, ECX, R_LO(reg)); mov_r_m(e, EBX, R_HI(reg)); }
    else { mov_r_i(e, ECX, (u32)cval); mov_r_i(e, EBX, (u32)(cval >> 32)); }
    st_b(e, EDI, 0, ECX); st_b(e, EDI, 4, EBX);
    mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
}

/* [edi + d] with a disp8 or disp32 */
static void edi_disp(E *e, int r, i32 d) {
    if (d >= -128 && d < 128) { e8(e, 0x40 | r << 3 | EDI); e8(e, (u8)d); }
    else { e8(e, 0x80 | r << 3 | EDI); e32(e, (u32)d); }
}
static void sr_st(E *e, i32 d, int r) { e8(e, 0x89); edi_disp(e, r, d); }
static void sr_ld(E *e, int r, i32 d) { e8(e, 0x8b); edi_disp(e, r, d); }
static void sr_st_i(E *e, i32 d, u32 v) { e8(e, 0xc7); edi_disp(e, 0, d); e32(e, v); }
static void sr_rsp_add(T *t, int sub) {   /* RSP -= 8 (sub) or += 8, as two memory RMWs */
    E *e = &t->e;
    e8(e, 0x83); ebp_disp(e, sub ? 5 : 0, R_LO(4)); e8(e, 8);
    e8(e, 0x83); ebp_disp(e, sub ? 3 : 2, R_HI(4)); e8(e, 0);
}
static void sr_push_const(T *t, u64 v) {
    E *e = &t->e;
    prof_inc_e(e, &ph_sr_acc);
    mov_r_m(e, EDI, OFF(jit_sr));
    sr_st_i(e, t->sr_delta - 8, (u32)v); sr_st_i(e, t->sr_delta - 4, (u32)(v >> 32));
    sr_rsp_add(t, 1);
}

static int t_stack(T *t, const Ins *in) {
    E *e = &t->e;
    if (in->p66 || in->rep || in->lock) return 0;
    Slow s = {0};
    int op = in->op;
    if (t->sr_cur && !t->m32) {
        int n = (op & 7) | (in->b << 3);
        i32 d = t->sr_delta;
        prof_inc_e(e, &ph_sr_acc);
        if (op <= 0x57) {
            mov_r_m(e, ECX, R_LO(n)); mov_r_m(e, EBX, R_HI(n));
            mov_r_m(e, EDI, OFF(jit_sr));
            sr_st(e, d - 8, ECX); sr_st(e, d - 4, EBX);
            e8(e, 0x83); ebp_disp(e, 5, R_LO(4)); e8(e, 8);       /* sub dword [rsp_lo], 8 */
            e8(e, 0x83); ebp_disp(e, 3, R_HI(4)); e8(e, 0);       /* sbb dword [rsp_hi], 0 */
        } else {
            mov_r_m(e, EDI, OFF(jit_sr));
            sr_ld(e, ECX, d); sr_ld(e, EBX, d + 4);
            e8(e, 0x83); ebp_disp(e, 0, R_LO(4)); e8(e, 8);       /* add dword [rsp_lo], 8 */
            e8(e, 0x83); ebp_disp(e, 2, R_HI(4)); e8(e, 0);       /* adc dword [rsp_hi], 0 */
            store_reg(t, n, 8, ECX, EBX);
            fw_set(t, n, ECX, EBX);
        }
        return 1;
    }
    if (op >= 0x50 && op <= 0x57) {
        emit_push(t, &s, (op & 7) | (in->b << 3), 0);
        slow_continue(t, &s);
        return 1;
    }
    if (op >= 0x58 && op <= 0x5f && t->m32) {
        emit_pop32(t, &s);
        mov_m_r(e, R_LO(4), EAX); mov_m_i(e, R_HI(4), 0);
        store_reg(t, op & 7, 4, ECX, EBX);
        slow_continue(t, &s);
        return 1;
    }
    if (op >= 0x58 && op <= 0x5f) {
        int n = (op & 7) | (in->b << 3);
        mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
        emit_tlb(t, &s, 0, 8);
        ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        alu_r_i(e, 0, EAX, 8); alu_r_i(e, 2, EDX, 0);
        mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
        store_reg(t, n, 8, ECX, EBX);
        fw_set(t, n, ECX, EBX);
        slow_continue(t, &s);
        return 1;
    }
    return 0;
}

/* Leave the block for the dynamic target in ECX:EBX (this instruction
 * completed): store RIP, account icount, then jump straight to the target's
 * translation if the lookup table knows it, else return to the dispatcher. */
static void emit_dispatch_body(E *e, Jit *j, int m32, u8 **pj);
static void emit_dispatch(T *t) {
    E *e = &t->e;
    if (jit_off & 262144) { mov_m_r(e, OFF(rip), ECX); mov_m_r(e, OFF(rip) + 4, EBX); epilogue(t); return; }
    u8 *pj[3] = { 0, 0, 0 };
    if (t->nolink) emit_irq_pending_jumps(t, &pj[0], &pj[1], &pj[2]);
    if (!t->nolink && !(jit_off2 & 32) && !(jit_off & 0x4000000)) {
        /* shared lookup, but the indirect jump stays here: one predictor entry per exit site */
        e8(e, 0xe8); u8 *r = e->p; e32(e, 0); patch(e, r, t->j->lookup_fn[t->m32]);
        test_r_r(e, EDX, EDX);
        u8 *miss = jcc32(e, CC_E); patch(e, miss, t->j->miss_fn);
        e8(e, 0xff); e8(e, 0xe2);                                  /* jmp edx */
        return;
    }
    if (!(jit_off & 0x4000000)) {
        u8 *j = jmp32(e);
        patch(e, j, t->j->disp_fn[t->m32]);
        if (t->nolink) {
            for (int k = 0; k < 3; k++) patch(e, pj[k], e->p);
            mov_m_r(e, OFF(rip), ECX); mov_m_r(e, OFF(rip) + 4, EBX);
            epilogue(t);
        }
        return;
    }
    emit_dispatch_body(e, t->j, t->m32, pj);
}
/* jc lookup of ECX:EBX; jumps to the block's entry or returns to C (pj: extra miss jumps) */
/* jump-cache lookup of ECX:EBX as a routine: returns EDX = block entry, 0 if none (ECX:EBX kept) */
static void emit_dispatch_body2(E *e, Jit *j, int m32, u8 **pj, int as_fn);
static void emit_lookup_routine(E *e, Jit *j, int m32) { emit_dispatch_body2(e, j, m32, 0, 1); }
static void emit_miss_routine(E *e) {
    mov_m_r(e, OFF(rip), ECX); mov_m_r(e, OFF(rip) + 4, EBX);
    emit_epilogue(e);
}
static void emit_dispatch_body(E *e, Jit *j, int m32, u8 **pj) { emit_dispatch_body2(e, j, m32, pj, 0); }
static void emit_dispatch_body2(E *e, Jit *j, int m32, u8 **pj, int as_fn) {
    mov_r_r(e, EDX, ECX);
    shift_r_i(e, 5, EDX, 12);
    alu_r_r(e, 0x31, EDX, ECX);
    alu_r_i(e, 4, EDX, JC_SIZE - 1);
    shift_r_i(e, 4, EDX, 4);                                   /* * 16 = sizeof(JcEntry) */
    alu_r_i(e, 0, EDX, (u32)(uintptr_t)j->jc);
    e8(e, 0x3b); e8(e, 0x0a);                                  /* cmp ecx, [edx] */
    u8 *o1 = jcc32(e, CC_NE);
    e8(e, 0x3b); e8(e, 0x5a); e8(e, 4);                        /* cmp ebx, [edx+4] */
    u8 *o2 = jcc32(e, CC_NE);
    ld_b(e, EDX, EDX, 8);                                      /* JitBlock * */
    test_r_r(e, EDX, EDX);
    u8 *o3 = jcc32(e, CC_E);
    e8(e, 0x83); e8(e, 0x7a); e8(e, (u8)__builtin_offsetof(JitBlock, m32)); e8(e, (u8)m32);   /* cmp [edx+m32], mode */
    u8 *o8 = jcc32(e, CC_NE);
    /* exec TLB must map the target page to the block's page */
    mov_r_r(e, EDI, ECX);
    shift_r_i(e, 5, EDI, 12);
    alu_r_i(e, 4, EDI, TLB_SIZE - 1);
    shift_r_i(e, 4, EDI, 4);
    e8(e, 0x8d); e8(e, 0xbc); e8(e, 0x3d); e32(e, OFF(tlb_x));  /* lea edi, [ebp+edi+tlb_x] */
    mov_r_r(e, EAX, ECX);
    alu_r_i(e, 4, EAX, 0xfffff000u);
    alu_r_m(e, 0x0b, EAX, OFF(tlb_mode));
    e8(e, 0x3b); e8(e, 0x07);                                  /* cmp eax, [edi] */
    u8 *o5 = jcc32(e, CC_NE);
    e8(e, 0x3b); e8(e, 0x5f); e8(e, 4);                        /* cmp ebx, [edi+4] */
    u8 *o6 = jcc32(e, CC_NE);
    ld_b(e, EAX, EDI, 8);
    e8(e, 0x3b); e8(e, 0x42); e8(e, (u8)__builtin_offsetof(JitBlock, host_page));   /* cmp eax, [edx+host_page] */
    u8 *o7 = jcc32(e, CC_NE);
    ld_b(e, EDX, EDX, (u8)__builtin_offsetof(JitBlock, entry));
    test_r_r(e, EDX, EDX);
    u8 *o4 = jcc32(e, CC_E);
    if (as_fn) e8(e, 0xc3);                                    /* ret: EDX = entry */
    else { e8(e, 0xff); e8(e, 0xe2); }                         /* jmp edx */
    patch(e, o5, e->p); patch(e, o6, e->p); patch(e, o7, e->p); patch(e, o8, e->p);
    patch(e, o1, e->p); patch(e, o2, e->p); patch(e, o3, e->p); patch(e, o4, e->p);
    if (as_fn) { alu_r_r(e, 0x31, EDX, EDX); e8(e, 0xc3); return; }   /* xor edx, edx; ret */
    for (int k = 0; pj && k < 3; k++) if (pj[k]) patch(e, pj[k], e->p);
    mov_m_r(e, OFF(rip), ECX); mov_m_r(e, OFF(rip) + 4, EBX);
    emit_epilogue(e);
}

/* Re-create host flags for condition cc from the lazy state a native
 * flag producer earlier in this block left in memory (t->lf_prev):
 * kind 1 ADD, 2 SUB/CMP, 3 logic, 4 result-only (shifts, INC/DEC: ZF SF PF).
 * Clobbers EAX/EDX only. Returns 1 if the host flags now decide cc. */
static int emit_lf_recompute(T *t, int cc) {
    E *e = &t->e;
    int kind = t->lf_prev >> 8, sz = t->lf_prev & 15;
    int c2 = cc >> 1;   /* 0 O, 1 B, 2 E, 3 BE, 4 S, 5 P, 6 L, 7 LE */
    if (!kind) return 0;
    if (kind == 6) kind = 4;   /* INC/DEC: result-only flags here (CF is lf_c) */
    if (kind == 4 && c2 != 2 && c2 != 4 && c2 != 5) return 0;
    if (sz <= 4) {
        if (kind >= 3) {
            if (sz == 2) e8(e, 0x66);
            e8(e, sz == 1 ? 0x8a : 0x8b); e8(e, 0x85); e32(e, OFF(lf_res));       /* mov al/ax/eax, [res] */
            if (sz == 2) e8(e, 0x66);
            e8(e, sz == 1 ? 0x84 : 0x85); e8(e, 0xc0);                            /* test */
        } else {
            if (sz == 2) e8(e, 0x66);
            e8(e, sz == 1 ? 0x8a : 0x8b); e8(e, 0x85); e32(e, OFF(lf_a));
            if (sz == 2) e8(e, 0x66);
            e8(e, (kind == 1 ? 0x02 : 0x3a) | (sz == 1 ? 0 : 1)); e8(e, 0x85); e32(e, OFF(lf_b));   /* add/cmp */
        }
        return 1;
    }
    if (c2 == 5) { mov_r_m(e, EAX, OFF(lf_res)); e8(e, 0x84); e8(e, 0xc0); return 1; }   /* PF: low byte */
    if (c2 == 2 || (kind == 3 && c2 <= 3)) {
        mov_r_m(e, EAX, OFF(lf_res)); alu_r_m(e, 0x0b, EAX, OFF(lf_res) + 4);   /* ZF of 64 bits, CF = OF = 0 */
        return 1;
    }
    if ((kind >= 3 && c2 == 4) || (kind == 3 && c2 == 6)) {
        mov_r_m(e, EAX, OFF(lf_res) + 4); test_r_r(e, EAX, EAX);                 /* SF, OF = 0 */
        return 1;
    }
    if (kind <= 2 && (c2 == 0 || c2 == 1 || c2 == 4 || c2 == 6)) {
        mov_r_m(e, EAX, OFF(lf_a)); mov_r_m(e, EDX, OFF(lf_a) + 4);
        alu_r_m(e, kind == 1 ? 0x03 : 0x3b, EAX, OFF(lf_b));                     /* add / cmp low */
        alu_r_m(e, kind == 1 ? 0x13 : 0x1b, EDX, OFF(lf_b) + 4);                 /* adc / sbb high */
        return 1;
    }
    return 0;
}

/* BE/A or LE/G of a 64-bit result in EAX:EDX whose CF (or SF/OF) are in the
 * host flags but whose ZF covers only the high half: combine into ZF = !cond.
 * Clobbers ECX. Returns the host cc that is true when cc holds. */
static int emit_cc64_combine(T *t, int cc) {
    E *e = &t->e;
    e8(e, 0x0f); e8(e, (cc >> 1) == 3 ? 0x92 : 0x9c); e8(e, 0xc1);   /* setc cl / setl cl */
    e8(e, 0x09); e8(e, 0xd0);                                          /* or eax, edx */
    e8(e, 0x0f); e8(e, 0x94); e8(e, 0xc5);                             /* setz ch */
    e8(e, 0x08); e8(e, 0xe9);                                          /* or cl, ch */
    return (cc & 1) ? CC_E : CC_NE;
}

/* Host condition code deciding cc: from the live host flags of the previous ALU
 * op (when nothing in between touched them), else recomputed from the lazy
 * state. -1: call cpu_cond. Clobbers EAX, ECX, EDX. */
static int emit_host_cc(T *t, int cc, int fuse_ok) {
    int fz = fuse_ok ? t->fuse_now : 0;
    if (fz == 1 || fz == 2 || fz == 4 || (fz == 8 && (cc <= 3 || cc == 8 || cc == 9 || cc == 12 || cc == 13))) return cc;
    if (jit_off & 16384) return -1;
    if (fz == 8 && (cc == 4 || cc == 5)) { e8(&t->e, 0x09); e8(&t->e, 0xd0); return cc; }   /* or eax, edx: ZF of 64 bits */
    if (fz == 8 && ((cc >> 1) == 3 || (cc >> 1) == 7)) return emit_cc64_combine(t, cc);
    int kind = t->lf_prev >> 8, sz = t->lf_prev & 15;
    if (kind == 5) {
        /* flags materialized in rflags: test the bits */
        static const u32 mask[8] = { F_OF, F_CF, F_ZF, F_CF | F_ZF, F_SF, F_PF, 0, 0 };
        u32 mk = mask[cc >> 1];
        if (mk) {
            if (mk < 0x100) { e8(&t->e, 0xf6); ebp_disp(&t->e, 0, OFF(rflags)); e8(&t->e, mk); }        /* test byte [rflags], mk */
            else { e8(&t->e, 0xf6); ebp_disp(&t->e, 0, OFF(rflags) + 1); e8(&t->e, mk >> 8); }       /* test byte [rflags+1], mk>>8 */
            return (cc & 1) ? CC_E : CC_NE;
        }
        return -1;
    }
    if (sz == 8 && kind >= 1 && kind <= 3 && ((cc >> 1) == 3 || (cc >> 1) == 7)) {
        E *e = &t->e;
        if (kind == 3) {
            mov_r_m(e, EAX, OFF(lf_res)); mov_r_m(e, EDX, OFF(lf_res) + 4);
            test_r_r(e, EDX, EDX);                                             /* SF; CF = OF = 0 */
        } else {
            mov_r_m(e, EAX, OFF(lf_a)); mov_r_m(e, EDX, OFF(lf_a) + 4);
            alu_r_m(e, kind == 1 ? 0x03 : 0x2b, EAX, OFF(lf_b));
            alu_r_m(e, kind == 1 ? 0x13 : 0x1b, EDX, OFF(lf_b) + 4);
        }
        return emit_cc64_combine(t, cc);
    }
    if (emit_lf_recompute(t, cc)) return cc;
    if ((cc == 4 || cc == 5) && t->j->lfzf_fn) {           /* unknown producer: shared ZF routine */
        u8 *r = (e8(&t->e, 0xe8), t->e.p); e32(&t->e, 0); patch(&t->e, r, t->j->lfzf_fn);
        return cc;
    }
    return -1;
}

/* EAX = current CF (0/1), inline when the producer is known */
static void emit_cf_eax(T *t) {
    E *e = &t->e;
    if (jit_off & 16384) { push_cpu(e); call_abs(e, (const void *)cpu_lf_cf); add_esp(e, 4); return; }
    if ((t->lf_prev >> 8) == 3) { mov_r_i(e, EAX, 0); return; }
    if ((t->lf_prev >> 8) == 6) { mov_r_m(e, EAX, OFF(lf_c)); return; }   /* INC/DEC keep CF in lf_c */
    if ((t->lf_prev >> 8) == 5) { mov_r_m(e, EAX, OFF(rflags)); alu_r_i(e, 4, EAX, 1); return; }
    if ((t->lf_prev >> 8) == 1 || (t->lf_prev >> 8) == 2) {
        emit_lf_recompute(t, 2);
        e8(e, 0x0f); e8(e, 0x92); e8(e, 0xc0);                                   /* setc al */
        e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc0);                                   /* movzx eax, al */
        return;
    }
    prof_inc_e(e, &ph_lfcf[(t->lf_prev >> 8) & 7]);
    if (!(jit_off & 0x1u)) { u8 *r = (e8(e, 0xe8), e->p); e32(e, 0); patch(e, r, t->j->lfcf_fn); return; }
    push_cpu(e); call_abs(e, (const void *)cpu_lf_cf); add_esp(e, 4);
}

/* Guest CALL/RET on host call/ret, so the host return predictor works.
 * A call pushes {return rip lo, hi, RET_MAGIC} and calls the link to the
 * callee; the code right after that host call continues at the return
 * address. A RET whose popped target equals the top frame's rip returns
 * there with a host ret; anything else uses the normal dispatch. */
#define RET_MAGIC 0x5a17c0deu
static void emit_call_frame(T *t, u64 ra, void (*to_callee)(T *, u64), u64 target) {
    E *e = &t->e;
    if (!(jit_off & 0x800000)) { to_callee(t, target); return; }   /* opt-in (W32_JITOFF bit 23): measured no gain */
    e8(e, 0x3b); e8(e, 0xa5); e32(e, OFF(jit_limit));            /* cmp esp, [jit_limit] */
    u8 *deep = jcc32(e, 2);                                       /* jb: too deep, plain exit */
    push_i(e, (u32)ra); push_i(e, (u32)(ra >> 32)); push_i(e, RET_MAGIC);
    e8(e, 0xe8); u8 *call = e->p; e32(e, 0);                      /* call to_callee */
    exit_const(t, ra);                                            /* returned: continue at ra */
    patch(e, call, e->p);
    to_callee(t, target);
    patch(e, deep, e->p);
    to_callee(t, target);
}
static void callee_const(T *t, u64 target) { exit_const(t, target); }
static void callee_dispatch(T *t, u64 target) { (void)target; emit_dispatch(t); }
/* target in ECX:EBX (popped return address): host ret if it matches the top frame */
static void emit_ret_match(T *t) {
    E *e = &t->e;
    if (!(jit_off & 0x800000)) return;
    e8(e, 0x81); e8(e, 0x7c); e8(e, 0x24); e8(e, 4); e32(e, RET_MAGIC);   /* cmp dword [esp+4], MAGIC */
    u8 *n1 = jcc32(e, CC_NE);
    e8(e, 0x3b); e8(e, 0x4c); e8(e, 0x24); e8(e, 12);                     /* cmp ecx, [esp+12] */
    u8 *n2 = jcc32(e, CC_NE);
    e8(e, 0x3b); e8(e, 0x5c); e8(e, 0x24); e8(e, 8);                      /* cmp ebx, [esp+8] */
    u8 *n3 = jcc32(e, CC_NE);
    prof_inc_e(e, &px_rethit);
    e8(e, 0xc2); e8(e, 12); e8(e, 0);                                     /* ret 12 */
    patch(e, n1, e->p); patch(e, n2, e->p); patch(e, n3, e->p);
    prof_inc_e(e, &px_retmiss);
}

/* control transfers: returns 2 (block ends) or 0 */
static int t_branch(T *t, const Ins *in) {
    E *e = &t->e;
    int op = in->op;
    if (in->p66) return 0;
    if ((op >= 0x70 && op <= 0x7f) || (op >= 0x180 && op <= 0x18f)) {
        int cc = op & 15;
        u8 *tk;
        int hc = emit_host_cc(t, cc, 1);
        if (hc >= 0) {
            tk = jcc32(e, hc);                                  /* host flags decide */
        } else {
            push_i(e, (u32)cc); push_cpu(e);
            prof_inc_e(e, &ph_cond[cc * 8 + ((t->lf_prev >> 8) & 7)]);
    call_abs(e, (const void *)cpu_cond);
            add_esp(e, 8);
            test_r_r(e, EAX, EAX);
            tk = jcc32(e, CC_NE);
        }
        u64 tgt = t->next + (u64)in->imm;
        if (t->m32) tgt = (u32)tgt;
        if (!t->nolink && !(jit_off & 0x20000000) && t->nside < (int)(sizeof t->sides / sizeof t->sides[0])) {
            /* superblock: the taken direction is a side exit at the tail; translation
             * continues with the fall-through path */
            prof_inc_e(e, &ph_jcc_side);
            t->sides[t->nside].jcc_rel = tk;
            t->sides[t->nside].target = tgt;
            t->sides[t->nside].dk = t->defer_k; t->sides[t->nside].dsz = t->defer_sz;
            t->defer_k = 0;
            t->sides[t->nside++].executed = t->pending + 1;
            return 1;
        }
        t->pending++;
        u32 pend = t->pending;
        exit_const(t, t->next);
        patch(e, tk, e->p);
        t->pending = pend;
        exit_const(t, tgt);
        return 2;
    }
    if (op == 0xeb || op == 0xe9) {
        t->pending++;
        exit_const(t, t->m32 ? (u32)(t->next + (u64)in->imm) : t->next + (u64)in->imm);
        return 2;
    }
    if (op == 0xe8) {
        Slow s = {0};
        if (t->sr_cur && !t->m32) sr_push_const(t, t->next);
        else emit_push(t, &s, -1, t->next);
        u32 pend = t->pending;
        t->pending = pend + 1;
        emit_call_frame(t, t->next, callee_const, t->m32 ? (u32)(t->next + (u64)in->imm) : t->next + (u64)in->imm);
        t->pending = pend;
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xc3 && !in->rep && t->m32) {
        Slow s = {0};
        emit_pop32(t, &s);                                      /* ECX:EBX = target */
        mov_m_r(e, R_LO(4), EAX); mov_m_i(e, R_HI(4), 0);
        emit_ret_match(t);
        emit_dispatch(t);
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xc2 && !in->rep && t->m32) {
        Slow s = {0};
        emit_pop32(t, &s);                                      /* ECX:EBX = target, EAX = ESP + 4 */
        alu_r_i(e, 0, EAX, (u32)(in->imm & 0xffff));
        mov_m_r(e, R_LO(4), EAX); mov_m_i(e, R_HI(4), 0);
        emit_ret_match(t);
        emit_dispatch(t);
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xc2 && !in->rep && !t->m32) {
        Slow s = {0};
        mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
        emit_tlb(t, &s, 0, 8);
        ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        alu_r_i(e, 0, EAX, 8 + (u32)(in->imm & 0xffff)); alu_r_i(e, 2, EDX, 0);
        mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
        emit_ret_match(t);
        emit_dispatch(t);
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xc3 && !in->rep && t->sr_cur) {
        prof_inc_e(e, &ph_sr_acc);
        mov_r_m(e, EDI, OFF(jit_sr));
        sr_ld(e, ECX, t->sr_delta); sr_ld(e, EBX, t->sr_delta + 4);
        sr_rsp_add(t, 0);
        emit_ret_match(t);
        emit_dispatch(t);
        return 2;
    }
    if (op == 0xc3 && !in->rep) {
        Slow s = {0};
        mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
        emit_tlb(t, &s, 0, 8);
        ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        alu_r_i(e, 0, EAX, 8); alu_r_i(e, 2, EDX, 0);
        mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
        emit_ret_match(t);
        emit_dispatch(t);
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xff && ((in->reg & 7) == 2 || (in->reg & 7) == 4) && !in->rep && !in->lock && t->m32) {
        Slow s = {0};
        if (in->mod == 3) { mov_r_m(e, ECX, R_LO(in->rm)); mov_r_i(e, EBX, 0); }
        else {
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 0, 4);
            ld_b(e, ECX, EDI, 0); mov_r_i(e, EBX, 0);
        }
        if ((in->reg & 7) == 2) {
            Slow s2 = {0};
            push_r(e, ECX);
            emit_push32(t, &s2, -1, (u32)t->next);
            pop_r(e, ECX); mov_r_i(e, EBX, 0);
            emit_call_frame(t, (u32)t->next, callee_dispatch, 0);
            if (s2.n || s2.tn) {   /* stack miss: drop the saved target and interpret the call */
                for (int i = 0; i < s2.n; i++) patch(e, s2.jumps[i], e->p);
                for (int i = 0; i < s2.tn; i++) patch(e, s2.tjumps[i], e->p);
                add_esp(e, 4);
                set_rip(t, t->rip);
                emit_icount_exact(t, t->pending);
                call_step(t);
                epilogue(t);
            }
        } else emit_dispatch(t);
        slow_end(t, &s);
        return 2;
    }
    if (op == 0xff && ((in->reg & 7) == 2 || (in->reg & 7) == 4) && !in->rep && !in->lock) {
        Slow s = {0};
        if (in->mod == 3) {
            mov_r_m(e, ECX, R_LO(in->rm)); mov_r_m(e, EBX, R_HI(in->rm));
        } else {
            if (!mem_ok(in)) return 0;
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 0, 8);
            ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        }
        if ((in->reg & 7) == 2) {
            /* push the return address; target is in ECX:EBX */
            if (t->sr_cur) {
                sr_push_const(t, t->next);
                emit_call_frame(t, t->next, callee_dispatch, 0);
                slow_end(t, &s);
                return 2;
            }
            Slow s2 = {0};
            push_r(e, ECX); push_r(e, EBX);
            mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
            alu_r_i(e, 5, EAX, 8); alu_r_i(e, 3, EDX, 0);
            emit_tlb(t, &s2, 1, 8);
            st_b_i(e, EDI, 0, (u32)t->next); st_b_i(e, EDI, 4, (u32)(t->next >> 32));
            mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
            pop_r(e, EBX); pop_r(e, ECX);
            emit_call_frame(t, t->next, callee_dispatch, 0);
            if (s2.n || s2.tn) {   /* stack miss: drop the saved target and interpret the call */
                for (int i = 0; i < s2.n; i++) patch(e, s2.jumps[i], e->p);
                for (int i = 0; i < s2.tn; i++) patch(e, s2.tjumps[i], e->p);
                add_esp(e, 8);
                for (int i = 0; i < s.n; i++) { (void)0; }
                set_rip(t, t->rip);
                emit_icount_exact(t, t->pending);
                call_step(t);
                epilogue(t);
            }
        } else {
            emit_dispatch(t);
        }
        slow_end(t, &s);
        return 2;
    }
    return 0;
}


int cpu_cond(Cpu *c, int cc);
u64 cpu_imul(Cpu *c, u64 a, u64 b, u32 sz);
u64 cpu_shift(Cpu *c, u32 op, u64 a, u32 cnt, u32 sz);
void cpu_muldiv(Cpu *c, u32 r, u64 src, u32 sz);
void sse_scalar_op(Cpu *c, u32 code, Xmm *x, const u8 *src);
void sse_packed_op(Cpu *c, u32 code, Xmm *x, const u8 *src);
u64 sse_cvt_int(Cpu *c, u32 code, const u8 *src);

#define X_OFF(n) (OFF(xmm) + 16u * (u32)(n))


static void call_cond(T *t, int cc) {
    E *e = &t->e;
    prof_inc_e(e, &ph_cond[cc * 8 + ((t->lf_prev >> 8) & 7)]);
    push_i(e, (u32)cc); push_cpu(e);
    call_abs(e, (const void *)cpu_cond);
    add_esp(e, 8);
    test_r_r(e, EAX, EAX);
}

/* SHL/SHR/SAR by a constant (C1 ib, D1) on 32/64-bit operands */
/* SHL/SHR/SAR on 8/16-bit operands by an immediate (count masked to 5 bits, nonzero) */
static int t_shift_small(T *t, const Ins *in, int sz, int op, u32 n) {
    E *e = &t->e;
    Slow s = {0};
    int mem = in->mod != 3;
    if (mem) {
        if (in->a67) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_hmem(t, EAX, sz);
    } else ld_gpr(t, in, EAX, in->rm, sz);
    if (!t->flags_dead) { mov_m_r(e, OFF(lf_a), EAX); if (sz == 8) mov_m_i(e, OFF(lf_a) + 4, 0); }
    int lf;
    if (op == 4 || op == 6) { lf = LF_SHL; shift_r_i(e, 4, EAX, (u8)n); }
    else if (op == 5) { lf = LF_SHR; shift_r_i(e, 5, EAX, (u8)n); }
    else {
        lf = LF_SAR;
        e8(e, 0x0f); e8(e, sz == 1 ? 0xbe : 0xbf); e8(e, 0xc0);    /* movsx eax, al/ax */
        shift_r_i(e, 7, EAX, (u8)n);
    }
    e8(e, 0x0f); e8(e, sz == 1 ? 0xb6 : 0xb7); e8(e, 0xc0);        /* truncate */
    if (!t->flags_dead) {
        mov_m_i(e, OFF(lf_b), n); if (sz == 8) mov_m_i(e, OFF(lf_b) + 4, 0);
        mov_m_r(e, OFF(lf_res), EAX); if (sz == 8) mov_m_i(e, OFF(lf_res) + 4, 0);
        mov_m_i(e, OFF(lf_op), (u32)((u32)lf) | (u32)((u32)sz) << 16);
        t->lf_known = 4 << 8 | sz;
    }
    if (mem) st_hmem(t, sz, EAX, EDX);
    else st_gpr(t, in, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

static int t_shift(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = (in->op == 0xc0 || in->op == 0xd0) ? 1 : in->osz, op = in->reg & 7;
    if (in->rep || in->lock || op < 4) return 0;
    u32 n = (in->op == 0xd1 || in->op == 0xd0) ? 1u : (u32)in->imm & (sz == 8 ? 63u : 31u);
    if (!n) return 0;
    if (sz < 4) return t_shift_small(t, in, sz, op, n);
    Slow s = {0};
    int mem = in->mod != 3;
    if (mem) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_b(e, EAX, EDI, 0);
        if (sz == 8) ld_b(e, EDX, EDI, 4); else mov_r_i(e, EDX, 0);
    } else {
        mov_r_m(e, EAX, R_LO(in->rm));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->rm)); else mov_r_i(e, EDX, 0);
    }
    if (!t->flags_dead) { mov_m_r(e, OFF(lf_a), EAX); if (sz == 8) mov_m_r(e, OFF(lf_a) + 4, EDX); }
    int lf;
    if (op == 4 || op == 6) {
        lf = LF_SHL;
        if (sz == 4) shift_r_i(e, 4, EAX, (u8)n);
        else if (n < 32) { shld_r_r_i(e, EDX, EAX, (u8)n); shift_r_i(e, 4, EAX, (u8)n); }
        else { mov_r_r(e, EDX, EAX); if (n > 32) shift_r_i(e, 4, EDX, (u8)(n - 32)); mov_r_i(e, EAX, 0); }
    } else if (op == 5) {
        lf = LF_SHR;
        if (sz == 4) shift_r_i(e, 5, EAX, (u8)n);
        else if (n < 32) { shrd_r_r_i(e, EAX, EDX, (u8)n); shift_r_i(e, 5, EDX, (u8)n); }
        else { mov_r_r(e, EAX, EDX); if (n > 32) shift_r_i(e, 5, EAX, (u8)(n - 32)); mov_r_i(e, EDX, 0); }
    } else {
        lf = LF_SAR;
        if (sz == 4) shift_r_i(e, 7, EAX, (u8)n);
        else if (n < 32) { shrd_r_r_i(e, EAX, EDX, (u8)n); shift_r_i(e, 7, EDX, (u8)n); }
        else { mov_r_r(e, EAX, EDX); if (n > 32) shift_r_i(e, 7, EAX, (u8)(n - 32)); shift_r_i(e, 7, EDX, 31); }
    }
    if (!t->flags_dead) {
        mov_m_i(e, OFF(lf_b), n); if (sz == 8) mov_m_i(e, OFF(lf_b) + 4, 0);
        mov_m_r(e, OFF(lf_res), EAX); if (sz == 8) mov_m_r(e, OFF(lf_res) + 4, EDX);
        mov_m_i(e, OFF(lf_op), (u32)((u32)lf) | (u32)((u32)sz) << 16);
        t->lf_known = 4 << 8 | sz;
    }
    if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
    else store_reg(t, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* NOT r/m8 (F6 /2): no flags */
static int t_not8(T *t, const Ins *in) {
    E *e = &t->e;
    if (in->rep || (in->lock && in->mod == 3) || (in->mod != 3 && in->a67)) return 0;
    Slow s = {0};
    if (in->mod != 3) {
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, 1);
        e8(e, 0xf6); e8(e, 0x57); e8(e, 0);                  /* not byte [edi] */
    } else {
        e8(e, 0xf6); e8(e, 0x95); e32(e, r8_off(in, in->rm));   /* not byte [ebp+reg] */
    }
    slow_continue(t, &s);
    return 1;
}


/* NEG r/m8, r/m16 */
static int t_neg_small(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->op == 0xf6 ? 1 : 2, mem = in->mod != 3;
    if (in->rep || in->lock || (mem && !mem_ok(in))) return 0;
    Slow s = {0};
    if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 1, sz); ld_hmem(t, ECX, sz); }
    else ld_gpr(t, in, ECX, in->rm, sz);
    mov_r_i(e, EAX, 0);
    mov_m_r(e, OFF(lf_a), EAX); mov_m_r(e, OFF(lf_a) + 4, EAX);
    mov_m_r(e, OFF(lf_b), ECX); mov_m_r(e, OFF(lf_b) + 4, EAX);
    alu_r_r(e, 0x29, EAX, ECX);                                    /* 0 - x */
    e8(e, 0x0f); e8(e, sz == 1 ? 0xb6 : 0xb7); e8(e, 0xc0);        /* movzx eax, al/ax */
    mov_m_r(e, OFF(lf_res), EAX); mov_m_i(e, OFF(lf_res) + 4, 0);
    mov_m_i(e, OFF(lf_op), (u32)LF_SUB | (u32)sz << 16);
    if (mem) st_hmem(t, sz, EAX, EDX); else st_gpr(t, in, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* XCHG r/m, reg (32/64-bit); a single CPU makes the memory form atomic */
static int t_xchg(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (in->op != 0x87 || sz < 4 || in->rep || in->lock) return 0;
    if (in->mod == 3) {
        load_reg_cb(t, in->rm, sz);
        mov_r_m(e, EAX, R_LO(in->reg));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->reg)); else mov_r_i(e, EDX, 0);
        store_reg_hz(t, in->reg, sz, ECX, EBX);
        store_reg_hz(t, in->rm, sz, EAX, EDX);
        return 1;
    }
    if (!mem_ok(in)) return 0;
    Slow s = {0};
    emit_ea(t, in, 1);
    emit_tlb(t, &s, 1, sz);
    ld_b(e, ECX, EDI, 0);
    if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    mov_r_m(e, EAX, R_LO(in->reg)); st_b(e, EDI, 0, EAX);
    if (sz == 8) { mov_r_m(e, EAX, R_HI(in->reg)); st_b(e, EDI, 4, EAX); }
    store_reg_hz(t, in->reg, sz, ECX, EBX);
    slow_continue(t, &s);
    return 1;
}

/* push of the value saved in jit_x (64-bit code); a stack miss interprets the instruction */
static void push_saved(T *t, Slow *s) {
    E *e = &t->e;
    mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
    alu_r_i(e, 5, EAX, 8); alu_r_i(e, 3, EDX, 0);
    emit_tlb(t, s, 1, 8);
    mov_r_m(e, ECX, OFF(jit_x)); st_b(e, EDI, 0, ECX);
    mov_r_m(e, ECX, OFF(jit_x) + 4); st_b(e, EDI, 4, ECX);
    mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
}

/* PUSH r/m64 (FF /6) and PUSHFQ */
static int t_push_more(T *t, const Ins *in) {
    E *e = &t->e;
    if (t->m32 || in->p66 || in->rep || in->lock) return 0;
    Slow s = {0};
    if (in->op == 0x9c) {
        push_cpu(e); call_abs(e, (const void *)get_flags); add_esp(e, 4);
        alu_r_i(e, 4, EAX, 0x3c7fd7);
        mov_m_r(e, OFF(jit_x), EAX); mov_m_i(e, OFF(jit_x) + 4, 0);
    } else {
        if (in->mod == 3) {
            mov_r_m(e, ECX, R_LO(in->rm)); mov_r_m(e, EBX, R_HI(in->rm));
        } else {
            if (!mem_ok(in)) return 0;
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 0, 8);
            ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        }
        mov_m_r(e, OFF(jit_x), ECX); mov_m_r(e, OFF(jit_x) + 4, EBX);
    }
    push_saved(t, &s);
    slow_continue(t, &s);
    return 1;
}

/* 8-bit MOV: 88, 8A, C6 /0, B0+r */
static int t_mov8(T *t, const Ins *in) {
    E *e = &t->e;
    if (in->rep || in->lock) return 0;
    Slow s = {0};
    if (in->op >= 0xb0 && in->op <= 0xb7) {
        st8_m_i(e, r8_off(in, (in->op & 7) | (in->b << 3)), (u8)in->imm);
        return 1;
    }
    if (in->op == 0xc6 && (in->reg & 7)) return 0;
    if (in->mod == 3) {
        if (in->op == 0xc6) { st8_m_i(e, r8_off(in, in->rm), (u8)in->imm); return 1; }
        int src = in->op == 0x88 ? in->reg : in->rm, dst = in->op == 0x88 ? in->rm : in->reg;
        movx_m(e, 0xb6, EAX, r8_off(in, src));
        st8_m_r(e, r8_off(in, dst), EAX);
        return 1;
    }
    if (!mem_ok(in)) return 0;
    emit_ea(t, in, 1);
    emit_tlb(t, &s, in->op != 0x8a, 1);
    if (in->op == 0xc6) { e8(e, 0xc6); e8(e, 0x47); e8(e, 0); e8(e, (u32)(u8)in->imm); }
    else if (in->op == 0x88) { movx_m(e, 0xb6, ECX, r8_off(in, in->reg)); e8(e, 0x88); e8(e, 0x4f); e8(e, 0); }
    else { movx_b(e, 0xb6, ECX, EDI); st8_m_r(e, r8_off(in, in->reg), ECX); }
    slow_continue(t, &s);
    return 1;
}

/* IMUL r, r/m (0F AF) and IMUL r, r/m, imm (69, 6B) through the interpreter's
 * arithmetic helper (flags exact) */
static int t_imul(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (sz < 4 || in->rep || in->lock) return 0;
    Slow s = {0};
    if (in->mod != 3) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 0, sz);
        ld_b(e, ECX, EDI, 0);
        if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    } else {
        load_reg_cb(t, in->rm, sz);
    }
    if (sz == 4) {
        /* EAX = a, ECX = b; host IMUL gives the same low half and CF=OF */
        if (in->op == 0x1af) mov_r_m(e, EAX, R_LO(in->reg));
        else { mov_r_r(e, EAX, ECX); mov_r_i(e, ECX, (u32)in->imm); }
        e8(e, 0x0f); e8(e, 0xaf); e8(e, 0xc1);                           /* imul eax, ecx */
        e8(e, 0x0f); e8(e, 0x90); e8(e, 0xc2);                           /* seto dl */
        e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xd2);                           /* movzx edx, dl */
        if (!t->flags_dead) {
            mov_m_r(e, OFF(lf_res), EAX); mov_m_i(e, OFF(lf_res) + 4, 0);
            mov_m_r(e, OFF(lf_c), EDX);
            mov_m_i(e, OFF(lf_op), (u32)(LF_MUL) | (u32)(4) << 16);
        }
        store_reg(t, in->reg, 4, EAX, EDX);
        slow_continue(t, &s);
        return 1;
    }
    if (t->flags_dead && !(jit_off & 0x4000)) {
        /* 64-bit, flags dead: low 64 bits of the product, A0*B0 + (A0*B1 + A1*B0) << 32 */
        u32 alo, ahi;
        if (in->op == 0x1af) { alo = R_LO(in->reg); ahi = R_HI(in->reg); }
        else {
            mov_m_r(e, OFF(jit_x), ECX); mov_m_r(e, OFF(jit_x) + 4, EBX);   /* A = r/m */
            alo = OFF(jit_x); ahi = OFF(jit_x) + 4;
            mov_r_i(e, ECX, (u32)in->imm); mov_r_i(e, EBX, (u32)((u64)in->imm >> 32));   /* B = imm */
        }
        mov_r_m(e, EDI, alo); e8(e, 0x0f); e8(e, 0xaf); e8(e, 0xfb);   /* imul edi, ebx: A0*B1 */
        mov_r_m(e, EAX, ahi); e8(e, 0x0f); e8(e, 0xaf); e8(e, 0xc1);   /* imul eax, ecx: A1*B0 */
        alu_r_r(e, 0x01, EDI, EAX);
        mov_r_m(e, EAX, alo); e8(e, 0xf7); e8(e, 0xe1);                 /* mul ecx: A0*B0 */
        alu_r_r(e, 0x01, EDX, EDI);
        store_reg(t, in->reg, 8, EAX, EDX);
        slow_continue(t, &s);
        return 1;
    }
    push_i(e, (u32)sz);
    if (in->op == 0x1af) {
        push_r(e, EBX); push_r(e, ECX);
        if (sz == 8) { e8(e, 0xff); e8(e, 0xb5); e32(e, R_HI(in->reg)); }    /* push [reg hi] */
        else push_i(e, 0);
        e8(e, 0xff); e8(e, 0xb5); e32(e, R_LO(in->reg));                 /* push [reg lo] */
    } else {
        u64 imm = sz == 8 ? (u64)in->imm : (u64)(u32)in->imm;
        push_i(e, (u32)(imm >> 32)); push_i(e, (u32)imm);
        push_r(e, EBX); push_r(e, ECX);
    }
    push_cpu(e);
    call_abs(e, (const void *)cpu_imul);
    e8(e, 0x83); e8(e, 0xc4); e8(e, 24);                                  /* add esp, 24 */
    store_reg(t, in->reg, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* CMOVcc (0F 40-4F), 32/64-bit */
static int t_cmov(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (sz < 2 || in->rep || in->lock) return 0;
    Slow s = {0};
    if (in->mod != 3) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 0, sz);           /* the source is read even when the move is not taken */
    }
    if (sz == 2) {   /* 16-bit: the upper bits of the destination never change */
        u8 *no;
        int hc = emit_host_cc(t, in->op & 15, in->mod == 3);
        if (hc >= 0) no = jcc32(e, hc ^ 1);
        else { call_cond(t, in->op & 15); no = jcc32(e, CC_E); }
        if (in->mod != 3) movx_b(e, 0xb7, ECX, EDI); else mov_r_m(e, ECX, R_LO(in->rm));
        e8(e, 0x66); mov_m_r(e, R_LO(in->reg), ECX);
        patch(e, no, e->p);
        slow_continue(t, &s);
        return 1;
    }
    u8 *no;
    int hc = emit_host_cc(t, in->op & 15, in->mod == 3);
    if (hc >= 0) no = jcc32(e, hc ^ 1);   /* EDI kept */
    else { call_cond(t, in->op & 15); no = jcc32(e, CC_E); }   /* EDI survives the call (callee-saved) */
    if (in->mod != 3) { ld_b(e, ECX, EDI, 0); if (sz == 8) ld_b(e, EBX, EDI, 4); }
    else load_reg_cb(t, in->rm, sz);
    store_reg(t, in->reg, sz, ECX, EBX);
    u8 *done = jmp32(e);
    patch(e, no, e->p);
    if (sz == 4) mov_m_i(e, R_HI(in->reg), 0);
    patch(e, done, e->p);
    slow_continue(t, &s);
    return 1;
}

/* ---- SSE ---- */

static int sse_ok(Cpu *c) { return !(c->cr0 & 0xc) && (c->cr4 & 0x200); }

void cpu_jit_set_zf(Cpu *c, u32 zf);
void cpu_jit_rdtscp(Cpu *c);
/* CMPXCHG8B / CMPXCHG16B m (as the interpreter: memory is written only when equal) */
static int t_cx(T *t, const Ins *in) {
    E *e = &t->e;
    int w16 = in->w, n = w16 ? 16 : 8;
    if (in->mod == 3 || in->rep || in->a67 || in->p66) return 0;
    Slow s = {0};
    emit_ea(t, in, 1);
    if (w16) { e8(e, 0xa9); e32(e, 15); s.jumps[s.n++] = jcc32(e, CC_NE); }   /* misaligned: #GP via the interpreter */
    emit_tlb(t, &s, 0, n);
    /* compare with RDX:RAX (16) or EDX:EAX (8) */
    static const u32 w16c[4] = { 0, 4, 16, 20 };      /* rax lo/hi, rdx lo/hi as offsets from R_LO(0) */
    u8 *ne[4];
    for (int k = 0; k < n / 4; k++) {
        u32 roff = w16 ? R_LO(0) + w16c[k] : (k == 0 ? R_LO(0) : R_LO(2));
        ld_b(e, ECX, EDI, (u8)(4 * k));
        alu_r_m(e, 0x3b, ECX, roff);                  /* cmp ecx, [reg] */
        ne[k] = jcc32(e, CC_NE);
    }
    /* equal: store RCX:RBX (ECX:EBX); needs write access */
    emit_tlb(t, &s, 1, n);
    if (w16) {
        mov_r_m(e, ECX, R_LO(3)); st_b(e, EDI, 0, ECX);
        mov_r_m(e, ECX, R_HI(3)); st_b(e, EDI, 4, ECX);
        mov_r_m(e, ECX, R_LO(1)); st_b(e, EDI, 8, ECX);
        mov_r_m(e, ECX, R_HI(1)); st_b(e, EDI, 12, ECX);
    } else {
        mov_r_m(e, ECX, R_LO(3)); st_b(e, EDI, 0, ECX);
        mov_r_m(e, ECX, R_LO(1)); st_b(e, EDI, 4, ECX);
    }
    push_i(e, 1);
    u8 *done = jmp32(e);
    for (int k = 0; k < n / 4; k++) patch(e, ne[k], e->p);
    /* not equal: load into RDX:RAX */
    if (w16) {
        ld_b(e, ECX, EDI, 0); mov_m_r(e, R_LO(0), ECX);
        ld_b(e, ECX, EDI, 4); mov_m_r(e, R_HI(0), ECX);
        ld_b(e, ECX, EDI, 8); mov_m_r(e, R_LO(2), ECX);
        ld_b(e, ECX, EDI, 12); mov_m_r(e, R_HI(2), ECX);
    } else {
        ld_b(e, ECX, EDI, 0); mov_m_r(e, R_LO(0), ECX); mov_m_i(e, R_HI(0), 0);
        ld_b(e, ECX, EDI, 4); mov_m_r(e, R_LO(2), ECX); mov_m_i(e, R_HI(2), 0);
    }
    push_i(e, 0);
    patch(e, done, e->p);
    push_cpu(e); call_abs(e, (const void *)cpu_jit_set_zf); add_esp(e, 8);
    slow_continue(t, &s);
    return 1;
}

/* LDMXCSR / STMXCSR m32 */
static int t_mxcsr(T *t, Cpu *c, const Ins *in) {
    E *e = &t->e;
    if (!sse_ok(c) || in->mod == 3 || !mem_ok(in) || in->rep || in->p66) return 0;
    Slow s = {0};
    int st = (in->reg & 7) == 3;
    emit_ea(t, in, 1);
    emit_tlb(t, &s, st, 4);
    if (st) { mov_r_m(e, ECX, OFF(mxcsr)); st_b(e, EDI, 0, ECX); }
    else {
        ld_b(e, ECX, EDI, 0);
        e8(e, 0xf7); e8(e, 0xc1); e32(e, 0xffff0000u);              /* test ecx, reserved bits: #GP via the interpreter */
        s.jumps[s.n++] = jcc32(e, CC_NE);
        mov_m_r(e, OFF(mxcsr), ECX);
    }
    slow_continue(t, &s);
    return 1;
}

/* copy n dwords between [src_base+so] and [dst_base+do] using EAX */
static void copy_dw(E *e, int db, u32 dof, int sb, u32 sof, int n) {
    for (int i = 0; i < n; i++) {
        if (sb == EBP) mov_r_m(e, EAX, sof + 4u * (u32)i); else ld_b(e, EAX, sb, (u8)(sof + 4u * (u32)i));
        if (db == EBP) mov_m_r(e, dof + 4u * (u32)i, EAX); else st_b(e, db, (u8)(dof + 4u * (u32)i), EAX);
    }
}

static void align16_check(T *t, Slow *s) {
    E *e = &t->e;
    e8(e, 0xa9); e32(e, 15);                     /* test eax, 15 */
    s->jumps[s->n++] = jcc32(e, CC_NE);
}

static int t_sse(T *t, Cpu *c, const Ins *in) {
    E *e = &t->e;
    if (!sse_ok(c) || in->lock || in->a67) return 0;
    int b = in->op & 0xff, pfx = in->rep == 0xf3 ? 3 : in->rep == 0xf2 ? 2 : in->p66 ? 1 : 0;
    if (in->rep && in->p66) return 0;
    u32 dx = X_OFF(in->reg);
    Slow s = {0};
    int mem = in->mod != 3;
    u32 sx = mem ? 0 : X_OFF(in->rm);

    /* 128-bit loads/stores: MOVAPS/PD, MOVUPS/PD, MOVDQA/DQU */
    int load128 = (b == 0x28 && pfx <= 1) || (b == 0x10 && pfx <= 1) || (b == 0x6f && (pfx == 1 || pfx == 3));
    int store128 = (b == 0x29 && pfx <= 1) || (b == 0x11 && pfx <= 1) || (b == 0x7f && (pfx == 1 || pfx == 3));
    int movnt = ((b == 0xe7 && pfx == 1) || (b == 0x2b && pfx <= 1)) && mem && !(jit_off & 32);   /* MOVNTDQ / MOVNTPS/PD */
    store128 |= movnt;
    if (load128 || store128) {
        int aligned = b == 0x28 || b == 0x29 || ((b == 0x6f || b == 0x7f) && pfx == 1) || movnt;
        if (!mem) {
            if (load128) copy_dw(e, EBP, dx, EBP, sx, 4); else copy_dw(e, EBP, sx, EBP, dx, 4);
            return 1;
        }
        emit_ea(t, in, 1);
        if (aligned) align16_check(t, &s);
        emit_tlb(t, &s, store128, 16);
        if (load128) copy_dw(e, EBP, dx, EDI, 0, 4); else copy_dw(e, EDI, 0, EBP, dx, 4);
        slow_continue(t, &s);
        return 1;
    }
    /* MOVSS / MOVSD */
    if ((b == 0x10 || b == 0x11) && pfx >= 2) {
        int n = pfx == 3 ? 1 : 2;
        if (!mem) {
            if (b == 0x10) copy_dw(e, EBP, dx, EBP, sx, n); else copy_dw(e, EBP, sx, EBP, dx, n);
            return 1;
        }
        emit_ea(t, in, 1);
        emit_tlb(t, &s, b == 0x11, n * 4);
        if (b == 0x10) {
            copy_dw(e, EBP, dx, EDI, 0, n);
            for (int i = n; i < 4; i++) mov_m_i(e, dx + 4u * (u32)i, 0);
        } else copy_dw(e, EDI, 0, EBP, dx, n);
        slow_continue(t, &s);
        return 1;
    }
    /* bitwise: ANDPS/ANDNPS/ORPS/XORPS (+PD), PAND/PANDN/POR/PXOR */
    int logic = -1;
    if (b >= 0x54 && b <= 0x57 && pfx <= 1) logic = b - 0x54;          /* 0 and, 1 andn, 2 or, 3 xor */
    if (pfx == 1 && b == 0xdb) logic = 0;
    if (pfx == 1 && b == 0xdf) logic = 1;
    if (pfx == 1 && b == 0xeb) logic = 2;
    if (pfx == 1 && b == 0xef) logic = 3;
    if (logic >= 0) {
        int sb = EBP;
        u32 so = sx;
        if (mem) {
            emit_ea(t, in, 1);
            align16_check(t, &s);
            emit_tlb(t, &s, 0, 16);
            sb = EDI; so = 0;
        } else if (in->rm == in->reg && logic == 3) {
            for (int i = 0; i < 4; i++) mov_m_i(e, dx + 4u * (u32)i, 0);   /* xor x, x */
            return 1;
        }
        static const u8 opc[4] = { 0x23, 0x23, 0x0b, 0x33 };            /* and, (not+and), or, xor: r, r/m */
        for (int i = 0; i < 4; i++) {
            mov_r_m(e, EAX, dx + 4u * (u32)i);
            if (logic == 1) { e8(e, 0xf7); e8(e, 0xd0); }                /* not eax */
            if (sb == EBP) alu_r_m(e, opc[logic], EAX, so + 4u * (u32)i);
            else { e8(e, opc[logic]); e8(e, 0x40 | EAX << 3 | EDI); e8(e, 4u * (u32)i); }
            mov_m_r(e, dx + 4u * (u32)i, EAX);
        }
        slow_continue(t, &s);
        return 1;
    }
    /* MOVD/MOVQ xmm <- r/m (66 0F 6E), r/m <- xmm (66 0F 7E), MOVQ xmm <- xmm/m64 (F3 0F 7E),
     * MOVQ xmm/m64 <- xmm (66 0F D6) */
    if (pfx == 1 && b == 0x6e) {
        int n = in->w ? 2 : 1;
        if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 0, n * 4); copy_dw(e, EBP, dx, EDI, 0, n); }
        else copy_dw(e, EBP, dx, EBP, R_LO(in->rm), n);
        for (int i = n; i < 4; i++) mov_m_i(e, dx + 4u * (u32)i, 0);
        slow_continue(t, &s);
        return 1;
    }
    if (pfx == 1 && b == 0x7e) {
        int n = in->w ? 2 : 1;
        if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 1, n * 4); copy_dw(e, EDI, 0, EBP, dx, n); }
        else { copy_dw(e, EBP, R_LO(in->rm), EBP, dx, n); if (n == 1) mov_m_i(e, R_HI(in->rm), 0); }
        slow_continue(t, &s);
        return 1;
    }
    /* scalar arithmetic / compare / min / max / sqrt and COMIS/UCOMIS through the C helpers
     * (MXCSR rounding, DAZ/FTZ and SSE NaN rules); CVTSI2Sx and CVTSx2Sx are interpreted */
    if (((b == 0x2e || b == 0x2f) && pfx <= 1) ||
        ((b == 0xc2 || b == 0x5d || b == 0x5f || b == 0x51 || b == 0x58 || b == 0x59 || b == 0x5c || b == 0x5e) && pfx >= 2)) {
        int dbl = pfx == 1 || pfx == 2;
        if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 0, dbl ? 8 : 4); push_r(e, EDI); }
        else { e8(e, 0x8d); e8(e, 0x85); e32(e, sx); push_r(e, EAX); }
        e8(e, 0x8d); e8(e, 0x85); e32(e, dx); push_r(e, EAX);
        push_i(e, (u32)b | (u32)pfx << 8 | ((u32)(u8)in->imm << 16));
        push_cpu(e);
        call_abs(e, (const void *)sse_scalar_op);
        add_esp(e, 16);
        slow_continue(t, &s);
        return 1;
    }
    /* packed arithmetic / conversions / shuffles through the interpreter's own lane code */
    if (!(jit_off2 & 8) && ((((b >= 0x58 && b <= 0x5f && b != 0x5b && b != 0x5a) || b == 0x5a || b == 0xc6) && pfx <= 1) || (b == 0x5b && pfx != 2))) {
        int n8 = b == 0x5a && pfx == 0;   /* CVTPS2PD reads 8 bytes (no alignment rule) */
        if (mem) {
            emit_ea(t, in, 1);
            if (!n8) align16_check(t, &s);
            emit_tlb(t, &s, 0, n8 ? 8 : 16);
            push_r(e, EDI);
        } else { e8(e, 0x8d); e8(e, 0x85); e32(e, sx); push_r(e, EAX); }
        e8(e, 0x8d); e8(e, 0x85); e32(e, dx); push_r(e, EAX);
        push_i(e, (u32)b | (u32)pfx << 8 | ((u32)(u8)in->imm << 16));
        push_cpu(e);
        call_abs(e, (const void *)sse_packed_op);
        add_esp(e, 16);
        slow_continue(t, &s);
        return 1;
    }
    if ((b == 0x2c || b == 0x2d) && pfx >= 2) {
        int dsz = in->w ? 8 : 4;
        if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 0, pfx == 2 ? 8 : 4); push_r(e, EDI); }
        else { e8(e, 0x8d); e8(e, 0x85); e32(e, sx); push_r(e, EAX); }
        push_i(e, (u32)b | (u32)pfx << 8 | (u32)dsz << 16);
        push_cpu(e);
        call_abs(e, (const void *)sse_cvt_int);
        add_esp(e, 12);
        store_reg(t, in->reg, dsz, EAX, EDX);
        slow_continue(t, &s);
        return 1;
    }
    /* PSRLW/PSRAW/PSLLW, PSRLQ/PSLLQ, PSRLDQ/PSLLDQ by immediate */
    if ((jit_off & 65536) && ((b == 0x71 || b == 0x73) || (b == 0x70 && pfx >= 2) || b == 0x12 || b == 0x13 || b == 0x16 || b == 0x17)) return 0;
    if (pfx == 1 && (b == 0x71 || b == 0x73) && !mem) {
        int r = in->reg & 7;
        u32 n = (u32)(u8)in->imm, xo = X_OFF(in->rm);
        if (b == 0x71) {
            if (r != 2 && r != 4 && r != 6) return 0;
            if (n > 15 && r != 4) { for (int i = 0; i < 4; i++) mov_m_i(e, xo + 4u * (u32)i, 0); return 1; }
            for (int i = 0; i < 8; i++) {
                movx_m(e, r == 4 ? 0xbf : 0xb7, EAX, xo + 2u * (u32)i);
                shift_r_i(e, r == 2 ? 5 : r == 4 ? 7 : 4, EAX, (u8)(n > 15 ? 15 : n));
                e8(e, 0x66); mov_m_r(e, xo + 2u * (u32)i, EAX);
            }
            return 1;
        }
        if (r == 2 || r == 6) {
            if (n > 63) { for (int i = 0; i < 4; i++) mov_m_i(e, xo + 4u * (u32)i, 0); return 1; }
            for (int q = 0; q < 2; q++) {
                u32 o = xo + 8u * (u32)q;
                mov_r_m(e, EAX, o); mov_r_m(e, EDX, o + 4);
                if (r == 6) {
                    if (n >= 32) { mov_r_r(e, EDX, EAX); mov_r_i(e, EAX, 0); if (n > 32) shift_r_i(e, 4, EDX, (u8)(n - 32)); }
                    else if (n) { shld_r_r_i(e, EDX, EAX, (u8)n); shift_r_i(e, 4, EAX, (u8)n); }
                } else {
                    if (n >= 32) { mov_r_r(e, EAX, EDX); mov_r_i(e, EDX, 0); if (n > 32) shift_r_i(e, 5, EAX, (u8)(n - 32)); }
                    else if (n) { shrd_r_r_i(e, EAX, EDX, (u8)n); shift_r_i(e, 5, EDX, (u8)n); }
                }
                mov_m_r(e, o, EAX); mov_m_r(e, o + 4, EDX);
            }
            return 1;
        }
        if (r == 3 || r == 7) {
            if (n > 15) n = 16;
            for (int k = 0; k < 16; k++) {
                int i = r == 3 ? k : 15 - k;                       /* PSRLDQ ascending, PSLLDQ descending */
                int from = r == 3 ? i + (int)n : i - (int)n;
                if (from < 0 || from > 15) { e8(e, 0xc6); e8(e, 0x85); e32(e, xo + (u32)i); e8(e, 0); }
                else { e8(e, 0x8a); e8(e, 0x85); e32(e, xo + (u32)from); st8_m_r(e, xo + (u32)i, EAX); }
            }
            return 1;
        }
        return 0;
    }
    /* PSHUFLW (F2) / PSHUFHW (F3) */
    if (b == 0x70 && pfx >= 2) {
        if (mem) { emit_ea(t, in, 1); align16_check(t, &s); emit_tlb(t, &s, 0, 16); }
        u32 half = pfx == 2 ? 0 : 8, keep = pfx == 2 ? 8 : 0;
        static const int wr[4] = { EAX, ECX, EDX, EBX };
        for (int i = 0; i < 4; i++) {
            u32 o = half + 2u * (u32)i;
            if (mem) { e8(e, 0x0f); e8(e, 0xb7); e8(e, 0x40 | wr[i] << 3 | EDI); e8(e, o); }
            else movx_m(e, 0xb7, wr[i], sx + o);
        }
        for (int i = 0; i < 2; i++) {
            u32 o = keep + 4u * (u32)i;
            if (mem) { e8(e, 0xff); e8(e, 0x77); e8(e, (u8)o); }         /* push dword [edi+o] */
            else { e8(e, 0xff); e8(e, 0xb5); e32(e, sx + o); }            /* push dword [ebp+src] */
            e8(e, 0x8f); e8(e, 0x85); e32(e, dx + o);                     /* pop dword [ebp+dst] */
        }
        for (int i = 0; i < 4; i++) { e8(e, 0x66); mov_m_r(e, dx + half + 2u * (u32)i, wr[((u32)in->imm >> (2 * i)) & 3]); }
        slow_continue(t, &s);
        return 1;
    }
    /* MOVLPS/MOVHPS/MOVLPD/MOVHPD loads and stores, MOVHLPS/MOVLHPS */
    if ((b == 0x12 || b == 0x13 || b == 0x16 || b == 0x17) && pfx <= 1) {
        u32 hi = b >= 0x16 ? 8 : 0;
        if (b == 0x12 || b == 0x16) {
            if (!mem) {
                if (pfx == 1) return 0;
                copy_dw(e, EBP, dx + hi, EBP, sx + (8 - hi), 2);
                return 1;
            }
            emit_ea(t, in, 1); emit_tlb(t, &s, 0, 8);
            copy_dw(e, EBP, dx + hi, EDI, 0, 2);
        } else {
            if (!mem) return 0;
            emit_ea(t, in, 1); emit_tlb(t, &s, 1, 8);
            copy_dw(e, EDI, 0, EBP, dx + hi, 2);
        }
        slow_continue(t, &s);
        return 1;
    }
    /* packed integer: native PADDD/PSUBD, shifts by immediate, PSHUFD; helper for the rest */
    if (pfx == 1 && (b == 0x72) && !mem) {
        int r = in->reg & 7;
        if (r != 2 && r != 4 && r != 6) return 0;
        u32 n = (u32)(u8)in->imm;
        u32 xo = X_OFF(in->rm);
        for (int i = 0; i < 4; i++) {
            if (n > 31 && r != 4) { mov_m_i(e, xo + 4u * (u32)i, 0); continue; }
            mov_r_m(e, EAX, xo + 4u * (u32)i);
            shift_r_i(e, r == 2 ? 5 : r == 4 ? 7 : 4, EAX, (u8)(n > 31 ? 31 : n));
            mov_m_r(e, xo + 4u * (u32)i, EAX);
        }
        return 1;
    }
    if (pfx == 1 && (b == 0xfe || b == 0xfa || b == 0x70)) {
        int sb = EBP;
        u32 so = sx;
        if (mem) {
            emit_ea(t, in, 1);
            align16_check(t, &s);
            emit_tlb(t, &s, 0, 16);
            sb = EDI; so = 0;
        }
        if (b == 0x70) {
            static const int regs[4] = { EAX, ECX, EDX, EBX };
            for (int i = 0; i < 4; i++) {
                if (sb == EBP) mov_r_m(e, regs[i], so + 4u * (u32)i); else ld_b(e, regs[i], EDI, (u8)(4 * i));
            }
            for (int i = 0; i < 4; i++) mov_m_r(e, dx + 4u * (u32)i, regs[((u32)in->imm >> (2 * i)) & 3]);
        } else {
            for (int i = 0; i < 4; i++) {
                mov_r_m(e, EAX, dx + 4u * (u32)i);
                u8 opc = b == 0xfe ? 0x03 : 0x2b;
                if (sb == EBP) alu_r_m(e, opc, EAX, so + 4u * (u32)i);
                else { e8(e, opc); e8(e, 0x40 | EAX << 3 | EDI); e8(e, 4u * (u32)i); }
                mov_m_r(e, dx + 4u * (u32)i, EAX);
            }
        }
        slow_continue(t, &s);
        return 1;
    }
    /* PUNPCKLWD, PUNPCKLDQ, PCMPEQD, PCMPGTD natively */
    if (pfx == 1 && (b == 0x61 || b == 0x62 || b == 0x76 || b == 0x66)) {
        int sb = EBP;
        u32 so = sx;
        if (mem) {
            emit_ea(t, in, 1);
            align16_check(t, &s);
            emit_tlb(t, &s, 0, 16);
            sb = EDI; so = 0;
        }
#define SRC_LD(op2, r, off) do { if (sb == EBP) movx_m(e, op2, r, so + (off)); else { e8(e, 0x0f); e8(e, op2); e8(e, 0x40 | (r) << 3 | EDI); e8(e, (off)); } } while (0)
#define SRC_LD32(r, off) do { if (sb == EBP) mov_r_m(e, r, so + (off)); else ld_b(e, r, EDI, (u8)(off)); } while (0)
        if (b == 0x61) {          /* dword k = x.w[k] | s.w[k] << 16, highest first (in place) */
            for (int k = 3; k >= 0; k--) {
                movx_m(e, 0xb7, EAX, dx + 2u * (u32)k);
                SRC_LD(0xb7, ECX, 2u * (u32)k);
                shift_r_i(e, 4, ECX, 16);
                alu_r_r(e, 0x09, EAX, ECX);
                mov_m_r(e, dx + 4u * (u32)k, EAX);
            }
        } else if (b == 0x62) {   /* d3 = s.d1, d2 = x.d1, d1 = s.d0 */
            SRC_LD32(EAX, 4); mov_m_r(e, dx + 12, EAX);
            mov_r_m(e, EAX, dx + 4); mov_m_r(e, dx + 8, EAX);
            SRC_LD32(EAX, 0); mov_m_r(e, dx + 4, EAX);
        } else {
            for (int k = 0; k < 4; k++) {
                mov_r_m(e, EAX, dx + 4u * (u32)k);
                if (sb == EBP) alu_r_m(e, 0x3b, EAX, so + 4u * (u32)k);
                else { e8(e, 0x3b); e8(e, 0x40 | EAX << 3 | EDI); e8(e, 4u * (u32)k); }
                e8(e, 0x0f); e8(e, b == 0x76 ? 0x94 : 0x9f); e8(e, 0xc1);   /* sete/setg cl */
                e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc9);                      /* movzx ecx, cl */
                e8(e, 0xf7); e8(e, 0xd9);                                   /* neg ecx */
                mov_m_r(e, dx + 4u * (u32)k, ECX);
            }
        }
#undef SRC_LD
#undef SRC_LD32
        slow_continue(t, &s);
        return 1;
    }
    if (pfx == 1 && ((b >= 0x60 && b <= 0x6d) || (b >= 0x74 && b <= 0x76) ||
                     (b >= 0xd1 && b <= 0xfe && b != 0xd6 && b != 0xd7 && b != 0xe6 && b != 0xe7 && b != 0xf7))) {
        if (mem) {
            emit_ea(t, in, 1);
            align16_check(t, &s);
            emit_tlb(t, &s, 0, 16);
            push_r(e, EDI);
        } else {
            e8(e, 0x8d); e8(e, 0x85); e32(e, sx);            /* lea eax, [ebp+src] */
            push_r(e, EAX);
        }
        e8(e, 0x8d); e8(e, 0x85); e32(e, dx);                /* lea eax, [ebp+dst] */
        push_r(e, EAX);
        push_i(e, (u32)b);
        call_abs(e, (const void *)sse_int_binop);
        add_esp(e, 12);
        slow_continue(t, &s);
        return 1;
    }
    if ((pfx == 3 && b == 0x7e) || (pfx == 1 && b == 0xd6)) {
        int load = b == 0x7e;
        if (mem) {
            emit_ea(t, in, 1);
            emit_tlb(t, &s, !load, 8);
            if (load) { copy_dw(e, EBP, dx, EDI, 0, 2); mov_m_i(e, dx + 8, 0); mov_m_i(e, dx + 12, 0); }
            else copy_dw(e, EDI, 0, EBP, dx, 2);
        } else {
            u32 to = load ? dx : sx, from = load ? sx : dx;
            copy_dw(e, EBP, to, EBP, from, 2);
            mov_m_i(e, to + 8, 0); mov_m_i(e, to + 12, 0);
        }
        slow_continue(t, &s);
        return 1;
    }
    return 0;
}

/* SSE4.1 subset: PMOVZX/PMOVSX (66 0F38 20-25, 30-35), PEXTRB/W/D/Q (66 0F3A 14-16),
 * PINSRD/Q (66 0F3A 22) */
static int t_sse4(T *t, Cpu *c, const Ins *in) {
    E *e = &t->e;
    if (!sse_ok(c) || in->lock || in->a67 || !in->p66 || in->rep) return 0;
    int op = in->op;
    u32 dx = X_OFF(in->reg);
    int mem = in->mod != 3;
    Slow s = {0};
    if ((op >= 0x220 && op <= 0x225) || (op >= 0x230 && op <= 0x235)) {
        static const int srcb[6] = { 8, 4, 2, 8, 4, 8 }, ssz[6] = { 1, 1, 1, 2, 2, 4 }, dsz[6] = { 2, 4, 8, 4, 8, 8 };
        int k = op & 7, zx = op >= 0x230;
        u32 tmp = OFF(jit_x);
        if (mem) {
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 0, srcb[k]);
            if (srcb[k] == 2) { movx_b(e, 0xb7, EAX, EDI); mov_m_r(e, tmp, EAX); }
            else copy_dw(e, EBP, tmp, EDI, 0, srcb[k] / 4);
        } else copy_dw(e, EBP, tmp, EBP, X_OFF(in->rm), 2);
        int lanes = 16 / dsz[k];
        for (int i = 0; i < lanes; i++) {
            u32 so = tmp + (u32)(i * ssz[k]), d = dx + (u32)(i * dsz[k]);
            if (ssz[k] == 4) mov_r_m(e, EAX, so);
            else movx_m(e, (u8)((zx ? 0xb6 : 0xbe) | (ssz[k] == 2)), EAX, so);
            if (dsz[k] == 2) { e8(e, 0x66); mov_m_r(e, d, EAX); }
            else if (dsz[k] == 4) mov_m_r(e, d, EAX);
            else {
                mov_m_r(e, d, EAX);
                if (zx) mov_m_i(e, d + 4, 0);
                else { e8(e, 0x99); mov_m_r(e, d + 4, EDX); }   /* cdq */
            }
        }
        slow_continue(t, &s);
        return 1;
    }
    if (op >= 0x314 && op <= 0x316) {
        u32 imm = (u32)(u8)in->imm;
        int sz = op == 0x314 ? 1 : op == 0x315 ? 2 : in->w ? 8 : 4;
        u32 so = dx + (sz == 1 ? (imm & 15) : sz == 2 ? 2 * (imm & 7) : sz == 4 ? 4 * (imm & 3) : 8 * (imm & 1));
        if (mem) { emit_ea(t, in, 1); emit_tlb(t, &s, 1, sz); }
        if (sz == 1) movx_m(e, 0xb6, ECX, so); else if (sz == 2) movx_m(e, 0xb7, ECX, so); else mov_r_m(e, ECX, so);
        if (sz == 8) mov_r_m(e, EBX, so + 4);
        if (!mem) {
            mov_m_r(e, R_LO(in->rm), ECX);
            if (sz == 8) mov_m_r(e, R_HI(in->rm), EBX); else mov_m_i(e, R_HI(in->rm), 0);
            return 1;
        }
        st_hmem(t, sz, ECX, EBX);
        slow_continue(t, &s);
        return 1;
    }
    if (op == 0x322) {
        int sz = in->w ? 8 : 4;
        u32 d = dx + (sz == 4 ? 4 * ((u32)in->imm & 3) : 8 * ((u32)in->imm & 1));
        if (mem) {
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 0, sz);
            copy_dw(e, EBP, d, EDI, 0, sz / 4);
        } else copy_dw(e, EBP, d, EBP, R_LO(in->rm), sz / 4);
        slow_continue(t, &s);
        return 1;
    }
    return 0;
}

/* MUL/IMUL/DIV/IDIV (F6/F7 /4-/7) through the interpreter's arithmetic: may
 * raise #DE, so RIP and icount are made exact around the call */
static int t_muldiv(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->op == 0xf6 ? 1 : in->osz;
    if (in->rep || in->lock || sz == 2) return 0;
    Slow s = {0};
    if (in->mod != 3) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 0, sz);
        ld_hmem(t, ECX, sz);
        if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    } else {
        ld_gpr(t, in, ECX, in->rm, sz);
        if (sz == 8) mov_r_m(e, EBX, R_HI(in->rm)); else mov_r_i(e, EBX, 0);
    }
    int r = in->reg & 7;
    if (r == 4 && sz >= 4 && !(jit_off & 0x200000)) {
        /* MUL: RDX:RAX = RAX * src; CF = OF = high half != 0 (LF_MUL, result = low half) */
        u32 x = OFF(jit_x);
        if (sz == 4) {
            mov_r_m(e, EAX, R_LO(0));
            e8(e, 0xf7); e8(e, 0xe1);                                   /* mul ecx */
            mov_m_r(e, R_LO(0), EAX); mov_m_i(e, R_HI(0), 0);
            mov_m_r(e, R_LO(2), EDX); mov_m_i(e, R_HI(2), 0);
            if (!t->flags_dead) {
                mov_m_r(e, OFF(lf_res), EAX);
                test_r_r(e, EDX, EDX);
            }
        } else {
            mov_r_m(e, EAX, R_LO(0)); e8(e, 0xf7); e8(e, 0xe1);         /* A0*B0 */
            mov_m_r(e, x, EAX); mov_m_r(e, x + 4, EDX);                 /* x0, c1 */
            mov_r_m(e, EAX, R_LO(0)); e8(e, 0xf7); e8(e, 0xe3);         /* A0*B1 */
            e8(e, 0x01); e8(e, 0x85); e32(e, x + 4);                    /* add [c1], eax */
            alu_r_i(e, 2, EDX, 0);
            mov_m_r(e, x + 8, EDX); mov_m_i(e, x + 12, 0);              /* c2, c3 */
            mov_r_m(e, EAX, R_HI(0)); e8(e, 0xf7); e8(e, 0xe1);         /* A1*B0 */
            e8(e, 0x01); e8(e, 0x85); e32(e, x + 4);                    /* add [c1], eax */
            e8(e, 0x11); e8(e, 0x95); e32(e, x + 8);                    /* adc [c2], edx */
            alu_m_i(e, 2, x + 12, 0);                                   /* adc [c3], 0 */
            mov_r_m(e, EAX, R_HI(0)); e8(e, 0xf7); e8(e, 0xe3);         /* A1*B1 */
            e8(e, 0x01); e8(e, 0x85); e32(e, x + 8);                    /* add [c2], eax */
            e8(e, 0x11); e8(e, 0x95); e32(e, x + 12);                   /* adc [c3], edx */
            mov_r_m(e, EAX, x);
            mov_m_r(e, R_LO(0), EAX); mov_r_m(e, EDX, x + 4); mov_m_r(e, R_HI(0), EDX);
            mov_r_m(e, ECX, x + 8); mov_r_m(e, EBX, x + 12);
            mov_m_r(e, R_LO(2), ECX); mov_m_r(e, R_HI(2), EBX);
            if (!t->flags_dead) {
                mov_m_r(e, OFF(lf_res), EAX); mov_m_r(e, OFF(lf_res) + 4, EDX);
                alu_r_r(e, 0x09, ECX, EBX);                             /* or ecx, ebx: high != 0 */
            }
        }
        if (!t->flags_dead) {
            e8(e, 0x0f); e8(e, 0x95); e8(e, 0xc0);                      /* setnz al */
            e8(e, 0x0f); e8(e, 0xb6); e8(e, 0xc0);                      /* movzx eax, al */
            mov_m_r(e, OFF(lf_c), EAX);
            mov_m_i(e, OFF(lf_op), (u32)LF_MUL | (u32)sz << 16);
        }
        slow_continue(t, &s);
        return 1;
    }
    if (r >= 6) {   /* DIV/IDIV may raise #DE: rip and icount exact */
        set_rip(t, t->rip);
        emit_icount_exact(t, t->pending);
    }
    push_i(e, (u32)sz); push_r(e, EBX); push_r(e, ECX); push_i(e, (u32)(in->reg & 7)); push_cpu(e);
    call_abs(e, (const void *)cpu_muldiv);
    add_esp(e, 20);
    slow_continue(t, &s);
    return 1;
}

/* CBW/CWDE/CDQE (98), CWD/CDQ/CQO (99) */
static int t_cext(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (in->rep || in->lock) return 0;
    if (in->op == 0x98) {
        if (sz == 8) { mov_r_m(e, EAX, R_LO(0)); e8(e, 0x99); store_reg(t, 0, 8, EAX, EDX); }
        else if (sz == 4) { movx_m(e, 0xbf, EAX, R_LO(0)); store_reg(t, 0, 4, EAX, EDX); }
        else { movx_m(e, 0xbe, EAX, R_LO(0)); e8(e, 0x66); mov_m_r(e, R_LO(0), EAX); }
    } else {
        if (sz == 8) { mov_r_m(e, EAX, R_HI(0)); shift_r_i(e, 7, EAX, 31); store_reg(t, 2, 8, EAX, EAX); }
        else if (sz == 4) { mov_r_m(e, EAX, R_LO(0)); shift_r_i(e, 7, EAX, 31); store_reg(t, 2, 4, EAX, EDX); }
        else { movx_m(e, 0xbf, EAX, R_LO(0)); shift_r_i(e, 7, EAX, 15); e8(e, 0x66); mov_m_r(e, R_LO(2), EAX); }
    }
    return 1;
}

/* F7: TEST r/m, imm (/0), NOT (/2), NEG (/3) on 32/64-bit operands */
static int t_grp3(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz, r = in->reg & 7;
    if (sz < 4 || in->rep || in->lock || (r != 0 && r != 2 && r != 3)) return 0;
    Slow s = {0};
    int mem = in->mod != 3;
    if (mem) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, r != 0, sz);
        ld_b(e, ECX, EDI, 0);
        if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    } else load_reg_cb(t, in->rm, sz);
    if (r == 2) {
        e8(e, 0xf7); e8(e, 0xd1);                       /* not ecx */
        if (sz == 8) { e8(e, 0xf7); e8(e, 0xd3); }      /* not ebx */
        mov_r_r(e, EAX, ECX); mov_r_r(e, EDX, EBX);
    } else {
        if (r == 0) {          /* a = operand, b = imm */
            mov_r_r(e, EAX, ECX); mov_r_r(e, EDX, EBX);
            mov_r_i(e, ECX, (u32)in->imm);
            mov_r_i(e, EBX, sz == 8 ? (u32)((u64)in->imm >> 32) : 0);
        } else {               /* NEG: a = 0, b = operand */
            mov_r_i(e, EAX, 0); mov_r_i(e, EDX, 0);
        }
        mov_m_r(e, OFF(lf_a), EAX); mov_m_r(e, OFF(lf_a) + 4, EDX);
        mov_m_r(e, OFF(lf_b), ECX); mov_m_r(e, OFF(lf_b) + 4, EBX);
        if (r == 0) { alu_r_r(e, 0x21, EAX, ECX); if (sz == 8) alu_r_r(e, 0x21, EDX, EBX); }
        else { alu_r_r(e, 0x29, EAX, ECX); if (sz == 8) alu_r_r(e, 0x19, EDX, EBX); }
        mov_m_r(e, OFF(lf_res), EAX); mov_m_r(e, OFF(lf_res) + 4, EDX);
        mov_m_i(e, OFF(lf_op), (u32)(r == 0 ? LF_LOGIC : LF_SUB) | (u32)((u32)sz) << 16);
    }
    if (r != 0) {
        if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
        else store_reg(t, in->rm, sz, EAX, EDX);
    }
    slow_continue(t, &s);
    return 1;
}

/* SHL/SHR/SAR r/m, CL on 32/64-bit operands. Count 0 leaves the flags alone
 * (the operand is still written back, as the interpreter does). */
static int t_shift_cl(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->op == 0xd2 ? 1 : in->osz, op = in->reg & 7;
    if (in->rep || in->lock || op < 4) return 0;
    Slow s = {0};
    int mem = in->mod != 3;
    if (sz < 4) {   /* 8/16-bit: the host shift by CL (masked to 5 bits) matches the guest's */
        if (mem) {
            if (!mem_ok(in)) return 0;
            emit_ea(t, in, 1);
            emit_tlb(t, &s, 1, sz);
            ld_hmem(t, EAX, sz);
        } else ld_gpr(t, in, EAX, in->rm, sz);
        movx_m(e, 0xb6, ECX, R_LO(1));
        alu_r_i(e, 4, ECX, 31);
        u8 *zero = jcc32(e, CC_E);
        mov_m_r(e, OFF(lf_a), EAX); mov_m_i(e, OFF(lf_a) + 4, 0);
        mov_m_r(e, OFF(lf_b), ECX); mov_m_i(e, OFF(lf_b) + 4, 0);
        int ext = op == 6 ? 4 : op;
        if (sz == 2) e8(e, 0x66);
        e8(e, sz == 1 ? 0xd2 : 0xd3); e8(e, 0xc0 | ext << 3 | EAX);          /* upper bits of EAX stay 0 */
        mov_m_r(e, OFF(lf_res), EAX); mov_m_i(e, OFF(lf_res) + 4, 0);
        mov_m_i(e, OFF(lf_op), (u32)(ext == 4 ? LF_SHL : ext == 5 ? LF_SHR : LF_SAR) | (u32)((u32)sz) << 16);
        patch(e, zero, e->p);
        if (mem) st_hmem(t, sz, EAX, EDX);
        else st_gpr(t, in, in->rm, sz, EAX, EDX);
        slow_continue(t, &s);
        return 1;
    }
    if (mem) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_b(e, EAX, EDI, 0);
        if (sz == 8) ld_b(e, EDX, EDI, 4); else mov_r_i(e, EDX, 0);
    } else {
        mov_r_m(e, EAX, R_LO(in->rm));
        if (sz == 8) mov_r_m(e, EDX, R_HI(in->rm)); else mov_r_i(e, EDX, 0);
    }
    movx_m(e, 0xb6, ECX, R_LO(1));
    alu_r_i(e, 4, ECX, sz == 8 ? 63 : 31);
    u8 *zero = jcc32(e, CC_E);                      /* flags from the AND: count == 0 */
    mov_m_r(e, OFF(lf_a), EAX); mov_m_r(e, OFF(lf_a) + 4, EDX);
    mov_m_r(e, OFF(lf_b), ECX); mov_m_i(e, OFF(lf_b) + 4, 0);
    int ext = op == 6 ? 4 : op;                     /* SAL = SHL */
    if (sz == 4) { e8(e, 0xd3); e8(e, 0xc0 | ext << 3 | EAX); }
    else {
        if (ext == 4) { e8(e, 0x0f); e8(e, 0xa5); e8(e, 0xc0 | EAX << 3 | EDX); }   /* shld edx, eax, cl */
        else { e8(e, 0x0f); e8(e, 0xad); e8(e, 0xc0 | EDX << 3 | EAX); }            /* shrd eax, edx, cl */
        e8(e, 0xd3); e8(e, 0xc0 | ext << 3 | (ext == 4 ? EAX : EDX));
        e8(e, 0xf6); e8(e, 0xc1); e8(e, 32);                                         /* test cl, 32 */
        u8 *small = jcc32(e, CC_E);
        if (ext == 4) { mov_r_r(e, EDX, EAX); mov_r_i(e, EAX, 0); }
        else if (ext == 5) { mov_r_r(e, EAX, EDX); mov_r_i(e, EDX, 0); }
        else { mov_r_r(e, EAX, EDX); shift_r_i(e, 7, EDX, 31); }
        patch(e, small, e->p);
    }
    mov_m_r(e, OFF(lf_res), EAX); mov_m_r(e, OFF(lf_res) + 4, EDX);
    mov_m_i(e, OFF(lf_op), (u32)(ext == 4 ? LF_SHL : ext == 5 ? LF_SHR : LF_SAR) | (u32)((u32)sz) << 16);
    patch(e, zero, e->p);
    if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
    else store_reg(t, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* shifts/rotates the native path does not cover (by CL, rotates): interpreter helper */
static int t_shift_helper(T *t, const Ins *in) {
    E *e = &t->e;
    int sz = in->osz;
    if (sz < 4 || in->rep || in->lock || (in->reg & 7) == 2 || (in->reg & 7) == 3) return 0;
    Slow s = {0};
    int mem = in->mod != 3;
    if (mem) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, sz);
        ld_b(e, ECX, EDI, 0);
        if (sz == 8) ld_b(e, EBX, EDI, 4); else mov_r_i(e, EBX, 0);
    } else load_reg_cb(t, in->rm, sz);
    push_i(e, (u32)sz);
    if (in->op == 0xd3) { movx_m(e, 0xb6, EAX, R_LO(1)); push_r(e, EAX); }
    else push_i(e, in->op == 0xd1 ? 1u : (u32)(u8)in->imm);
    push_r(e, EBX); push_r(e, ECX);
    push_i(e, (u32)(in->reg & 7));
    push_cpu(e);
    prof_inc_e(e, &ph_shift[in->reg & 7]);
    call_abs(e, (const void *)cpu_shift);
    add_esp(e, 24);
    if (mem) { st_b(e, EDI, 0, EAX); if (sz == 8) st_b(e, EDI, 4, EDX); }
    else store_reg(t, in->rm, sz, EAX, EDX);
    slow_continue(t, &s);
    return 1;
}

/* SETcc r/m8 */
static int t_setcc(T *t, const Ins *in) {
    E *e = &t->e;
    int cc = in->op & 15, fz = t->fuse_now;
    if (in->rep || in->lock) return 0;
    Slow s = {0};
    (void)fz;
    if (in->mod != 3) {
        if (!mem_ok(in)) return 0;
        emit_ea(t, in, 1);
        emit_tlb(t, &s, 1, 1);
    }
    int hc = emit_host_cc(t, cc, in->mod == 3);
    if (hc >= 0) { e8(e, 0x0f); e8(e, 0x90 | hc); e8(e, 0xc0); }       /* setcc al */
    else call_cond(t, cc);                                               /* EAX = 0/1, EDI kept */
    if (in->mod == 3) st8_m_r(e, r8_off(in, in->rm), EAX);
    else { e8(e, 0x88); e8(e, 0x47); e8(e, 0); }                         /* mov [edi], al */
    slow_continue(t, &s);
    return 1;
}

/* PUSH imm, LEAVE, CLD/STD, NOPs */
static int t_misc(T *t, const Ins *in) {
    E *e = &t->e;
    int op = in->op;
    if (in->lock) return 0;
    if (op == 0x90 && !in->b && in->rep == 0xf3 && t->j->m->ncpu > 1) return 0;   /* PAUSE with several CPUs: interpreted, ends the block (yield) */
    if (op == 0x90 && !in->b) { if (in->rep == 0xf3) prof_inc_e(e, &ph_pause); return 1; }   /* NOP / PAUSE */
    if ((op >= 0x118 && op <= 0x11f) || op == 0x10d) return 1;          /* hint NOPs, PREFETCH */
    if (op == 0xfc) { alu_m_i(e, 4, OFF(rflags), ~(u32)F_DF); return 1; }
    if (op == 0xfd) { alu_m_i(e, 1, OFF(rflags), F_DF); return 1; }
    if (in->p66 || in->rep) return 0;
    Slow s = {0};
    if (op == 0x68 || op == 0x6a) {
        emit_push(t, &s, -1, (u64)in->imm);
        slow_continue(t, &s);
        return 1;
    }
    if (op == 0xc9 && t->m32) {                                          /* LEAVE: ESP = EBP, pop EBP */
        mov_r_m(e, EAX, R_LO(5)); mov_r_i(e, EDX, 0);
        emit_tlb(t, &s, 0, 4);
        ld_b(e, ECX, EDI, 0);
        alu_r_i(e, 0, EAX, 4);
        mov_m_r(e, R_LO(4), EAX); mov_m_i(e, R_HI(4), 0);
        mov_m_r(e, R_LO(5), ECX); mov_m_i(e, R_HI(5), 0);
        slow_continue(t, &s);
        return 1;
    }
    if (op == 0xc9) {
        mov_r_m(e, EAX, R_LO(5)); mov_r_m(e, EDX, R_HI(5));
        emit_tlb(t, &s, 0, 8);
        ld_b(e, ECX, EDI, 0); ld_b(e, EBX, EDI, 4);
        alu_r_i(e, 0, EAX, 8); alu_r_i(e, 2, EDX, 0);
        mov_m_r(e, R_LO(4), EAX); mov_m_r(e, R_HI(4), EDX);
        store_reg(t, 5, 8, ECX, EBX);
        slow_continue(t, &s);
        return 1;
    }
    return 0;
}

static int t_native(T *t, Cpu *c, const Ins *in) {
    int op = in->op;
    if (op == 0x90 || (op >= 0x118 && op <= 0x11f) || op == 0x10d || op == 0xfc || op == 0xfd ||
        op == 0x68 || op == 0x6a || op == 0xc9) return t_misc(t, in);
    if (op >= 0x190 && op <= 0x19f) return t_setcc(t, in);
    if (op == 0x1ae && in->mod == 3 && (in->reg & 7) >= 5) return 1;   /* LFENCE/MFENCE/SFENCE: one CPU, program order */
    if (op == 0x101 && in->mod == 3 && (in->reg & 7) == 7 && (in->rm & 7) == 1 && !in->rep && !in->lock && !in->p66) {   /* RDTSCP */
        emit_icount_exact(t, t->pending);
        push_cpu(&t->e); call_abs(&t->e, (const void *)cpu_jit_rdtscp); add_esp(&t->e, 4);
        return 1;
    }
    if (op == 0x1c7 && (in->reg & 7) == 1 && in->mod != 3 && !(jit_off & 1024)) return t_cx(t, in);
    if (op == 0x1ae && in->mod != 3 && ((in->reg & 7) == 2 || (in->reg & 7) == 3)) return t_mxcsr(t, c, in);
    if (op == 0x122 && in->mod == 3 && in->reg == 8 && !in->rep && !in->lock && !t->m32 && !(jit_off & 16384 * 64)) return t_sys(t, in);
    if (!(jit_off & 16384 * 64) && (op == 0xfa || op == 0xfb || op == 0x120 ||
        (op == 0x101 && in->mod == 3 && (in->reg & 7) == 7 && (in->rm & 7) == 0))) return t_sys(t, in);
    if ((op == 0x1ba || op == 0x1a3 || op == 0x1ab || op == 0x1b3 || op == 0x1bb) && !(jit_off & 128) && t_bt(t, in)) return 1;
    if ((op == 0x1b1 || op == 0x1c1) && !(jit_off & 1024)) return t_cmpxchg_xadd(t, in);
    if (op >= 0x1c8 && op <= 0x1cf && !(jit_off & 2048)) return t_bswap(t, in);
    if ((op == 0x1bc || op == 0x1bd) && !(jit_off & 4096)) return t_bsf(t, in);
    if (op == 0x80 || op == 0x84 || (op < 0x40 && ((op & 7) == 0 || (op & 7) == 2 || (op & 7) == 4)))
        return t_alu(t, in);
    if (op >= 0x200) return (jit_off & 512) ? 0 : t_sse4(t, c, in);
    if (((op == 0xf6 || op == 0xf7) && (in->reg & 7) == 0) || op == 0xa8 || op == 0xa9) return (jit_off & 2) ? 0 : t_alu(t, in);
    if ((op == 0xfe || op == 0xff) && (in->reg & 7) < 2) return (jit_off & 1) ? 0 : t_incdec(t, in);
    if ((op == 0xc1 || op == 0xd1) && (in->reg & 7) < 2 && !(jit_off & 64) && (t->flags_dead || !(jit_off & 0x80000000u)) && t_rot(t, in)) return 1;
    if ((op == 0xf6 || op == 0xf7) && (in->reg & 7) >= 4) return t_muldiv(t, in);
    if ((op == 0xf6 || (op == 0xf7 && in->osz == 2)) && (in->reg & 7) == 3) return t_neg_small(t, in);
    if (op == 0xf7) return t_grp3(t, in);
    if (op == 0x87) return t_xchg(t, in);
    if (op == 0x9c || (op == 0xff && (in->reg & 7) == 6)) return t_push_more(t, in);
    if (op == 0x98 || op == 0x99) return t_cext(t, in);
    if ((op == 0xc1 || op == 0xd1 || op == 0xc0 || op == 0xd0) && (in->reg & 7) >= 4 && !(jit_off & 8192)) return t_shift(t, in);
    if (op == 0xf6 && (in->reg & 7) == 2 && !(jit_off & 8192)) return t_not8(t, in);
    if ((op == 0xd3 || op == 0xd2) && (in->reg & 7) >= 4 && t_shift_cl(t, in)) return 1;
    if (op == 0xc1 || op == 0xd1 || op == 0xd3) return t_shift_helper(t, in);
    if (op == 0x88 || op == 0x8a || op == 0xc6 || (op >= 0xb0 && op <= 0xb7)) return t_mov8(t, in);
    if (op == 0x1af || op == 0x69 || op == 0x6b) return t_imul(t, in);
    if (op >= 0x140 && op <= 0x14f) return t_cmov(t, in);
    if (op >= 0x110 && op <= 0x1ff && op != 0x1b6 && op != 0x1b7 && op != 0x1be && op != 0x1bf && !(op >= 0x180 && op <= 0x18f)) return t_sse(t, c, in);
    if (op == 0x89 || op == 0x8b || op == 0xc7 || op == 0x8d || (op >= 0xb8 && op <= 0xbf)) return t_mov(t, in);
    if ((op < 0x40 && ((op & 7) == 1 || (op & 7) == 3 || (op & 7) == 5)) || op == 0x81 || op == 0x83 || op == 0x85)
        return t_alu(t, in);
    if (op == 0x1b6 || op == 0x1b7 || op == 0x1be || op == 0x1bf || op == 0x63) return t_movx(t, in);
    if (op >= 0x50 && op <= 0x5f) return t_stack(t, in);
    return t_branch(t, in);
}

/* ================= flag liveness ================= */

enum { K_BARRIER, K_NEUTRAL, K_WRITE_REG, K_WRITE_MEM, K_PASS, K_PASS_MEM, K_READ_WRITE };

/* Conservative mirror of the native translators: K_NEUTRAL = native,
 * register-only, cannot fault, no flag access; K_WRITE_* = native and
 * replaces the whole flag state (REG: also cannot fault). Anything else is a
 * barrier (may read flags, fault, call the interpreter or leave the block). */
static int fclass(const Ins *in) {
    int op = in->op;
    int basic = in->osz >= 4 && !in->rep && !in->lock;
    if ((op < 0x40 && ((op & 7) == 1 || (op & 7) == 3 || (op & 7) == 5)) || op == 0x81 || op == 0x83 || op == 0x85) {
        int ext = op < 0x40 ? op >> 3 : op == 0x85 ? 4 : in->reg & 7;
        if (!basic) return K_BARRIER;
        if (ext == 2 || ext == 3) return in->a67 ? K_BARRIER : K_READ_WRITE;
        int mem = (op < 0x40 && (op & 7) == 5) ? 0 : in->mod != 3;
        if (mem) return in->a67 ? K_BARRIER : K_WRITE_MEM;
        return K_WRITE_REG;
    }
    if ((op == 0xc1 || op == 0xd1) && (in->reg & 7) >= 4 && basic) {
        u32 n = op == 0xd1 ? 1u : (u32)in->imm & (in->osz == 8 ? 63u : 31u);
        if (!n) return K_BARRIER;
        if (in->mod != 3) return in->a67 ? K_BARRIER : K_WRITE_MEM;
        return K_WRITE_REG;
    }
    if ((op == 0xa8 || op == 0xa9) && !in->rep && !in->lock) return K_WRITE_REG;
    if ((op == 0xf6 || op == 0xf7) && (in->reg & 7) == 0 && !in->rep && !in->lock)
        return in->mod == 3 ? K_WRITE_REG : in->a67 ? K_BARRIER : K_WRITE_MEM;
    /* INC/DEC keep CF; ROL/ROR keep everything but CF/OF: their flags pass through */
    if ((op == 0xfe || op == 0xff) && (in->reg & 7) < 2 && !in->rep)
        return in->mod == 3 ? (in->lock ? K_BARRIER : K_PASS) : in->a67 ? K_BARRIER : K_PASS_MEM;
    if ((op == 0xc1 || op == 0xd1) && (in->reg & 7) < 2 && basic && in->mod == 3) return K_PASS;
    if ((op == 0x1af || op == 0x69 || op == 0x6b) && in->osz >= 4 && !in->rep && !in->lock)   /* IMUL r, r/m(, imm) */
        return in->mod == 3 ? K_WRITE_REG : in->a67 ? K_BARRIER : K_WRITE_MEM;
    if (op == 0xf7 && (in->reg & 7) == 4 && in->osz >= 4 && !in->rep && !in->lock)
        return in->mod == 3 ? K_WRITE_REG : in->a67 ? K_BARRIER : K_WRITE_MEM;
    if ((op == 0x1b1 || op == 0x1c1 || op == 0x1bc || op == 0x1bd) && in->osz >= 4 && !in->rep)
        return in->mod == 3 ? (in->lock ? K_BARRIER : K_WRITE_REG) : in->a67 ? K_BARRIER : K_WRITE_MEM;
    if (op == 0x1ba && (in->reg & 7) >= 4 && in->osz >= 4 && !in->rep)
        return in->mod == 3 ? (in->lock ? K_BARRIER : K_PASS) : in->a67 ? K_BARRIER : K_PASS_MEM;
    if ((op == 0x1a3 || op == 0x1ab || op == 0x1b3 || op == 0x1bb) && in->osz >= 4 && !in->rep && !in->lock)
        return in->mod == 3 ? K_PASS : in->a67 ? K_BARRIER : K_PASS_MEM;
    if (basic && in->mod == 3 && (op == 0x89 || op == 0x8b || op == 0x1b6 || op == 0x1b7 || op == 0x1be || op == 0x1bf))
        return K_NEUTRAL;
    if (op == 0x63 && in->w && in->mod == 3 && !in->rep) return K_NEUTRAL;
    if (op >= 0xb8 && op <= 0xbf && basic) return K_NEUTRAL;
    if (op == 0xc7 && (in->reg & 7) == 0 && in->mod == 3 && basic) return K_NEUTRAL;
    if (op == 0x8d && basic && !in->a67 && in->mod != 3) return K_NEUTRAL;
    if ((op == 0x88 || op == 0x8a) && in->mod == 3 && !in->rep && !in->lock) return K_NEUTRAL;
    if (op == 0xc6 && (in->reg & 7) == 0 && in->mod == 3 && !in->rep && !in->lock) return K_NEUTRAL;
    if (op >= 0xb0 && op <= 0xb7 && !in->rep && !in->lock) return K_NEUTRAL;
    return K_BARRIER;
}

/* natively translated forms of these never write the flags (memory forms too) */
static int keeps_flags(const Ins *in) {
    int op = in->op;
    if (op == 0x88 || op == 0x89 || op == 0x8a || op == 0x8b || op == 0x8d || op == 0x63) return 1;
    if ((op == 0xc6 || op == 0xc7) && (in->reg & 7) == 0) return 1;
    if ((op >= 0x50 && op <= 0x5f) || (op >= 0xb0 && op <= 0xbf) || op == 0x68 || op == 0x6a) return 1;
    if (op == 0x1b6 || op == 0x1b7 || op == 0x1be || op == 0x1bf || (op >= 0x1c8 && op <= 0x1cf)) return 1;
    if (op == 0xfa || op == 0xfb || op == 0x120 || op == 0x101) return 1;
    if ((op >= 0x70 && op <= 0x7f) || (op >= 0x180 && op <= 0x18f)) return 1;   /* Jcc (superblock side exit) */
    if (op == 0xeb || op == 0xe9) return 1;                                         /* inlined jmp */
    if ((op >= 0x140 && op <= 0x14f) || (op >= 0x190 && op <= 0x19f)) return 1;
    if (op == 0x110 || op == 0x111 || op == 0x128 || op == 0x129 || op == 0x16f || op == 0x17f ||
        op == 0x1e7 || op == 0x12b || op == 0x16e || op == 0x17e || op == 0x1d6) return 1;
    return 0;
}

/* ================= block cache ================= */

static void flush_all(Machine *m) {
    Jit *j = m->jit;
    j->code_ptr = j->code;
    j->cold_ptr = j->cold;
    j->pcode_ptr = j->pcode;
    j->nblocks = 0;
    j->nlinks = 0;
    cmap_n = 0;
    e_memset(j->hash, 0, HASH_SIZE * sizeof *j->hash);
    e_memset(j->page_head, 0, j->npages * sizeof *j->page_head);
    e_memset(j->code_bits, 0, (j->npages + 7) / 8);
    e_memset(j->jc, 0, JC_SIZE * sizeof *j->jc);
    j->flushes++;
    j->gen++;
}

static u32 hash_pa(u64 pa) { return (u32)(pa ^ (pa >> 20)) & (HASH_SIZE - 1); }

void jit_invalidate_page(Machine *m, u64 pa);
/* retire one block (its replacement exists): entry exits to C, links unlinked, out of the hash */
static void block_retire(Jit *j, JitBlock *b) {
    b->valid = 0;
    if (b->entry) {
        u32 rel = (u32)(uintptr_t)b->inval - (u32)(uintptr_t)(b->entry + 5);
        b->entry[0] = 0xe9; b->entry[1] = (u8)rel; b->entry[2] = (u8)(rel >> 8); b->entry[3] = (u8)(rel >> 16); b->entry[4] = (u8)(rel >> 24);
    }
    for (u32 l = b->in_links; l; l = j->links[l - 1].next) {
        u8 *site = j->links[l - 1].site;
        u32 r = (u32)(uintptr_t)j->links[l - 1].stub - (u32)(uintptr_t)(site + 4);
        site[0] = (u8)r; site[1] = (u8)(r >> 8); site[2] = (u8)(r >> 16); site[3] = (u8)(r >> 24);
    }
    b->in_links = 0;
    JitBlock **pp = &j->hash[hash_pa(b->pa)];
    while (*pp && *pp != b) pp = &(*pp)->hnext;
    if (*pp) *pp = b->hnext;
}

void jit_invalidate_page(Machine *m, u64 pa) {
    Jit *j = m->jit;
    u32 pg = (u32)(pa >> 12);
    if (pg >= j->npages || !(j->code_bits[pg >> 3] & (1u << (pg & 7)))) return;
    j->code_bits[pg >> 3] &= (u8)~(1u << (pg & 7));
    for (JitBlock *b = j->page_head[pg]; b; b = b->pnext) {
        b->valid = 0;
        if (b->entry) {
            u32 rel = (u32)(uintptr_t)b->inval - (u32)(uintptr_t)(b->entry + 5);
            b->entry[0] = 0xe9; b->entry[1] = (u8)rel; b->entry[2] = (u8)(rel >> 8); b->entry[3] = (u8)(rel >> 16); b->entry[4] = (u8)(rel >> 24);
        }
        for (u32 l = b->in_links; l; l = j->links[l - 1].next) {
            u8 *site = j->links[l - 1].site;    /* back to the link stub */
            u32 r = (u32)(uintptr_t)j->links[l - 1].stub - (u32)(uintptr_t)(site + 4);
            site[0] = (u8)r; site[1] = (u8)(r >> 8); site[2] = (u8)(r >> 16); site[3] = (u8)(r >> 24);
        }
        b->in_links = 0;
        JitBlock **pp = &j->hash[hash_pa(b->pa)];
        while (*pp && *pp != b) pp = &(*pp)->hnext;
        if (*pp) *pp = b->hnext;
    }
    j->page_head[pg] = 0;
    j->invalidations++;
}

int jit_page_is_code(Machine *m, u64 pa);
int jit_page_is_code(Machine *m, u64 pa) {
    Jit *j = m->jit;
    u32 pg = (u32)(pa >> 12);
    return pg < j->npages && (j->code_bits[pg >> 3] & (1u << (pg & 7)));
}

void jit_flush_lookup(Machine *m);
void jit_flush_lookup(Machine *m) {
    /* Guest mappings changed. Blocks are keyed by (physical address, RIP) and chain
     * only within a page, so they stay valid; only the RIP-keyed jump cache goes. */
    Jit *j = m->jit;
    j->gen++;
}

/* translations depend on CR0/CR4 SSE enables: drop everything when they change */
void jit_flush_code(Machine *m);
void jit_flush_code(Machine *m) { flush_all(m); }


/* ================= stack region =================
 * Plan which RSP-based accesses of a block can use one host pointer checked at
 * block entry: RSP moves by known amounts (push/pop, add/sub rsp, imm) until an
 * instruction that may write RSP or change address translation ends the region. */
static int sr_safe(const Ins *in) {
    int op = in->op;
    if (op < 0x40) return (op & 7) <= 5;
    if (op == 0x63 || op == 0x69 || op == 0x6b || (op >= 0x70 && op <= 0x7f)) return 1;
    if ((op >= 0x80 && op <= 0x85) || (op >= 0x88 && op <= 0x8b) || op == 0x8d) return 1;
    if (op == 0x90 && !in->b) return 1;
    if (op == 0x98 || op == 0x99 || op == 0xa8 || op == 0xa9 || (op >= 0xb0 && op <= 0xbf)) return 1;
    if (op == 0xc0 || op == 0xc1 || op == 0xc6 || op == 0xc7 || (op >= 0xd0 && op <= 0xd3)) return 1;
    if (op == 0xeb || op == 0xe9) return 1;
    if (op == 0xf6 || op == 0xf7) return 1;
    if ((op == 0xfe || op == 0xff) && (in->reg & 7) < 2) return 1;
    if ((op >= 0x140 && op <= 0x14f) || (op >= 0x180 && op <= 0x19f) || op == 0x1af) return 1;
    if (op == 0x1a3 || op == 0x1b6 || op == 0x1b7 || op == 0x1be || op == 0x1bf || op == 0x1bc || op == 0x1bd || op == 0x11f) return 1;
    if (op == 0x110 || op == 0x111 || op == 0x128 || op == 0x129 || op == 0x16f || op == 0x17f || op == 0x1d6 || op == 0x157 || op == 0x1ef) return 1;
    return 0;
}
/* may write RSP (conservative) */
static int sr_writes_rsp(const Ins *in) {
    int op = in->op;
    if (op >= 0xb0 && op <= 0xbf) return op >= 0xb8 && ((op & 7) | (in->b << 3)) == 4;
    if (op >= 0x110 && op <= 0x17f) return 0;
    if (op == 0x1d6 || op == 0x1ef || op == 0x11f) return 0;
    if (!in->has_modrm) return 0;
    int grp = op == 0x80 || op == 0x81 || op == 0x83 || op == 0xc0 || op == 0xc1 || op == 0xc6 || op == 0xc7 ||
              (op >= 0xd0 && op <= 0xd3) || op == 0xf6 || op == 0xf7 || op == 0xfe || op == 0xff || (op >= 0x190 && op <= 0x19f);
    int cmp = op == 0x38 || op == 0x39 || op == 0x3a || op == 0x3b || op == 0x84 || op == 0x85 || op == 0x1a3;
    if (cmp) return 0;
    if (in->mod == 3 && in->rm == 4) {
        if ((op == 0x80 || op == 0x81 || op == 0x83) && (in->reg & 7) == 7) return 0;
        if ((op == 0xf6 || op == 0xf7) && (in->reg & 7) < 2) return 0;
        if (!grp && (op < 0x40 ? (op & 2) : (op == 0x8b || op == 0x8a || op == 0x63 || op == 0x8d || op >= 0x100 || op == 0x69 || op == 0x6b))) return 0;   /* rm is the source */
        return 1;
    }
    if (grp) return 0;
    if (in->reg != 4) return 0;
    if (op < 0x40) return (op & 2) != 0;               /* reg is the destination */
    if (op == 0x88 || op == 0x89) return 0;
    return 1;
}
/* guest registers an instruction may write (conservative; ~0 = unknown) */
static u32 regs_written(const Ins *in) {
    int op = in->op;
    u32 m = 0;
    if (!sr_safe(in)) return ~0u;
    if ((op >= 0xb0 && op <= 0xbf) || (op >= 0x50 && op <= 0x5f)) return 1u << ((op & 7) | (in->b << 3)) | 1u << 4;
    if (op == 0x98 || op == 0x99 || ((op == 0xf6 || op == 0xf7) && (in->reg & 7) >= 4)) return 1u << 0 | 1u << 2;
    if (op < 0x40 && (op & 7) >= 4) return (op >> 3) == 7 ? 0 : 1u << 0;     /* op AL/eAX, imm */
    if (!in->has_modrm) return 0;
    int cmp = op == 0x38 || op == 0x39 || op == 0x3a || op == 0x3b || op == 0x84 || op == 0x85 || op == 0x1a3 ||
              ((op >= 0x80 && op <= 0x83) && (in->reg & 7) == 7) || ((op == 0xf6 || op == 0xf7) && (in->reg & 7) < 2);
    if (in->mod == 3 && !cmp && in->rm < 16) m |= 1u << in->rm;
    int regdst = (op < 0x40 && (op & 2) && !cmp) || op == 0x8a || op == 0x8b || op == 0x8d || op == 0x63 || op == 0x69 || op == 0x6b ||
                 (op >= 0x100 && op < 0x200 && op != 0x1a3 && !(op >= 0x110 && op <= 0x17f && !(op >= 0x140 && op <= 0x14f)) && !(op >= 0x180 && op <= 0x19f));
    if (regdst && in->reg < 16) m |= 1u << in->reg;
    if (op >= 0x190 && op <= 0x19f && in->mod == 3 && in->rm < 16) m |= 1u << in->rm;
    return m;
}
/* base reuse candidates: handlers that only use EDI after emit_tlb (t_mov, t_alu, t_movx) */
static int br_user(const Ins *in, int *write, int *sz) {
    int op = in->op;
    if (in->mod == 3 || in->a67 || in->rep || in->base < 0 || in->base == 4 || in->index >= 0 || in->riprel || in->seg == FS || in->seg == GS) return 0;
    if (op == 0x89 || op == 0x8b) { if (in->osz < 4 || in->lock) return 0; *write = op == 0x89; *sz = in->osz; return 1; }
    if (op == 0xc7) { if (in->osz < 4 || (in->reg & 7) || in->lock) return 0; *write = 1; *sz = in->osz; return 1; }
    if (op == 0x1b6 || op == 0x1b7 || op == 0x1be || op == 0x1bf) { if (in->osz < 4 || in->lock) return 0; *write = 0; *sz = (op & 1) ? 2 : 1; return 1; }
    if (op == 0x63) { if (in->osz != 8 || in->lock) return 0; *write = 0; *sz = 4; return 1; }
    int alu = (op < 0x40 && (op & 7) <= 3) || op == 0x80 || op == 0x81 || op == 0x83 || op == 0x84 || op == 0x85;
    if (!alu) return 0;
    int ext = op < 0x40 ? op >> 3 : (op == 0x84 || op == 0x85) ? 7 : in->reg & 7;
    int dst = op < 0x40 ? !(op & 2) : 1;
    *write = dst && ext != 7 && op != 0x84 && op != 0x85;
    if (in->lock && !*write) return 0;
    *sz = (op < 0x40 ? !(op & 1) : op == 0x80 || op == 0x84) ? 1 : in->osz;
    return 1;
}

static int sr_user(const Ins *in) {   /* the handler's access can use the region (t_mov / t_alu) */
    int op = in->op;
    if (in->mod == 3 || in->a67 || in->rep || in->base != 4 || in->index >= 0 || in->riprel || in->seg == FS || in->seg == GS) return 0;
    if (op == 0x89 || op == 0x8b) return in->osz >= 4;
    if (op == 0xc7) return in->osz >= 4 && (in->reg & 7) == 0;
    if (op < 0x40) return (op & 7) <= 3;
    return op == 0x80 || op == 0x81 || op == 0x83 || op == 0x84 || op == 0x85;
}
static u8 sr_use[MAX_BLOCK_INSNS];
static i32 sr_d[MAX_BLOCK_INSNS];
/* returns the number of region accesses; lo, hi = byte range relative to RSP_entry, w = any write */
static int sr_plan(const Ins *ins, u32 cnt, i32 *lo, i32 *hi, int *w) {
    i32 d = 0, mn = 0x7fffffff, mx = -0x7fffffff;
    int n = 0, wr = 0;
    for (u32 k = 0; k < cnt; k++) { sr_use[k] = 0; sr_d[k] = 0; }
    for (u32 k = 0; k < cnt; k++) {
        const Ins *in = &ins[k];
        sr_d[k] = d;
        int op = in->op;
        if (in->lock) break;
        if (op >= 0x50 && op <= 0x5f && !in->p66) {
            int r = (op & 7) | (in->b << 3);
            if (op <= 0x57) { if (d - 8 < mn) mn = d - 8; if (d - 1 > mx) mx = d - 1; wr = 1; d -= 8; }
            else { if (r == 4) break; if (d < mn) mn = d; if (d + 7 > mx) mx = d + 7; d += 8; }
            sr_use[k] = 1; n++;
            continue;
        }
        if ((op == 0xe8 || (op == 0xff && (in->reg & 7) == 2 && (in->mod == 3 || in->base != 4) && !in->rep)) && !in->p66) {
            if (d - 8 < mn) mn = d - 8;
            if (d - 1 > mx) mx = d - 1;
            wr = 1;
            sr_use[k] = 1; n++;
            break;
        }
        if (op == 0xc3 && !in->rep && !in->p66) {
            if (d < mn) mn = d;
            if (d + 7 > mx) mx = d + 7;
            sr_use[k] = 1; n++;
            break;
        }
        if ((op == 0x83 || op == 0x81) && in->mod == 3 && in->rm == 4 && in->osz == 8 && ((in->reg & 7) == 0 || (in->reg & 7) == 5)) {
            i64 v = in->imm;
            if (v > 0x10000 || v < -0x10000) break;
            d += (in->reg & 7) == 0 ? (i32)v : -(i32)v;
            if (d > 0x10000 || d < -0x10000) break;
            continue;
        }
        if (!sr_safe(in)) break;
        if (sr_user(in)) {
            i32 a = d + (i32)in->disp;
            if (in->disp > 0x10000 || in->disp < -0x10000) break;
            if (a < mn) mn = a;
            if (a + 7 > mx) mx = a + 7;
            int writes = op == 0x89 || op == 0xc7 || (op < 0x40 && !(op & 2) && (op >> 3) != 7) ||
                         ((op == 0x80 || op == 0x81 || op == 0x83) && (in->reg & 7) != 7);
            if (writes) wr = 1;
            sr_use[k] = 1; n++;
        }
        if (sr_writes_rsp(in)) break;
    }
    *lo = mn; *hi = mx; *w = wr;
    return n;
}

static JitBlock *translate(Cpu *c, u64 rip, u64 pa, const u8 *page) {
    Machine *m = c->m;
    Jit *j = m->jit;
    if (j->nblocks >= j->maxblocks || (u32)(j->code_end - j->code_ptr) < CODE_RESERVE || (u32)(j->cold_end - j->cold_ptr) < 65536) flush_all(m);
    if (tr_hot && (u32)(j->pcode_end - j->pcode_ptr) < CODE_RESERVE) return 0;   /* promoted arena full: keep the old copy */
    JitBlock *b = &j->blocks[j->nblocks++];
    j->translations++;
    b->pa = pa;
    b->rip = rip;
    b->host_page = m->ram + (uintptr_t)(pa & ~0xfffull);
    b->m32 = !c->mode64;
    b->valid = 1;
    b->ninsns = 0;
    T t;
    t.j = j;
    t.b = b;
    int split = !(jit_off & 0x80000);   /* out-of-line arena for cold code */
    t.e.p = split ? j->cold_ptr : j->code_ptr;   /* the C entry prologue goes out of line */
    t.e.end = split ? j->cold_end : j->code_end;
    t.e.over = 0;
    t.pending = 0;
    t.nnfix = 0;
    t.fuse_sz = t.fuse_now = 0;
    t.lf_known = 0;
    t.lf_prev = 0;
    t.m32 = !c->mode64;
    t.nolink = 0;
    t.fw_reg = t.fws_reg = -1;
    dec_m32 = t.m32;
    t.nstubs = 0;
    t.ncold = 0;
    t.nside = 0;
    t.page = rip & ~0xfffull;
    t.sr_cur = 0; t.sr_delta = 0; t.sr_pending = 0; t.sr_off = 0;
    t.defer_k = 0; t.can_defer = 0;
    for (int r = 0; r < 16; r++) t.br_ok[r] = 0;
    t.br_mode = 0;
    b->slow = 0;
    u8 *start = t.e.p;
    cmap_mark(j, start, PKEYS);
    push_r(&t.e, EBP); push_r(&t.e, EBX); push_r(&t.e, ESI); push_r(&t.e, EDI);
    e8(&t.e, 0x8b); e8(&t.e, 0x6c); e8(&t.e, 0x24); e8(&t.e, 20);     /* mov ebp, [esp+20] */
    e8(&t.e, 0x83); e8(&t.e, 0xed); e8(&t.e, (u8)-CPU_BIAS);            /* sub ebp, -128: biased */
    /* chain entry from C: remember the host stack, allow 128 KiB of call frames,
     * and push a sentinel frame that no RET matches */
    mov_m_r(&t.e, OFF(jit_esp), ESP);
    e8(&t.e, 0x8d); e8(&t.e, 0x84); e8(&t.e, 0x24); e32(&t.e, (u32)-0x20000);   /* lea eax, [esp - 128K] */
    mov_m_r(&t.e, OFF(jit_limit), EAX);
    push_i(&t.e, 0); push_i(&t.e, 0); push_i(&t.e, 0); push_i(&t.e, 0);
    mov_r_m(&t.e, ESI, OFF(chain_budget));
    u8 *to_entry = jmp32(&t.e);
    E cold_e = t.e;
    if (split) { t.e.p = tr_hot ? j->pcode_ptr : j->code_ptr; t.e.end = tr_hot ? j->pcode_end : j->code_end; t.e.over = 0; }
    else { cold_e.p = j->cold_ptr; cold_e.end = j->cold_end; }
    /* chain entry: claim N instructions of budget, leave if there were fewer.
     * Invalidation overwrites the sub with a jmp to the inval stub. */
    u8 *entry = t.e.p;
    patch(&cold_e, to_entry, entry);
    e8(&t.e, 0x83); e8(&t.e, 0xee); u8 *n8 = t.e.p; e8(&t.e, 0);                 /* sub esi, N (N <= 64; patched below) */
    u8 *out1 = jcc32(&t.e, 2);                                           /* jb out */
    u8 *promo = 0;
    b->heat = 1000; b->hot = (u32)tr_hot;
    if (!tr_hot && !tr_nosr && split && !(jit_off2 & 64)) {
        e8(&t.e, 0xff); e8(&t.e, 0x0d); e32(&t.e, (u32)(uintptr_t)&b->heat);   /* dec dword [heat] */
        promo = jcc32(&t.e, CC_E);                                       /* hot now: retranslate densely */
    }
    prof_inc_e(&t.e, &prof_blk_exec);
    if (jit_prof_on && bexec) { bexec[b - j->blocks] = 0; prof_inc_e(&t.e, &bexec[b - j->blocks]); }
    u32 n = 0, native = 0;
    /* pre-decode the block, then mark instructions whose flags are dead */
    static Ins ins[MAX_BLOCK_INSNS];
    static u8 dead[MAX_BLOCK_INSNS], jinl[MAX_BLOCK_INSNS], dead_after[MAX_BLOCK_INSNS];
    static u64 irip[MAX_BLOCK_INSNS];
    u32 cnt = 0;
    for (u64 a = rip; cnt < MAX_BLOCK_INSNS;) {
        u32 off = (u32)(a & 0xfff);
        if (!decode(page + off, 0x1000 - off, &ins[cnt])) break;
        irip[cnt] = a;
        jinl[cnt] = 0;
        a += (u64)ins[cnt].len;
        cnt++;
        {
            const Ins *l = &ins[cnt - 1];
            int jcc = ((l->op >= 0x70 && l->op <= 0x7f) || (l->op >= 0x180 && l->op <= 0x18f)) && !l->p66 && !(jit_off & 0x20000000);
            if ((l->op == 0xeb || l->op == 0xe9) && !l->p66 && !(jit_off & 0x40000000)) {
                /* forward jump within the page: keep translating at the target */
                u64 tg = a + (u64)l->imm;
                if (dec_m32) tg = (u32)tg;
                if ((tg & ~0xfffull) == (rip & ~0xfffull) && tg > a - (u64)l->len && (tg & 0xfff)) {
                    jinl[cnt - 1] = 1;
                    a = tg;
                    if (cnt >= 2 && ins[cnt - 2].op == 0xfb) break;
                    continue;
                }
            }
            if ((ends_block(l) && !jcc) || (a & 0xfff) == 0) break;
        }
        /* STI: the next instruction runs in its interrupt shadow, then the block
         * ends so the dispatcher checks for interrupts exactly where the
         * interpreter would */
        if (cnt >= 2 && ins[cnt - 2].op == 0xfb) break;
    }
    {
        int ow = 0;
        for (int k = (int)cnt - 1; k >= 0; k--) {
            int cl = fclass(&ins[k]);
            dead_after[k] = (u8)ow;                          /* flags overwritten before read after k */
            dead[k] = 0;
            if (cl == K_WRITE_REG) { dead[k] = (u8)ow; ow = 1; }
            else if (cl == K_WRITE_MEM) { dead[k] = (u8)ow; ow = 0; }
            else if (cl == K_PASS) dead[k] = (u8)ow;               /* ow unchanged */
            else if (cl == K_PASS_MEM || cl == K_READ_WRITE) { dead[k] = (u8)ow; ow = 0; }
            else if (cl == K_BARRIER) ow = 0;
        }
    }
    if (jit_prof_on && bmem) {
        /* base-register reuse estimate: memory operands [base + disp] whose base was used earlier unmodified */
        u32 nm = 0, nr = 0, nr2 = 0;
        int seen[16]; i32 dmin[16], dmax[16];
        for (int r = 0; r < 16; r++) seen[r] = 0;
        for (u32 k = 0; k < cnt; k++) {
            const Ins *in = &ins[k];
            int op = in->op;
            if (in->has_modrm && in->mod != 3 && !in->riprel && in->base >= 0 && in->base != 4 && op != 0x8d && !in->a67) {
                int b = in->base;
                nm++;
                if (seen[b]) { nr++; if (in->disp - dmin[b] < 2048 && dmax[b] - in->disp < 2048) nr2++; if (in->disp < dmin[b]) dmin[b] = in->disp; if (in->disp > dmax[b]) dmax[b] = in->disp; }
                else { seen[b] = 1; dmin[b] = dmax[b] = in->disp; }
            }
            /* registers this instruction may write: forget them */
            int w = -1, all = 0;
            if (in->has_modrm) {
                if (in->mod == 3) w = in->rm;
                if (op < 0x40 && (op & 2)) w = in->reg; else if (op == 0x8b || op == 0x8a || op == 0x8d || op == 0x63 || op == 0x69 || op == 0x6b || (op >= 0x100 && op < 0x200)) { if (in->mod == 3) seen[in->rm] = seen[in->rm]; w = in->reg; }
                if (op == 0x86 || op == 0x87 || op == 0x1b0 || op == 0x1b1 || op == 0x1c0 || op == 0x1c1 || op == 0x1c7 || ((op == 0xf6 || op == 0xf7) && (in->reg & 7) >= 4)) all = 1;
            } else {
                if ((op >= 0x50 && op <= 0x5f) || (op >= 0x90 && op <= 0x97) || (op >= 0xb0 && op <= 0xbf) || (op >= 0x1c8 && op <= 0x1cf)) w = (op & 7) | (in->b << 3);
                if ((op >= 0xa4 && op <= 0xaf) || op == 0x98 || op == 0x99 || op >= 0x100) all = 1;
            }
            if (all) for (int r = 0; r < 16; r++) seen[r] = 0;
            if (w >= 0 && w < 16) seen[w] = 0;
            if (in->has_modrm && in->mod == 3 && in->rm < 16 && !(op == 0x39 || op == 0x3b || op == 0x85 || op == 0x84)) seen[in->rm] = 0;
        }
        u32 bi = (u32)(b - j->blocks);
        bmem[bi] = (u8)(nm > 255 ? 255 : nm); breuse[bi] = (u8)(nr > 255 ? 255 : nr); breuse2[bi] = (u8)(nr2 > 255 ? 255 : nr2);
    }
    if (jit_dump_rip && jit_dump_rip >= rip && cnt) {
        u64 endr = rip; for (u32 k = 0; k < cnt; k++) endr += (u64)ins[k].len;
        if (jit_dump_rip < endr) {
            u64 a2 = rip;
            emu_log(m, "JITDUMP block %llx (%u insns, dead flags marked *)", (unsigned long long)rip, cnt);
            for (u32 k = 0; k < cnt; k++) {
                char hx[64]; u32 o = (u32)(a2 & 0xfff), L = (u32)ins[k].len, q = 0;
                for (u32 i2 = 0; i2 < L && q + 3 < sizeof hx; i2++) { const char *H = "0123456789abcdef"; hx[q++] = H[page[o + i2] >> 4]; hx[q++] = H[page[o + i2] & 15]; hx[q++] = ' '; }
                hx[q] = 0;
                emu_log(m, "  %llx%s %s", (unsigned long long)a2, dead[k] ? "*" : " ", hx);
                a2 += L;
            }
        }
    }
    u8 *srf1 = 0, *srf2 = 0;
    int sr_n = 0;
    {
        i32 lo, hi; int w;
        sr_n = (!t.m32 && !tr_nosr && !(jit_off & 0x1000000) && cnt) ? sr_plan(ins, cnt, &lo, &hi, &w) : 0;
        if (sr_n >= 2 && hi - lo < 1024) {
            E *e = &t.e;
            mov_r_m(e, EAX, R_LO(4)); mov_r_m(e, EDX, R_HI(4));
            if (lo) { alu_r_i(e, 0, EAX, (u32)lo); alu_r_i(e, 2, EDX, lo < 0 ? 0xffffffffu : 0); }
            emit_tlb_inline(e, w, (int)(hi - lo + 1), &srf1, &srf2);
            if (lo) { e8(e, 0x8d); e8(e, 0xbf); e32(e, (u32)-lo); }     /* lea edi, [edi - lo] */
            mov_m_r(e, OFF(jit_sr), EDI);
            prof_inc_e(e, &ph_sr_chk);
        } else sr_n = 0;
    }
    u64 cur = rip;
    if (!cnt && !t.m32 && !(jit_off & 0x800)) {
        /* the instruction crosses into the next page: translate it from both pages,
         * guarded at run time by the exec TLB mapping and the bytes on the next page */
        t.rip = rip; t.pending = 0; t.pkey = 0;
        u8 *fail[8]; int nf = 0, done = 0;
        u32 n1 = 0x1000 - (u32)(rip & 0xfff);
        u64 la2 = (rip & ~0xfffull) + 0x1000;
        u8 *h2 = (!(jit_off2 & 4) && n1 < 15) ? cpu_exec_probe(c, la2) : 0;
        static Ins xin;
        if (h2) {
            u8 buf[16];
            for (u32 i = 0; i < 16; i++) buf[i] = i < n1 ? page[(rip & 0xfff) + i] : h2[i - n1];
            if (decode(buf, 16, &xin) && (u32)xin.len > n1) {
                E *e = &t.e;
                u32 n2 = (u32)xin.len - n1;
                u32 tx = OFF(tlb_x) + (((u32)(la2 >> 12) & (TLB_SIZE - 1)) << 4);
                mov_r_i(e, EAX, (u32)la2 & 0xfffff000u);
                alu_r_m(e, 0x0b, EAX, OFF(tlb_mode));
                alu_r_m(e, 0x3b, EAX, tx); fail[nf++] = jcc32(e, CC_NE);
                e8(e, 0x81); e8(e, 0xbd); e32(e, tx + 4); e32(e, (u32)(la2 >> 32)); fail[nf++] = jcc32(e, CC_NE);
                e8(e, 0x81); e8(e, 0xbd); e32(e, tx + 8); e32(e, (u32)(uintptr_t)h2); fail[nf++] = jcc32(e, CC_NE);
                for (u32 i = 0; i < n2; i += 4) {
                    u32 v = (u32)h2[i] | (u32)h2[i + 1] << 8 | (u32)h2[i + 2] << 16 | (u32)h2[i + 3] << 24;
                    e8(e, 0x81); e8(e, 0x3d); e32(e, (u32)(uintptr_t)(h2 + i)); e32(e, v);   /* cmp dword [abs], bytes */
                    fail[nf++] = jcc32(e, CC_NE);
                }
                t.next = rip + (u64)xin.len;
                t.fuse_now = t.fuse_sz = 0; t.flags_dead = 0; t.lf_prev = t.lf_known = 0;
                t.fw_reg = t.fws_reg = -1; t.fw_pos = e->p;
                t.sr_cur = 0; t.sr_pending = 0; t.br_mode = 0; t.can_defer = 0;
                {
                    int sse = (xin.op >= 0x110 && xin.op <= 0x17f) || (xin.op >= 0x1c2 && xin.op <= 0x1ff && xin.op != 0x1c7 && (xin.op < 0x1c8 || xin.op > 0x1cf)) || xin.op >= 0x200;
                    u32 sub = sse ? (u32)((xin.rep == 0xf3 ? 3 : xin.rep == 0xf2 ? 2 : xin.p66 ? 1 : 0) | (xin.mod == 3 ? 4 : 0)) : (u32)(xin.has_modrm ? (xin.reg & 7) : 0);
                    t.pkey = (u32)((xin.op << 3) | sub);
                }
                int r = xin.op == 0xfb ? 0 : t_native(&t, c, &xin);
                if (r == 1) { exit_const(&t, t.next); done = 1; }
                else if (r == 2) done = 1;
                else { for (int k = 0; k < nf; k++) patch(e, fail[k], e->p); nf = 0; t.pkey = 0; }
                prof_inc_e(e, &px_xpage);
            }
        }
        for (int k = 0; k < nf; k++) patch(&t.e, fail[k], t.e.p);
        if (!done || nf) { t.pkey = 0; t.pending = 0; t.nolink = 0; fallback(&t, 1); }   /* (handlers may have bumped pending) */
        n = 1; native = 1;
    }
    for (;;) {
        if (n >= cnt) {
            if (n && cnt) exit_const(&t, cur);
            break;
        }
        Ins in = ins[n];
        t.pending = n;
        t.rip = irip[n];
        t.next = irip[n] + (u64)in.len;
        {
            int sse = (in.op >= 0x110 && in.op <= 0x17f) || (in.op >= 0x1c2 && in.op <= 0x1ff && in.op != 0x1c7 && (in.op < 0x1c8 || in.op > 0x1cf)) || in.op >= 0x200;
            u32 sub = sse ? (u32)((in.rep == 0xf3 ? 3 : in.rep == 0xf2 ? 2 : in.p66 ? 1 : 0) | (in.mod == 3 ? 4 : 0)) : (u32)(in.has_modrm ? (in.reg & 7) : 0);
            t.pkey = (u32)((in.op << 3) | sub);   /* SSE: prefix (0 none, 1 66, 2 F2, 3 F3) + 4 if register source */
        }
        if (t.defer_k && !((in.op >= 0x70 && in.op <= 0x7f) || (in.op >= 0x180 && in.op <= 0x18f))) {
            emit_deferred_lf(&t, t.defer_k, t.defer_sz);   /* not expected: keep the state exact anyway */
            t.defer_k = 0;
        }
        t.can_defer = 0;
        if (n + 1 < cnt && !t.nolink && !(jit_off & 0x20000000) && t.nside < 63 && dead_after[n + 1] && !dead[n]) {
            const Ins *jn = &ins[n + 1];
            if (((jn->op >= 0x70 && jn->op <= 0x7f) || (jn->op >= 0x180 && jn->op <= 0x18f)) && !jn->p66 && !jinl[n + 1])
                t.can_defer = 16 | (jn->op & 15);
        }
        t.br_mode = 0; t.br_emitted = 0;
        {
            int bw, bsz;
            if (!t.m32 && !(jit_off2 & 2) && br_user(&in, &bw, &bsz) ) {
                int B = in.base;
                t.br_reg = B;
                i32 dd = in.disp - t.br_disp[B];
                if (t.br_ok[B] && (!bw || t.br_w[B]) && dd >= -0x100000 && dd <= 0x100000) { t.br_mode = 2; t.br_d = dd; }
                else t.br_mode = 1;
            }
        }
        t.sr_cur = sr_n && sr_use[n];
        t.sr_delta = sr_d[n];
        t.sr_pending = 0;
        t.fuse_now = t.fuse_sz;
        t.fuse_sz = 0;
        t.flags_dead = dead[n];
        int prev_known = t.lf_known;
        t.lf_prev = prev_known;
        t.lf_known = 0;
        if (n > 0 && ins[n - 1].op == 0xfb) t.nolink = 1;              /* in the STI shadow */
        t.fw_reg = t.fws_reg; t.fw_lo = t.fws_lo; t.fw_hi = t.fws_hi;
        t.fws_reg = -1;
        t.fw_pos = t.e.p;
        cmap_mark(j, t.e.p, t.pkey);
        if (jit_prof_on && pexec) prof_inc_e(&t.e, &pexec[t.pkey < PKEYS ? t.pkey : 0]);   /* executed count by kind (flags clobbered: profiling only) */
        u8 *ins_start = t.e.p;
        int r = jinl[n] ? 1 : (in.op == 0xfb && n + 1 >= cnt) ? 0 : t_native(&t, c, &in);   /* last: interpret (sets the shadow) */
        if (r == 1 && !t.lf_known && (fclass(&in) == K_NEUTRAL || (!(jit_off & 32768) && keeps_flags(&in)))) t.lf_known = prev_known;   /* flags untouched */
        if (t.br_mode == 1 && r == 1 && t.br_emitted) {
            int bw, bsz; br_user(&in, &bw, &bsz);
            t.br_ok[in.base] = 1; t.br_w[in.base] = bw; t.br_disp[in.base] = in.disp;
        }
        t.br_mode = 0;
        if (!r) for (int q = 0; q < 16; q++) t.br_ok[q] = 0;
        else { u32 wm = regs_written(&in); for (int q = 0; q < 16; q++) if (wm >> q & 1) t.br_ok[q] = 0; }
        if (jit_prof_on) { pb_bytes[t.pkey] += (u64)(t.e.p - ins_start); pb_cnt[t.pkey]++; }
        if (jit_prof_on) {
            static int dumped[PKEYS];
            static const u32 want[] = { 0x89u << 3, 0x89u << 3 | 1, 0x8bu << 3, 0x3bu << 3 };
            for (u32 w = 0; w < 4; w++) if (t.pkey == want[w] && !t.m32 && in.w && dumped[t.pkey]++ == 3) {

                char fn[32]; u32 len = (u32)(t.e.p - ins_start);
                emu_log(m, "DUMPKEY %x len %u at %x", t.pkey, len, (u32)(uintptr_t)ins_start);
                extern void jit_dump_bytes(const u8 *, u32, u32);
                jit_dump_bytes(ins_start, len, t.pkey); (void)fn;
            }
        }
        n++;
        if (r) {
            native++;
            if (r == 2) break;
        } else {
            if (jit_prof_on) {
                static int lg[PKEYS];
                if (t.pkey < PKEYS && lg[t.pkey]++ < 3 && (t.pkey == 0 || t.pkey == (0xd2u << 3 | 4) || t.pkey == (0xc7u << 3) || t.pkey == (0x147u << 3 | 5) || t.pkey == (0xf6u << 3 | 3) || t.pkey == (0xffu << 3 | 6))) {
                    u32 o = (u32)(t.rip & 0xfff);
                    emu_log(m, "FB %x rip %llx: %02x %02x %02x %02x %02x %02x %02x %02x lock %d a67 %d rep %d p66 %d osz %d mod %d", t.pkey, (unsigned long long)t.rip,
                        page[o], page[(o+1)&0xfff], page[(o+2)&0xfff], page[(o+3)&0xfff], page[(o+4)&0xfff], page[(o+5)&0xfff], page[(o+6)&0xfff], page[(o+7)&0xfff], in.lock, in.a67, in.rep, in.p66, in.osz, in.mod);
                }
            }
            int end = ends_block(&in) || in.op == 0xfb || (in.op == 0x90 && in.rep == 0xf3 && m->ncpu > 1);   /* STI shadow; PAUSE yield: back to C */
            fallback(&t, end);
            if (end && !t.m32 && !(jit_off & 0x800)) native = 1;   /* has code worth keeping: it dispatches on */
            if (end) break;
        }
        cur = jinl[n - 1] ? (t.m32 ? (u32)(t.next + (u64)in.imm) : t.next + (u64)in.imm) : t.next;
        if (t.defer_k && (n >= MAX_BLOCK_INSNS || (cur & 0xfff) == 0 || (u32)(t.e.end - t.e.p) < 1024)) { emit_deferred_lf(&t, t.defer_k, t.defer_sz); t.defer_k = 0; }
        if (n >= MAX_BLOCK_INSNS || (cur & 0xfff) == 0) { exit_const(&t, cur); break; }
        if ((u32)(t.e.end - t.e.p) < 1024) { exit_const(&t, cur); break; }
    }
    E hot_e = t.e;
    if (split) t.e = cold_e;         /* tails go out of line */
    emit_colds(&t);
    if (srf1) {
        /* stack-region check failed: give the budget back, then run the conservative
         * variant (the jmp is linked to it; unlinked it returns to C asking for it) */
        patch(&t.e, srf1, t.e.p); patch(&t.e, srf2, t.e.p);
        prof_inc_e(&t.e, &ph_sr_fail);
        emit_refund(&t, 0);
        u8 *site = jmp32(&t.e);
        patch(&t.e, site, t.e.p);
        e8(&t.e, 0xc7); ebp_disp(&t.e, 0, OFF(jit_sr_fail)); e32(&t.e, 1);   /* mov dword [jit_sr_fail], 1 */
        emit_exit(&t, rip, site);
    }
    if (promo) {
        patch(&t.e, promo, t.e.p);
        emit_refund(&t, 0);
        e8(&t.e, 0xc7); ebp_disp(&t.e, 0, OFF(jit_sr_fail)); e32(&t.e, 2);   /* ask C to promote this block */
        emit_exit(&t, rip, 0);
    }
    /* not enough budget: give it back and return to the dispatcher at this block */
    cmap_mark(j, t.e.p, PKEYS + 1);
    patch(&t.e, out1, t.e.p);
    emit_refund(&t, 0);
    prof_inc_e(&t.e, &px_budget);
    u8 *inval = t.e.p;               /* invalidated: entry jumps here (nothing claimed) */
    prof_inc_e(&t.e, &px_inval);
    emit_exit(&t, rip, 0);
    if (jit_prof_on) prof_tr_stubs += (u64)t.nstubs;
    u8 *tail0 = t.e.p;
    emit_stubs(&t);
    if (jit_prof_on) { pb_bytes[PKEYS + 1] += (u64)(t.e.p - tail0); pb_cnt[PKEYS + 1]++; }
    if (split) { cold_e = t.e; t.e = hot_e; }
    if (t.e.over || cold_e.over) { flush_all(m); return 0; }
    if (!native && cmap_n) { while (cmap_n && cmap_off[cmap_n - 1] >= (u32)(entry - j->code)) cmap_n--; }   /* code dropped */
    *n8 = (u8)n;
    if (jit_prof_on && bend) {
        const Ins *l = &ins[n ? n - 1 : 0];
        u8 why = 8;
        if ((l->op >= 0x70 && l->op <= 0x7f) || (l->op >= 0x180 && l->op <= 0x18f)) why = 0;
        else if (l->op == 0xeb || l->op == 0xe9) why = ((rip & ~0xfffull) == ((t.next + (u64)l->imm) & ~0xfffull)) ? 1 : 2;
        else if (l->op == 0xe8) why = 3;
        else if (l->op == 0xc3) why = 4;
        else if (l->op == 0xff) why = 5;
        else if ((t.next & 0xfff) == 0) why = 6;
        else if (n >= MAX_BLOCK_INSNS) why = 7;
        bend[b - j->blocks] = why;
        bninsn[b - j->blocks] = (u8)n;
    }
    for (int k = 0; k < t.nnfix; k++) {
        u32 v = n - t.nfix_k[k];
        u8 *f = t.nfix[k];
        f[0] = (u8)v; f[1] = (u8)(v >> 8); f[2] = (u8)(v >> 16); f[3] = (u8)(v >> 24);
    }
    if (jit_dump_rip && jit_dump_rip == rip && !tr_nosr) {
        extern void jit_dump_bytes(const u8 *, u32, u32);
        jit_dump_bytes(entry, (u32)(t.e.p - entry), 0xfff);
        emu_log(m, "JITDUMP host code %u hot bytes for %u insns, tail %u bytes", (u32)(t.e.p - entry), n, (u32)(cold_e.p - start));
    }
    b->entry = entry;
    b->inval = inval;
    b->in_links = 0;
    if (!native && jit_prof_on) {
        static int logged;
        if (logged < 12 && rip > 0xfffff80000000000ull) {
            logged++;
            u32 o = (u32)(rip & 0xfff);
            emu_log(m, "interp-only block %llx cnt %u first op %x bytes %02x %02x %02x %02x (page off %x)", (unsigned long long)rip, cnt,
                    cnt ? (u32)ins[0].op : 0u, page[o], o + 1 < 4096 ? page[o + 1] : 0, o + 2 < 4096 ? page[o + 2] : 0, o + 3 < 4096 ? page[o + 3] : 0, o);
        }
    }
    if (!native) {
        /* nothing translated natively: interpreting directly is cheaper */
        b->ninsns = 0;
        b->fn = 0;
        b->entry = 0;   /* its code is not kept: the inline dispatch must not jump there */
    } else {
        b->ninsns = n;
        b->fn = (void (*)(Cpu *))(void *)start;
        if (tr_hot) j->pcode_ptr = t.e.p; else j->code_ptr = t.e.p;
        j->cold_ptr = cold_e.p;
    }
    if (!tr_nosr) {
        u32 h = hash_pa(pa);
        b->hnext = j->hash[h];
        j->hash[h] = b;
    } else b->hnext = 0;
    u32 pg = (u32)(pa >> 12);
    b->pnext = j->page_head[pg];
    j->page_head[pg] = b;
    if (!(j->code_bits[pg >> 3] & (1u << (pg & 7)))) {
        j->code_bits[pg >> 3] |= (u8)(1u << (pg & 7));
        /* writes to this page must now take the slow path */
        u8 *host = m->ram + (uintptr_t)(pa & ~0xfffull);
        for (u32 i = 0; i < TLB_SIZE; i++) if (c->tlb_w[i].tag && c->tlb_w[i].host == host) c->tlb_w[i].tag |= TLB_CODE;
    }
    return b;
}

/* Block for c->rip, translating it if needed. NULL: interpret this step. */
static JitBlock *lookup(Cpu *c) {
    Machine *m = c->m;
    Jit *j = m->jit;
    u64 rip = c->rip;
    if ((rip & ~0xfffull) != c->fetch_la) fetch_page(c, rip);
    u8 *host = c->fetch_host;
    JcEntry *jc = &j->jc[(u32)(rip ^ (rip >> 12)) & (JC_SIZE - 1)];
    u32 m32 = !c->mode64;
    JitBlock *b = 0;
    if (jc->rip == rip && jc->b && jc->b->host_page == host && jc->b->valid && jc->b->m32 == m32) {
        if (!jc->b->cold_len) return jc->b;
        b = jc->b;                                     /* placeholder: count this execution below */
    }
    if (host < m->ram || host >= m->ram + (uintptr_t)m->ram_size) return 0;
    u64 pa = (u64)(uintptr_t)(host - m->ram) + (rip & 0xfff);
    if (!b) for (b = j->hash[hash_pa(pa)]; b; b = b->hnext) if (b->pa == pa && b->rip == rip && b->m32 == m32) break;
    if (b && b->cold_len) {
        if (++b->hits < jit_hot) { c->jit_skip = b->cold_len - 1; return b; }
        b = 0;                                         /* hot now: translate (the new block goes first in the chain) */
    } else if (!b && !(jit_off2 & 16) && j->nblocks < j->maxblocks) {
        /* first time here: a placeholder that only knows how far straight-line code goes */
        u32 n = 0;
        dec_m32 = (int)m32;
        for (u64 a = rip; n < MAX_BLOCK_INSNS;) {
            u32 off = (u32)(a & 0xfff);
            Ins in;
            if (!decode(host + off, 0x1000 - off, &in)) break;
            n++;
            a += (u64)in.len;
            if (ends_block(&in) || (in.op >= 0x70 && in.op <= 0x7f) || (in.op >= 0x180 && in.op <= 0x18f) || (a & 0xfff) == 0) break;
        }
        if (n) {
            b = &j->blocks[j->nblocks++];
            e_memset(b, 0, sizeof *b);
            b->pa = pa; b->rip = rip; b->host_page = host; b->m32 = m32; b->valid = 1;
            b->hits = 1; b->cold_len = n;
            u32 h = hash_pa(pa);
            b->hnext = j->hash[h]; j->hash[h] = b;
            c->jit_skip = n - 1;
            jc->rip = rip; jc->b = b;
            return b;
        }
    }
    if (!b) {
        u64 t0 = jit_prof_on ? rdtsc64() : 0;
        u8 *cp0 = j->code_ptr;
        b = translate(c, rip, pa, host);
        if (jit_prof_on) {
            prof_tr_cycles += rdtsc64() - t0;
            if (b && j->code_ptr > cp0) { prof_tr_bytes += (u64)(j->code_ptr - cp0); prof_tr_insns += b->ninsns; }
        }
    }
    if (!b) return 0;
    jc->rip = rip;
    jc->b = b;
    return b;
}

/* Called by the run loop: run one block if one fits the remaining budget.
 * Returns 1 if a block ran (icount advanced), 0 to interpret one step. */
int jit_run_block(Cpu *c, u64 budget);
int jit_run_block(Cpu *c, u64 budget) {
    Jit *j = c->m->jit;
    u8 *from = j->pending_from;
    j->pending_from = 0;
    JitBlock *b = lookup(c);
    if (c->jit_sr_fail == 2) {
        c->jit_sr_fail = 0;
        if (b && b->ninsns && b->rip == c->rip && !b->hot && b->m32 == (u32)!c->mode64) {
            u32 gen = j->gen;
            tr_hot = 1;
            JitBlock *nb = translate(c, b->rip, b->pa, c->fetch_host);
            tr_hot = 0;
            if (j->gen != gen) { from = 0; b = nb; }          /* flushed meanwhile */
            else if (nb && nb->ninsns) {
                block_retire(j, b);
                JcEntry *jc = &j->jc[(u32)(c->rip ^ (c->rip >> 12)) & (JC_SIZE - 1)];
                jc->rip = c->rip; jc->b = nb;
                b = nb;
            } else b->heat = 0x7fffffff;                       /* could not promote: stop asking */
        }
    }
    if (c->jit_sr_fail) {
        c->jit_sr_fail = 0;
        if (b && b->ninsns && b->rip == c->rip && b->m32 == (u32)!c->mode64) {
            if (!b->slow || !b->slow->valid) {
                u32 gen = j->gen;
                tr_nosr = 1;
                JitBlock *sb = translate(c, b->rip, b->pa, c->fetch_host);
                tr_nosr = 0;
                if (j->gen != gen) { from = 0; b = sb; }   /* the cache was flushed: b is gone */
                else b->slow = sb;
                if (!sb) return 0;
                if (j->gen == gen) b = sb;
            } else b = b->slow;
        }
    }
    if (!b || !b->ninsns || b->ninsns > budget) {
        if (jit_prof_on) {
            prof_out_why[!b ? 0 : !b->ninsns ? 1 : 2]++;
            if ((c->rip & ~0xfffull) == c->fetch_la) {
                u32 off = (u32)(c->rip & 0xfff);
                prof_out[raw_key(c->fetch_host + off, 0x1000 - off)]++;
            }
        }
        return 0;
    }
    if (from && j->pending_gen == j->gen && j->pending_rip == c->rip) {
        if (from[-1] != 0xe9) {
            /* cross-page link: from = host page immediate, its jmp rel32 11 bytes later */
            u32 h = (u32)(uintptr_t)b->host_page;
            from[0] = (u8)h; from[1] = (u8)(h >> 8); from[2] = (u8)(h >> 16); from[3] = (u8)(h >> 24);
            from += 11;
        }
        u8 *old_target = from + 4 + (i32)((u32)from[0] | (u32)from[1] << 8 | (u32)from[2] << 16 | (u32)from[3] << 24);
        u32 rel = (u32)(uintptr_t)b->entry - (u32)(uintptr_t)(from + 4);
        from[0] = (u8)rel; from[1] = (u8)(rel >> 8); from[2] = (u8)(rel >> 16); from[3] = (u8)(rel >> 24);
        if (j->nlinks < j->maxlinks) {   /* remembered so invalidating b can unlink it */
            j->links[j->nlinks].site = from;
            j->links[j->nlinks].stub = old_target;
            j->links[j->nlinks].next = b->in_links;
            b->in_links = ++j->nlinks;
        }
    }
    if (jit_prof_on) prof_c_entries++;
    c->chain_budget = budget > 0xffffffffull ? 0xffffffffu : (u32)budget;
    c->chain_from = 0;
    c->in_block = 1;
    c->blk_icount0 = c->icount;
    c->icount_base = c->icount + c->chain_budget;
    b->fn(c);
    c->icount = c->icount_base - c->chain_budget;
    c->in_block = 0;
    j->jit_insns += c->icount - c->blk_icount0;
    if (c->chain_from) {
        j->pending_from = (u8 *)(uintptr_t)c->chain_from;
        j->pending_rip = c->rip;
        j->pending_gen = j->gen;
    }
    return 1;
}

int jit_init(Machine *m, u8 *mem, u32 size) {
    u32 npages = (u32)(m->ram_size >> 12);
    u32 need = (u32)sizeof(Jit) + HASH_SIZE * 4 + npages * 4 + (npages + 7) / 8 + JC_SIZE * (u32)sizeof(JcEntry) + 64;
    u32 maxblocks = size / 512u;     /* ~1 block per 512 bytes of cache */
    need += maxblocks * (u32)sizeof(JitBlock) + 2 * maxblocks * (u32)sizeof(struct Link);
    if (size < need + (1u << 20)) return -1;
    u8 *p = (u8 *)(((uintptr_t)mem + 15) & ~(uintptr_t)15);
    Jit *j = (Jit *)p; p += (sizeof(Jit) + 15) & ~15u;
    e_memset(j, 0, sizeof *j);
    j->hash = (JitBlock **)p; p += HASH_SIZE * 4;
    j->page_head = (JitBlock **)p; p += npages * 4;
    j->code_bits = p; p += ((npages + 7) / 8 + 15) & ~15u;
    j->jc = (JcEntry *)p; p += JC_SIZE * sizeof(JcEntry);
    j->blocks = (JitBlock *)p; p += maxblocks * sizeof(JitBlock);
    j->links = (struct Link *)p; p += 2 * maxblocks * sizeof(struct Link);
    j->maxlinks = 2 * maxblocks;
    j->maxblocks = maxblocks;
    j->npages = npages;
    {
        E e0 = { (u8 *)(((uintptr_t)p + 15) & ~(uintptr_t)15), mem + size, 0 };
        j->interp_stub = e0.p;
        emit_interp_routine(&e0);
        j->exit_stub = e0.p;
        emit_exit_routine(&e0);
        for (int w = 0; w < 2; w++)
            for (int k = 0; k < 5; k++) { j->tlb_fn[w][k] = e0.p; emit_tlb_routine(&e0, w, 1 << k); }
        for (int m = 0; m < 2; m++) { j->disp_fn[m] = e0.p; emit_dispatch_body(&e0, j, m, 0); }
        for (int m = 0; m < 2; m++) { j->lookup_fn[m] = e0.p; emit_lookup_routine(&e0, j, m); }
        j->miss_fn = e0.p; emit_miss_routine(&e0);
        j->lfcf_fn = e0.p; emit_lfcf_routine(&e0);
        j->lfzf_fn = e0.p; emit_lfzf_routine(&e0);
        p = e0.p;
    }
    j->code = j->code_ptr = (u8 *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
    {
        /* hot block bodies below, their cold tails above (~40% of the bytes) */
        u32 avail = (u32)(mem + size - j->code);
        u8 *hot_end = j->code + (avail / 100u * 58u & ~15u);
        j->cold = j->cold_ptr = hot_end;
        j->cold_end = mem + size;
        u32 hsz = (u32)(hot_end - j->code);
        j->code_end = j->code + ((hsz - hsz / 4) & ~15u);   /* last quarter of the hot arena: promoted blocks */
        j->pcode = j->pcode_ptr = j->code_end;
        j->pcode_end = hot_end;
    }
    m->jit = j;
    j->m = m;
    if (jit_prof_on && !bexec) {
        bexec = (u32 *)jit_prof_alloc(maxblocks * 4); bend = (u8 *)jit_prof_alloc(maxblocks); bninsn = (u8 *)jit_prof_alloc(maxblocks);
        pexec = (u32 *)jit_prof_alloc(PKEYS * 4); if (pexec) e_memset(pexec, 0, PKEYS * 4);
        bmem = (u8 *)jit_prof_alloc(maxblocks); breuse = (u8 *)jit_prof_alloc(maxblocks); breuse2 = (u8 *)jit_prof_alloc(maxblocks);
        if (!bexec || !bend || !bninsn) bexec = 0, bend = 0, bninsn = 0;
    }
    if (jit_prof_on && !cmap_off) {
        cmap_off = (u32 *)jit_prof_alloc(CMAP_MAX * 4);
        cmap_key = (u16 *)jit_prof_alloc(CMAP_MAX * 2);
        if (!cmap_off || !cmap_key) cmap_off = 0;
    }
    flush_all(m);
    j->flushes = 0;
    m->cpu.jit_on = 1;
    for (int i = 1; i < m->ncpu; i++) m->cpus[i]->jit_on = 1;
    return 0;
}

/* host PC inside generated code -> its block (profiling); 0 if none */
int jit_pc_block(Machine *m, uintptr_t pc, u64 *rip, uintptr_t *start, u32 *ninsns);
int jit_pc_block(Machine *m, uintptr_t pc, u64 *rip, uintptr_t *start, u32 *ninsns) {
    Jit *j = m->jit;
    if (!j || pc < (uintptr_t)j->code || pc >= (uintptr_t)j->code_ptr) return 0;
    u32 lo = 0, hi = j->nblocks;      /* blocks with code are laid out in order */
    JitBlock *best = 0;
    while (lo < hi) {
        u32 mid = (lo + hi) / 2;
        JitBlock *b = &j->blocks[mid];
        uintptr_t f = (uintptr_t)(void *)b->entry;
        if (!f) {   /* interpreted-only block: find a neighbour with code */
            u32 k = mid; while (k > lo && !j->blocks[k].entry) k--;
            if (!j->blocks[k].entry) { lo = mid + 1; continue; }
            b = &j->blocks[k]; f = (uintptr_t)(void *)b->entry;
            if (f <= pc) { best = b; lo = mid + 1; } else hi = k;
            continue;
        }
        if (f <= pc) { best = b; lo = mid + 1; } else hi = mid;
    }
    if (!best) return 0;
    *rip = best->rip; *start = (uintptr_t)(void *)best->entry; *ninsns = best->ninsns;
    return 1;
}

void jit_stats(Machine *m, char *out, u32 cap);
void jit_stats(Machine *m, char *out, u32 cap) {
    Jit *j = m->jit;
    const char *fmt = "JIT: %llu translations, %u live blocks, %u KiB code, %u flushes, %u page invalidations";
    unsigned long long tr = j->translations;
    u32 kib = (u32)(j->code_ptr - j->code) >> 10;
    /* local snprintf */
    extern int emu_vsnprintf(char *, u32, const char *, __builtin_va_list);
    (void)fmt; (void)tr; (void)kib;
    emu_log(m, "JIT: %llu translations, %u live blocks, %u KiB code, %u flushes, %u page invalidations, %llu%% of insns in blocks",
            tr, j->nblocks, kib, j->flushes, j->invalidations,
            m->cpu.icount ? (unsigned long long)(j->jit_insns * 100 / m->cpu.icount) : 0ull);
    if (jit_prof_on) emu_log(m, "JIT: %llu insns outside 64-bit mode (%llu in 32-bit code)", (unsigned long long)m->cpu.icount_legacy, (unsigned long long)m->cpu.icount_legacy32);
    if (jit_prof_on) {
        for (int r = 0; r < 20; r++) {
            u32 best = 0;
            for (u32 i = 0; i < 4096; i++) if (legacy_hist_n[i] > legacy_hist_n[best]) best = i;
            if (!legacy_hist_n[best]) break;
            emu_log(m, "  legacy pc %08x  %u", legacy_hist_pc[best], legacy_hist_n[best]);
            legacy_hist_n[best] = 0;
        }
    }
    if (jit_prof_on) cmap_report(m);
    if (jit_prof_on) {
        emu_log(m, "JIT prof translate: %llu Mcycles, %llu bytes for %llu insns (%llu B/insn), %llu stubs",
                (unsigned long long)(prof_tr_cycles / 1000000), (unsigned long long)prof_tr_bytes,
                (unsigned long long)prof_tr_insns, (unsigned long long)(prof_tr_insns ? prof_tr_bytes / prof_tr_insns : 0),
                (unsigned long long)prof_tr_stubs);
        emu_log(m, "JIT prof: %u block executions, %u entries from C", prof_blk_exec, prof_c_entries);
        emu_log(m, "JIT exits to C: unlinked %u, tlb-guard %u, dispatch miss %u, budget %u (+inval %u), interp-end %u, sti %u",
                px_link, px_tag, px_disp, px_budget, px_inval - px_budget, px_interp, px_nolink);
        emu_log(m, "JIT ret frames: %u hit, %u miss", px_rethit, px_retmiss); px_rethit = px_retmiss = 0;
        px_link = px_tag = px_disp = px_budget = px_inval = px_interp = px_nolink = 0;
        { extern u32 stat_tlb_flush, stat_invlpg, stat_walks, stat_cr3;
          { extern u64 stat_glist_sum; emu_log(m, "TLB: %u flushes (%u cr3), %u invlpg, %u page walks, avg global list %u", stat_tlb_flush, stat_cr3, stat_invlpg, stat_walks, stat_tlb_flush ? (u32)(stat_glist_sum / stat_tlb_flush) : 0u); }
          stat_tlb_flush = stat_invlpg = stat_walks = stat_cr3 = 0; }
        prof_blk_exec = prof_c_entries = 0;
        emu_log(m, "JIT prof outside: no block %u, all-interpreted block %u, budget %u",
                prof_out_why[0], prof_out_why[1], prof_out_why[2]);
        e_memset(prof_out_why, 0, sizeof prof_out_why);
        prof_top(m, "fallback in block", prof_fb);
        prof_top(m, "slow path stub", prof_stub);
        prof_top(m, "outside blocks", prof_out);
    }
    if (cap) out[0] = 0;
}

#else  /* not an i386 host: interpreter only */

int jit_init(Machine *m, u8 *mem, u32 size) { (void)m; (void)mem; (void)size; return -1; }
int jit_prof_on;
u32 jit_off;
u32 jit_off2;
u32 jit_hot;
u64 jit_dump_rip;
void jit_prof_reset(void);
void jit_prof_reset(void) {}
void jit_prof_sample(Machine *m, uintptr_t pc);
void jit_prof_sample(Machine *m, uintptr_t pc) { (void)m; (void)pc; }
int jit_pc_block(Machine *m, uintptr_t pc, u64 *rip, uintptr_t *start, u32 *ninsns);
int jit_pc_block(Machine *m, uintptr_t pc, u64 *rip, uintptr_t *start, u32 *ninsns) { (void)m; (void)pc; (void)rip; (void)start; (void)ninsns; return 0; }
int jit_run_block(Cpu *c, u64 budget);
int jit_run_block(Cpu *c, u64 budget) { (void)c; (void)budget; return 0; }
void jit_invalidate_page(Machine *m, u64 pa);
void jit_invalidate_page(Machine *m, u64 pa) { (void)m; (void)pa; }
int jit_page_is_code(Machine *m, u64 pa);
int jit_page_is_code(Machine *m, u64 pa) { (void)m; (void)pa; return 0; }
void jit_flush_lookup(Machine *m);
void jit_flush_lookup(Machine *m) { (void)m; }
void jit_flush_code(Machine *m);
void jit_flush_code(Machine *m) { (void)m; }
void jit_stats(Machine *m, char *out, u32 cap);
void jit_stats(Machine *m, char *out, u32 cap) { (void)m; if (cap) out[0] = 0; }

#endif
