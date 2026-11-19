/* CPU conformance fuzzer: runs random register-form instructions natively on
 * the host CPU and in the emulator (32-bit protected mode, flat segments) from
 * the same register/flag state, and compares the results and the flags the
 * architecture defines for that instruction.
 *
 * Build: gcc -m32 ... tests/cpufuzz.c src/emu/*.c   (see Makefile: make cpufuzz)
 * Usage: cpufuzz [iterations] [seed]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "../src/emu/emu.h"

#define CF 0x001u
#define PF 0x004u
#define AF 0x010u
#define ZF 0x040u
#define SF 0x080u
#define OF 0x800u
#define ALL (CF | PF | AF | ZF | SF | OF)

typedef struct { u32 r[8]; u32 fl; } St;

/* memory operands: [EBX + disp8] or [EBX + ESI*s + disp8]; EBX = buffer base
 * (host address natively, GUEST_BUF in the emulator), ESI kept small. */
#define GUEST_BUF 0x40000u
#define BUFSZ 4096
static u8 nbuf[BUFSZ] __attribute__((aligned(64)));
static u8 init_buf[BUFSZ];
static int g_mem;          /* current instruction uses memory */
static int g_string;       /* string instruction: ESI/EDI/ECX set up as pointers/counts */

static u32 rng = 0x12345678u;
static u32 rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* ---------- native execution ---------- */
static u8 *nat;
static St nat_in, nat_out;
static u8 fpu_in[108], fpu_out[108];
static int g_fpu;          /* current test is an x87 instruction */
static int g_sse;          /* current test is an SSE instruction (fxrstor/fxsave around it) */
static u8 sse_in[512] __attribute__((aligned(16))), sse_out[512] __attribute__((aligned(16)));
static u8 sse_out_emu[512];
static long sse_flag_diffs;     /* only MXCSR exception flags differ */
static u32 nat_save_esp;

static void build_native(const u8 *insn, int len) {
    u8 *p = nat;
#define B(x) (*p++ = (u8)(x))
#define D(x) do { u32 _v = (u32)(x); memcpy(p, &_v, 4); p += 4; } while (0)
    B(0x60);                                   /* pushad */
    B(0x9c);                                   /* pushfd */
    B(0x89); B(0x25); D(&nat_save_esp);        /* mov [save], esp */
    B(0xff); B(0x35); D(&nat_in.fl); B(0x9d);  /* push [in.fl]; popfd */
    if (g_fpu) { B(0xdd); B(0x25); D(fpu_in); }            /* frstor [fpu_in] */
    if (g_sse) { B(0x0f); B(0xae); B(0x0d); D(sse_in); }   /* fxrstor [sse_in] */
    static const int order[7] = { 0, 1, 2, 3, 5, 6, 7 };
    for (int i = 0; i < 7; i++) { int r = order[i]; B(0x8b); B(0x05 | r << 3); D(&nat_in.r[r]); }
    memcpy(p, insn, (size_t)len); p += len;
    for (int i = 0; i < 7; i++) { int r = order[i]; B(0x89); B(0x05 | r << 3); D(&nat_out.r[r]); }
    B(0x9c); B(0x8f); B(0x05); D(&nat_out.fl);  /* pushfd; pop [out.fl] */
    if (g_fpu) { B(0xdd); B(0x35); D(fpu_out); B(0xdb); B(0xe3); }   /* fnsave [fpu_out]; fninit */
    if (g_sse) { B(0x0f); B(0xae); B(0x05); D(sse_out); B(0xdb); B(0xe3); }   /* fxsave [sse_out]; fninit */
    B(0x8b); B(0x25); D(&nat_save_esp);        /* mov esp, [save] */
    B(0x9d); B(0x61); B(0xc3);                 /* popfd; popad; ret */
}

/* ---------- emulated execution ---------- */
static Machine mach;
static u8 fpu_out_emu[108];
static u8 *ram;

static void log_cb(void *ctx, const char *s) { (void)ctx; fprintf(stderr, "[emu] %s\n", s); }

static int run_emu(const u8 *insn, int len, const St *in, St *out) {
    Cpu *c = &mach.cpu;
    memcpy(ram + 0x1000, insn, (size_t)len);
    ram[0x1000 + len] = 0xf4;
    if (g_fpu) {
        /* frstor [0x30000] ; insn ; fnsave [0x30100] executed by the emulator itself */
        memcpy(ram + 0x30000, fpu_in, 108);
        static const u8 pre[6] = { 0xdd, 0x25, 0x00, 0x00, 0x03, 0x00 };
        static const u8 post[6] = { 0xdd, 0x35, 0x00, 0x01, 0x03, 0x00 };
        memcpy(ram + 0x1000, pre, 6);
        memcpy(ram + 0x1006, insn, (size_t)len);
        memcpy(ram + 0x1006 + len, post, 6);
        len += 12;
    }
    if (g_sse) {
        /* fxrstor [0x30000] ; insn ; fxsave [0x30200] */
        memcpy(ram + 0x30000, sse_in, 512);
        static const u8 pre[7] = { 0x0f, 0xae, 0x0d, 0x00, 0x00, 0x03, 0x00 };
        static const u8 post[7] = { 0x0f, 0xae, 0x05, 0x00, 0x02, 0x03, 0x00 };
        memcpy(ram + 0x1000, pre, 7);
        memcpy(ram + 0x1007, insn, (size_t)len);
        memcpy(ram + 0x1007 + len, post, 7);
        len += 14;
    }
    for (int i = 0; i < 8; i++) c->r[i] = in->r[i];
    c->r[4] = 0x8000;
    memcpy(ram + GUEST_BUF, init_buf, BUFSZ);
    if (g_mem) {
        c->r[3] = GUEST_BUF + (in->r[3] - (u32)(uintptr_t)nbuf);
        if (g_string) { c->r[6] = GUEST_BUF + (in->r[6] - (u32)(uintptr_t)nbuf); c->r[7] = GUEST_BUF + (in->r[7] - (u32)(uintptr_t)nbuf); }
    }
    c->rflags = (in->fl & 0xcd5) | 2;
    c->lf_op = LF_NONE;
    c->rip = 0x1000;
    c->halted = 0;
    c->fatal = 0;
    for (int steps = 0; c->rip != 0x1000u + (u32)len; steps++) {
        if (c->fatal || c->halted || steps > 100000) return -1;
        cpu_step(c);
    }
    if (c->fatal) return -1;
    if (g_fpu) memcpy(fpu_out_emu, ram + 0x30100, 108);
    if (g_sse) memcpy(sse_out_emu, ram + 0x30200, 512);
    extern u64 get_flags(Cpu *c);
    for (int i = 0; i < 8; i++) out->r[i] = (u32)c->r[i];
    if (g_mem) {   /* rebase pointer registers to host addresses for comparison */
        out->r[3] = (u32)c->r[3] - GUEST_BUF + (u32)(uintptr_t)nbuf;
        if (g_string) { out->r[6] = (u32)c->r[6] - GUEST_BUF + (u32)(uintptr_t)nbuf; out->r[7] = (u32)c->r[7] - GUEST_BUF + (u32)(uintptr_t)nbuf; }
    }
    out->fl = (u32)get_flags(c);
    return 0;
}

/* ---------- instruction generator ---------- */
typedef struct { u8 b[16]; int len; u32 defined; const char *name; int skip_regs; } Gen;

static int reg3(void) { int r; do r = (int)(rnd() & 7); while (r == 4); return r; }   /* not ESP */
static u8 modrm_rr(int reg, int rm) { return (u8)(0xc0 | reg << 3 | rm); }

static void gen(Gen *g) {
    int k = (int)(rnd() % 22);
    int p66 = (rnd() & 3) == 0;
    int byte_op = 0;
    g->len = 0;
    g->defined = ALL;
    g->skip_regs = 0;
    if (p66) g->b[g->len++] = 0x66;
    int r1 = reg3(), r2 = reg3();
    switch (k) {
    case 0: case 1: {   /* ALU r/m, r (00-3B forms), byte or word/dword */
        int op = (int)(rnd() & 7);
        byte_op = rnd() & 1;
        if (byte_op && p66) g->len = 0;
        g->b[g->len++] = (u8)(op << 3 | (rnd() & 2) | (byte_op ? 0 : 1));
        int a = byte_op ? (int)(rnd() & 7) : r1, b = byte_op ? (int)(rnd() & 7) : r2;
        if (byte_op && ((a & 3) == 0 || (b & 3) == 0)) { a = 1; b = 2; }   /* avoid AL/AH vs SP-ish */
        g->b[g->len++] = modrm_rr(a, b);
        if (op == 1 || op == 4 || op == 6) g->defined = ALL & ~AF;
        g->name = "alu";
        break;
    }
    case 2: {   /* ALU with immediate 81/83 */
        int op = (int)(rnd() & 7);
        int imm8 = rnd() & 1;
        g->b[g->len++] = imm8 ? 0x83 : 0x81;
        g->b[g->len++] = (u8)(0xc0 | op << 3 | r1);
        if (imm8) g->b[g->len++] = (u8)rnd();
        else for (int i = 0; i < (p66 ? 2 : 4); i++) g->b[g->len++] = (u8)rnd();
        if (op == 1 || op == 4 || op == 6) g->defined = ALL & ~AF;
        g->name = "alu-imm";
        break;
    }
    case 3: case 4: {   /* shifts/rotates by imm8, 1, CL */
        int op = (int)(rnd() & 7);
        if (op == 6) op = 4;
        int form = (int)(rnd() % 3);
        g->b[g->len++] = form == 0 ? 0xc1 : form == 1 ? 0xd1 : 0xd3;
        g->b[g->len++] = (u8)(0xc0 | op << 3 | (r1 == 1 ? 0 : r1));
        int cnt = form == 0 ? (int)(rnd() & 31) : form == 1 ? 1 : -1;
        if (form == 0) g->b[g->len++] = (u8)cnt;
        /* defined: CF (count != 0, count <= width), OF only for count 1, SZP for shifts, AF undefined */
        g->name = op < 4 ? "rotate" : "shift";
        g->defined = 0x1000;                 /* decided at compare time from the count */
        break;
    }
    case 5: {   /* INC/DEC/NEG/NOT */
        int which = (int)(rnd() & 3);
        if (which < 2) { g->b[g->len++] = 0xff; g->b[g->len++] = (u8)(0xc0 | which << 3 | r1); g->defined = ALL & ~CF; }
        else { g->b[g->len++] = 0xf7; g->b[g->len++] = (u8)(0xc0 | (which == 2 ? 3 : 2) << 3 | r1); g->defined = which == 2 ? ALL : 0; }
        g->name = "incdec-neg-not";
        break;
    }
    case 6: {   /* MUL / IMUL one operand: only CF/OF defined */
        g->b[g->len++] = 0xf7; g->b[g->len++] = (u8)(0xc0 | (4 + (rnd() & 1)) << 3 | r1);
        g->defined = CF | OF; g->name = "mul1";
        break;
    }
    case 7: {   /* IMUL r, r/m and imm forms */
        int f = (int)(rnd() % 3);
        if (f == 0) { g->b[g->len++] = 0x0f; g->b[g->len++] = 0xaf; g->b[g->len++] = modrm_rr(r1, r2); }
        else { g->b[g->len++] = f == 1 ? 0x6b : 0x69; g->b[g->len++] = modrm_rr(r1, r2);
               for (int i = 0; i < (f == 1 ? 1 : (p66 ? 2 : 4)); i++) g->b[g->len++] = (u8)rnd(); }
        g->defined = CF | OF; g->name = "imul";
        break;
    }
    case 8: {   /* DIV/IDIV: inputs arranged not to fault */
        g->b[g->len++] = 0xf7; g->b[g->len++] = (u8)(0xc0 | (6 + (rnd() & 1)) << 3 | 3);   /* EBX divisor */
        g->defined = 0; g->name = "div"; g->skip_regs = 1;
        break;
    }
    case 9: {   /* BT/BTS/BTR/BTC reg, reg and imm */
        if (rnd() & 1) { static const u8 o[4] = { 0xa3, 0xab, 0xb3, 0xbb }; g->b[g->len++] = 0x0f; g->b[g->len++] = o[rnd() & 3]; g->b[g->len++] = modrm_rr(r2, r1); }
        else { g->b[g->len++] = 0x0f; g->b[g->len++] = 0xba; g->b[g->len++] = (u8)(0xc0 | (4 + (rnd() & 3)) << 3 | r1); g->b[g->len++] = (u8)rnd(); }
        g->defined = CF; g->name = "bt";
        break;
    }
    case 10: {  /* BSF/BSR: ZF defined; destination undefined when source is 0 */
        g->b[g->len++] = 0x0f; g->b[g->len++] = (rnd() & 1) ? 0xbc : 0xbd; g->b[g->len++] = modrm_rr(r1, r2);
        g->defined = ZF; g->name = "bsf/bsr"; g->skip_regs = 3;
        break;
    }
    case 11: {  /* SHLD/SHRD imm and CL */
        int cl = rnd() & 1;
        g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)((rnd() & 1 ? 0xa4 : 0xac) + cl);
        g->b[g->len++] = modrm_rr(r2 == 1 ? 2 : r2, r1 == 1 ? 3 : r1);
        if (!cl) g->b[g->len++] = (u8)(rnd() & 31);
        g->defined = 0x1000; g->name = "shld/shrd"; g->skip_regs = cl ? 2 : 0;
        break;
    }
    case 12: {  /* SETcc / CMOVcc */
        int cc = (int)(rnd() & 15);
        if (rnd() & 1) { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0x90 | cc); g->b[g->len++] = (u8)(0xc0 | (rnd() & 7)); }
        else { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0x40 | cc); g->b[g->len++] = modrm_rr(r1, r2); }
        g->defined = ALL; g->name = "setcc/cmov";
        break;
    }
    case 13: {  /* ADC/SBB with random carry in */
        g->b[g->len++] = (u8)((rnd() & 1 ? 0x11 : 0x19) | (rnd() & 2)); g->b[g->len++] = modrm_rr(r1, r2);
        g->name = "adc/sbb";
        break;
    }
    case 14: {  /* XADD / CMPXCHG reg forms */
        g->b[g->len++] = 0x0f; g->b[g->len++] = (rnd() & 1) ? 0xc1 : 0xb1; g->b[g->len++] = modrm_rr(r2, r1);
        g->name = "xadd/cmpxchg";
        break;
    }
    case 15: {  /* MOVZX/MOVSX, BSWAP, CBW/CWDE/CDQ */
        int f = (int)(rnd() % 4);
        if (f < 2) { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)((f ? 0xbe : 0xb6) | (rnd() & 1)); g->b[g->len++] = modrm_rr(r1, r2 & 3); }
        else if (f == 2) { if (p66) g->len = 0; g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0xc8 | r1); }
        else g->b[g->len++] = (rnd() & 1) ? 0x98 : 0x99;
        g->defined = ALL; g->name = "movx/bswap/cext";
        break;
    }
    case 16: {  /* LAHF/SAHF, CMC/CLC/STC */
        static const u8 o[5] = { 0x9f, 0x9e, 0xf5, 0xf8, 0xf9 };
        g->len = 0; g->b[g->len++] = o[rnd() % 5];
        g->defined = ALL; g->name = "flagops";
        break;
    }
    case 17: {  /* TEST r/m, imm; TEST r, r; 8-bit ALU imm (80) */
        if (rnd() & 1) { g->b[g->len++] = 0xf7; g->b[g->len++] = (u8)(0xc0 | r1); for (int i = 0; i < (p66 ? 2 : 4); i++) g->b[g->len++] = (u8)rnd(); g->defined = ALL & ~AF; }
        else { if (p66) g->len = 0; g->b[g->len++] = 0x80; int op = (int)(rnd() & 7); g->b[g->len++] = (u8)(0xc0 | op << 3 | (1 + (rnd() % 3))); g->b[g->len++] = (u8)rnd(); g->defined = (op == 1 || op == 4 || op == 6) ? ALL & ~AF : ALL; }
        g->name = "test/alu8";
        break;
    }
    case 18: {  /* DAA/DAS/AAA/AAS (BCD) */
        static const u8 o[4] = { 0x27, 0x2f, 0x37, 0x3f };
        g->len = 0; int w = (int)(rnd() & 3); g->b[g->len++] = o[w];
        g->defined = w < 2 ? (CF | AF | SF | ZF | PF) : (CF | AF); g->name = "bcd";
        break;
    }
    case 19: {  /* RCL/RCR by 1 and imm with carry in */
        g->b[g->len++] = 0xc1; g->b[g->len++] = (u8)(0xc0 | (2 + (rnd() & 1)) << 3 | r1); g->b[g->len++] = (u8)(rnd() & 31);
        g->defined = 0x1000; g->name = "rcl/rcr";
        break;
    }
    case 20: {  /* XCHG, NOP forms, LEA */
        if (rnd() & 1) { g->b[g->len++] = 0x87; g->b[g->len++] = modrm_rr(r1, r2); }
        else { g->b[g->len++] = 0x8d; g->b[g->len++] = (u8)(0x80 | r1 << 3 | r2); for (int i = 0; i < 4; i++) g->b[g->len++] = (u8)rnd(); }
        g->defined = ALL; g->name = "xchg/lea";
        break;
    }
    default: {  /* POPCNT */
        g->b[g->len++] = 0xf3; g->b[g->len++] = 0x0f; g->b[g->len++] = 0xb8; g->b[g->len++] = modrm_rr(r1, r2);
        g->defined = ALL; g->name = "popcnt";
        break;
    }
    }
}

/* memory-operand forms: ModRM [EBX+disp8] (01 reg 011) or [EBX+ESI*s+disp8] */
static int mem_modrm(u8 *b, int reg) {
    int disp = (int)(rnd() % 120);
    if (rnd() & 1) { b[0] = (u8)(0x43 | reg << 3); b[1] = (u8)disp; return 2; }
    b[0] = (u8)(0x44 | reg << 3); b[1] = (u8)((rnd() & 3) << 6 | 6 << 3 | 3); b[2] = (u8)disp; return 3;
}

static void gen_mem(Gen *g) {
    int k = (int)(rnd() % 12);
    int p66 = (rnd() & 3) == 0;
    g->len = 0; g->defined = ALL; g->skip_regs = 0; g_string = 0;
    if (p66 && k != 9) g->b[g->len++] = 0x66;
    int r = (int)(rnd() & 7); if (r == 4 || r == 3 || r == 6 || r == 7) r = 0;   /* not ESP/EBX/ESI (pointers), not BH */
    switch (k) {
    case 0: { int op = (int)(rnd() & 7); g->b[g->len++] = (u8)(op << 3 | (rnd() & 3)); g->len += mem_modrm(g->b + g->len, r);
              if (op == 1 || op == 4 || op == 6) g->defined = ALL & ~AF; g->name = "m-alu"; break; }
    case 1: { int op = (int)(rnd() & 7); int i8 = rnd() & 1; g->b[g->len++] = i8 ? 0x83 : 0x81; g->len += mem_modrm(g->b + g->len, op);
              for (int i = 0; i < (i8 ? 1 : p66 ? 2 : 4); i++) g->b[g->len++] = (u8)rnd();
              if (op == 1 || op == 4 || op == 6) g->defined = ALL & ~AF; g->name = "m-alu-imm"; break; }
    case 2: { int op = (int)(rnd() & 7); if (op == 6) op = 4; g->b[g->len++] = 0xc1; g->len += mem_modrm(g->b + g->len, op); g->b[g->len++] = (u8)(rnd() & 31);
              g->defined = 0x1000; g->name = "m-shift"; break; }
    case 3: { int w = (int)(rnd() % 4); g->b[g->len++] = w < 2 ? 0xff : 0xf7; g->len += mem_modrm(g->b + g->len, w < 2 ? w : w == 2 ? 3 : 2);
              g->defined = w < 2 ? ALL & ~CF : w == 2 ? ALL : 0; g->name = "m-incdec-neg-not"; break; }
    case 4: { static const u8 o[4] = { 0xa3, 0xab, 0xb3, 0xbb }; g->b[g->len++] = 0x0f; g->b[g->len++] = o[rnd() & 3];
              g->len += mem_modrm(g->b + g->len, 1); g->defined = CF; g->name = "m-bt-reg"; g->skip_regs = 4; break; }
    case 5: { g->b[g->len++] = 0x0f; g->b[g->len++] = (rnd() & 1) ? 0xc1 : 0xb1; g->len += mem_modrm(g->b + g->len, r); g->name = "m-xadd/cmpxchg"; break; }
    case 6: { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0xb6 | (rnd() & 1) | ((rnd() & 1) << 3)); g->len += mem_modrm(g->b + g->len, r); g->name = "m-movx"; break; }
    case 7: { int cc = (int)(rnd() & 15); if (rnd() & 1) { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0x40 | cc); }
              else { g->len = 0; g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)(0x90 | cc); }
              g->len += mem_modrm(g->b + g->len, r); g->name = "m-cmov/setcc"; break; }
    case 8: { g->b[g->len++] = 0x0f; g->b[g->len++] = (u8)((rnd() & 1 ? 0xa4 : 0xac)); g->len += mem_modrm(g->b + g->len, r == 1 ? 2 : r);
              g->b[g->len++] = (u8)(rnd() & 31); g->defined = 0x1000; g->name = "m-shld/shrd"; break; }
    case 9: {   /* string ops, optional REP/REPE/REPNE, both directions */
        static const u8 o[10] = { 0xa4, 0xa5, 0xa6, 0xa7, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf };
        int std = rnd() & 1;
        if (std) g->b[g->len++] = 0xfd;
        int rep = (int)(rnd() % 3);
        if (rep) g->b[g->len++] = rep == 1 ? 0xf3 : 0xf2;
        if (p66) g->b[g->len++] = 0x66;
        g->b[g->len++] = o[rnd() % 10];
        g->b[g->len++] = 0xfc;              /* cld again so the host returns sane */
        g_string = 1; g->name = "string"; g->skip_regs = 5;
        break;
    }
    case 10: { g->b[g->len++] = 0x87; g->len += mem_modrm(g->b + g->len, r); g->defined = ALL; g->name = "m-xchg"; break; }
    default: { g->b[g->len++] = 0xf7; g->len += mem_modrm(g->b + g->len, 4 + (int)(rnd() & 1)); g->defined = CF | OF; g->name = "m-mul"; break; }
    }
}

static void rand_f80(u8 *p) {
    int kind = (int)(rnd() % 10);
    for (int i = 0; i < 8; i++) p[i] = (u8)rnd();
    p[7] |= 0x80;                                    /* normal (explicit integer bit) */
    u32 exp = 0x3fff + (rnd() % 64) - 32;
    if (kind == 0) { memset(p, 0, 8); exp = 0; }                       /* zero */
    else if (kind == 1) { memset(p, 0, 8); p[7] = 0x80; exp = 0x7fff; } /* infinity */
    else if (kind == 2) { p[7] |= 0xc0; exp = 0x7fff; }                 /* QNaN */
    else if (kind == 3) { u64 iv = rnd() % 1000; memset(p, 0, 8); if (iv) { int sh = 63 - (31 - __builtin_clz((u32)iv)); u64 m = iv << sh; memcpy(p, &m, 8); exp = 0x3fff + 63 - sh; } else exp = 0; }
    p[8] = (u8)exp; p[9] = (u8)((exp >> 8) | ((rnd() & 1) ? 0x80 : 0));
}

/* MMX and SSSE3-on-MMX instructions; state is compared through the x87 save area */
static void gen_mmx(Gen *g) {
    static const u8 bin[] = { 0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x6b,0x74,0x75,0x76,
        0xd1,0xd2,0xd3,0xd4,0xd5,0xd8,0xd9,0xda,0xdb,0xdc,0xdd,0xde,0xdf,0xe0,0xe1,0xe2,0xe3,0xe4,0xe5,
        0xe8,0xe9,0xea,0xeb,0xec,0xed,0xee,0xef,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf8,0xf9,0xfa,0xfb,0xfc,0xfd,0xfe,0x6f };
    static const u8 s3[] = { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x1c,0x1d,0x1e };
    g->len = 0; g->defined = 0; g->skip_regs = 0; g_string = 0;
    int r1 = (int)(rnd() & 7), r2 = (int)(rnd() & 7), mem = (rnd() & 1) != 0;
    u8 *b = g->b;
    int k = (int)(rnd() % 12);
    g->name = "mmx";
    if (k < 6) {                                                     /* binary op / movq load */
        b[g->len++] = 0x0f; b[g->len++] = bin[rnd() % sizeof bin];
    } else if (k == 6) {                                             /* SSSE3 */
        b[g->len++] = 0x0f; b[g->len++] = 0x38; b[g->len++] = s3[rnd() % sizeof s3]; g->name = "mmx-ssse3";
    } else if (k == 7) {                                             /* PALIGNR / PSHUFW / PINSRW + imm */
        int f = (int)(rnd() % 3);
        if (f == 0) { b[g->len++] = 0x0f; b[g->len++] = 0x3a; b[g->len++] = 0x0f; }
        else { b[g->len++] = 0x0f; b[g->len++] = f == 1 ? 0x70 : 0xc4; }
        if (f == 2 && !mem) { b[g->len++] = modrm_rr(r1, reg3()); b[g->len++] = (u8)rnd(); g->name = "mmx-imm"; return; }
        if (mem) { g_mem = 1; g->len += mem_modrm(b + g->len, r1); } else b[g->len++] = modrm_rr(r1, r2);
        b[g->len++] = (u8)(f == 0 ? rnd() % 20 : rnd());
        g->name = "mmx-imm";
        return;
    } else if (k == 8) {                                             /* shift by immediate */
        static const u8 sh[][2] = { {0x71,2},{0x71,4},{0x71,6},{0x72,2},{0x72,4},{0x72,6},{0x73,2},{0x73,6} };
        int f = (int)(rnd() % 8);
        b[g->len++] = 0x0f; b[g->len++] = sh[f][0]; b[g->len++] = modrm_rr(sh[f][1], r2);
        b[g->len++] = (u8)(rnd() % 70);
        g->name = "mmx-shimm";
        return;
    } else if (k == 9) {                                             /* MOVD mm<->r32, PEXTRW, PMOVMSKB */
        int f = (int)(rnd() % 4), gr = reg3();
        static const u8 o[4] = { 0x6e, 0x7e, 0xc5, 0xd7 };
        b[g->len++] = 0x0f; b[g->len++] = o[f];
        if (f == 0) b[g->len++] = modrm_rr(r1, gr);
        else if (f == 1) b[g->len++] = modrm_rr(r1, gr);
        else b[g->len++] = modrm_rr(gr, r2);
        if (f == 2) b[g->len++] = (u8)rnd();
        g->name = "mmx-gpr";
        return;
    } else if (k == 10) {                                            /* stores */
        b[g->len++] = 0x0f; b[g->len++] = (rnd() & 1) ? 0x7f : 0xe7;
        g_mem = 1; g->len += mem_modrm(b + g->len, r1);
        g->name = "mmx-store";
        return;
    } else {                                                         /* EMMS */
        b[g->len++] = 0x0f; b[g->len++] = 0x77; g->name = "emms"; return;
    }
    if (mem) { g_mem = 1; g->len += mem_modrm(b + g->len, r1); } else b[g->len++] = modrm_rr(r1, r2);
}

/* SSE..SSE4.2 (what the emulator advertises). Memory operands are 16-byte aligned. */
enum { S_I = 1, S_MEM = 2, S_REG = 4, S_GREG = 8, S_GRM = 16, S_FL = 32 };
typedef struct { u8 pfx, map, op, f; } SseOp;
static const SseOp sse_ops[] = {
#define P0(o, f) { 0, 0, o, f }
#define P66(o, f) { 1, 0, o, f }
#define PF3(o, f) { 3, 0, o, f }
#define PF2(o, f) { 2, 0, o, f }
#define P38(o, f) { 1, 0x38, o, f }
#define P3A(o, f) { 1, 0x3a, o, S_I | (f) }
    P0(0x10,0),P0(0x11,0),P0(0x12,0),P0(0x13,S_MEM),P0(0x14,0),P0(0x15,0),P0(0x16,0),P0(0x17,S_MEM),P0(0x28,0),P0(0x29,0),
    P0(0x2e,S_FL),P0(0x2f,S_FL),P0(0x50,S_GREG|S_REG),P0(0x51,0),P0(0x54,0),P0(0x55,0),P0(0x56,0),P0(0x57,0),P0(0x58,0),P0(0x59,0),
    P0(0x5a,0),P0(0x5b,0),P0(0x5c,0),P0(0x5d,0),P0(0x5e,0),P0(0x5f,0),P0(0xc2,S_I),P0(0xc6,S_I),
    P66(0x10,0),P66(0x11,0),P66(0x12,S_MEM),P66(0x13,S_MEM),P66(0x14,0),P66(0x15,0),P66(0x16,S_MEM),P66(0x17,S_MEM),P66(0x28,0),P66(0x29,0),
    P66(0x2e,S_FL),P66(0x2f,S_FL),P66(0x50,S_GREG|S_REG),P66(0x51,0),P66(0x54,0),P66(0x55,0),P66(0x56,0),P66(0x57,0),P66(0x58,0),P66(0x59,0),
    P66(0x5a,0),P66(0x5b,0),P66(0x5c,0),P66(0x5d,0),P66(0x5e,0),P66(0x5f,0),
    P66(0x60,0),P66(0x61,0),P66(0x62,0),P66(0x63,0),P66(0x64,0),P66(0x65,0),P66(0x66,0),P66(0x67,0),P66(0x68,0),P66(0x69,0),P66(0x6a,0),
    P66(0x6b,0),P66(0x6c,0),P66(0x6d,0),P66(0x6e,S_GRM),P66(0x6f,0),P66(0x70,S_I),P66(0x74,0),P66(0x75,0),P66(0x76,0),P66(0x7c,0),P66(0x7d,0),
    P66(0x7e,S_GRM),P66(0x7f,0),P66(0xc2,S_I),P66(0xc4,S_I|S_GRM),P66(0xc5,S_I|S_GREG|S_REG),P66(0xc6,S_I),
    P66(0xd0,0),P66(0xd1,0),P66(0xd2,0),P66(0xd3,0),P66(0xd4,0),P66(0xd5,0),P66(0xd6,0),P66(0xd7,S_GREG|S_REG),P66(0xd8,0),P66(0xd9,0),
    P66(0xda,0),P66(0xdb,0),P66(0xdc,0),P66(0xdd,0),P66(0xde,0),P66(0xdf,0),P66(0xe0,0),P66(0xe1,0),P66(0xe2,0),P66(0xe3,0),P66(0xe4,0),
    P66(0xe5,0),P66(0xe6,0),P66(0xe7,S_MEM),P66(0xe8,0),P66(0xe9,0),P66(0xea,0),P66(0xeb,0),P66(0xec,0),P66(0xed,0),P66(0xee,0),P66(0xef,0),
    P66(0xf1,0),P66(0xf2,0),P66(0xf3,0),P66(0xf4,0),P66(0xf5,0),P66(0xf6,0),P66(0xf8,0),P66(0xf9,0),P66(0xfa,0),P66(0xfb,0),P66(0xfc,0),
    P66(0xfd,0),P66(0xfe,0),
    PF3(0x10,0),PF3(0x11,0),PF3(0x12,0),PF3(0x16,0),PF3(0x2a,S_GRM),PF3(0x2c,S_GREG),PF3(0x2d,S_GREG),PF3(0x51,0),PF3(0x58,0),PF3(0x59,0),
    PF3(0x5a,0),PF3(0x5b,0),PF3(0x5c,0),PF3(0x5d,0),PF3(0x5e,0),PF3(0x5f,0),PF3(0x6f,0),PF3(0x70,S_I),PF3(0x7e,0),PF3(0x7f,0),PF3(0xc2,S_I),PF3(0xe6,0),
    PF2(0x10,0),PF2(0x11,0),PF2(0x12,0),PF2(0x2a,S_GRM),PF2(0x2c,S_GREG),PF2(0x2d,S_GREG),PF2(0x51,0),PF2(0x58,0),PF2(0x59,0),PF2(0x5a,0),
    PF2(0x5c,0),PF2(0x5d,0),PF2(0x5e,0),PF2(0x5f,0),PF2(0x70,S_I),PF2(0x7c,0),PF2(0x7d,0),PF2(0xc2,S_I),PF2(0xd0,0),PF2(0xe6,0),PF2(0xf0,S_MEM),
    P38(0x00,0),P38(0x01,0),P38(0x02,0),P38(0x03,0),P38(0x04,0),P38(0x05,0),P38(0x06,0),P38(0x07,0),P38(0x08,0),P38(0x09,0),P38(0x0a,0),
    P38(0x0b,0),P38(0x10,0),P38(0x14,0),P38(0x15,0),P38(0x17,S_FL),P38(0x1c,0),P38(0x1d,0),P38(0x1e,0),P38(0x20,0),P38(0x21,0),P38(0x22,0),
    P38(0x23,0),P38(0x24,0),P38(0x25,0),P38(0x28,0),P38(0x29,0),P38(0x2a,S_MEM),P38(0x2b,0),P38(0x30,0),P38(0x31,0),P38(0x32,0),P38(0x33,0),
    P38(0x34,0),P38(0x35,0),P38(0x37,0),P38(0x38,0),P38(0x39,0),P38(0x3a,0),P38(0x3b,0),P38(0x3c,0),P38(0x3d,0),P38(0x3e,0),P38(0x3f,0),
    P38(0x40,0),P38(0x41,0),
    P3A(0x08,0),P3A(0x09,0),P3A(0x0a,0),P3A(0x0b,0),P3A(0x0c,0),P3A(0x0d,0),P3A(0x0e,0),P3A(0x0f,0),P3A(0x14,S_GRM),P3A(0x15,S_GRM),
    P3A(0x16,S_GRM),P3A(0x17,S_GRM),P3A(0x20,S_GRM),P3A(0x21,0),P3A(0x22,S_GRM),P3A(0x40,0),P3A(0x41,0),P3A(0x42,0),
    P3A(0x60,S_FL),P3A(0x61,S_FL),P3A(0x62,S_FL),P3A(0x63,S_FL),
};

static void gen_sse(Gen *g) {
    const SseOp *o;
    int shift = (rnd() % 12) == 0;
    g->len = 0; g->defined = ALL; g->skip_regs = 0; g_string = 0;
    u8 *b = g->b;
    if ((rnd() % 16) == 0) {                           /* MOVQ2DQ xmm,mm / MOVDQ2Q mm,xmm (MMX state in the FXSAVE area) */
        int f3 = rnd() & 1;
        b[g->len++] = f3 ? 0xf3 : 0xf2; b[g->len++] = 0x0f; b[g->len++] = 0xd6;
        b[g->len++] = modrm_rr((int)(rnd() & 7), (int)(rnd() & 7));
        g->name = f3 ? "movq2dq" : "movdq2q";
        return;
    }
    if (shift) {                                       /* 66 0F 71-73 /n ib */
        static const u8 sh[][2] = { {0x71,2},{0x71,4},{0x71,6},{0x72,2},{0x72,4},{0x72,6},{0x73,2},{0x73,3},{0x73,6},{0x73,7} };
        int f = (int)(rnd() % 10);
        b[g->len++] = 0x66; b[g->len++] = 0x0f; b[g->len++] = sh[f][0];
        b[g->len++] = modrm_rr(sh[f][1], (int)(rnd() & 7)); b[g->len++] = (u8)(rnd() % 70);
        g->name = "sse-shimm";
        return;
    }
    o = &sse_ops[rnd() % (sizeof sse_ops / sizeof sse_ops[0])];
    static const u8 pf[4] = { 0, 0x66, 0xf2, 0xf3 };
    if (o->pfx) b[g->len++] = pf[o->pfx];
    b[g->len++] = 0x0f;
    if (o->map) b[g->len++] = o->map;
    b[g->len++] = o->op;
    int mem = (o->f & S_MEM) ? 1 : (o->f & S_REG) ? 0 : (int)(rnd() & 1);
    int reg = (o->f & S_GREG) ? reg3() : (int)(rnd() & 7);
    if ((o->f & S_GREG) && reg == 3) reg = 2;          /* EBX is the memory base */
    if (mem) {
        g_mem = 1;
        b[g->len++] = (u8)(0x43 | reg << 3);           /* [EBX + disp8], 16-byte aligned */
        b[g->len++] = (u8)(16 * (rnd() % 8));
    } else {
        int rm = (o->f & S_GRM) ? reg3() : (int)(rnd() & 7);
        b[g->len++] = modrm_rr(reg, rm);
    }
    if (o->f & S_I) {
        u8 imm = (u8)rnd();
        if (o->map == 0x3a && o->op >= 0x08 && o->op <= 0x0b) imm &= 0x0f;   /* ROUND: bits 4-7 reserved */
        if ((o->op == 0xc2) && !o->map) imm &= 7;                            /* CMPccPS: 8 predicates (no AVX) */
        b[g->len++] = imm;
    }
    g->name = "sse";
    if (o->f & S_FL) g->name = "sse-flags";
}

static u32 rand_lane(void) {
    static const u32 sp[] = { 0, 0x80000000u, 0x3f800000u, 0xbfc00000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7fa00000u,
                              0x00000001u, 0x007fffffu, 0x7f7fffffu, 0x4f000000u, 0xcf000000u, 0x3effffffu, 0x40490fdbu };
    switch (rnd() % 6) {
    case 0: return sp[rnd() % (sizeof sp / sizeof sp[0])];
    case 1: return rnd() & 0xff;
    case 2: { float f = (float)((i32)rnd() % 100000) / (float)(1 + (rnd() & 255)); u32 v; memcpy(&v, &f, 4); return v; }
    default: return rnd();
    }
}

static void sse_setup(void) {
    memset(sse_in, 0, sizeof sse_in);
    u16 fcw = 0x37f; memcpy(sse_in, &fcw, 2);
    u32 mx = 0x1f80u | (rnd() & 3) << 13 | ((rnd() & 3) == 0 ? 0x8000u : 0) | ((rnd() & 3) == 0 ? 0x40u : 0);
    if (getenv("FUZZ_MXCSR")) mx = (u32)strtoul(getenv("FUZZ_MXCSR"), 0, 16);
    memcpy(sse_in + 24, &mx, 4);
    for (int i = 0; i < 8; i++) {                       /* MMX registers (x87 ST slots, all valid) */
        u32 *m = (u32 *)(void *)(sse_in + 32 + 16 * i);
        m[0] = rand_lane(); m[1] = rand_lane(); m[2] = 0xffff;
    }
    sse_in[4] = 0xff;                                   /* abridged FTW: all valid */
    for (int i = 0; i < 8; i++) {
        u32 *x = (u32 *)(void *)(sse_in + 160 + 16 * i);
        for (int k = 0; k < 4; k++) x[k] = rand_lane();
        if ((rnd() & 3) == 0) {                         /* some lanes as doubles */
            double d = (double)((i32)rnd()) / (double)(1 + (rnd() & 1023));
            memcpy(x, &d, 8);
        }
    }
}

static void gen_fpu(Gen *g) {
    g->len = 0; g->defined = ALL; g->skip_regs = 0; g_string = 0;
    int i = (int)(rnd() & 7);
    switch (rnd() % 10) {
    case 0: g->b[g->len++] = 0xd8; g->b[g->len++] = (u8)(0xc0 | (rnd() & 0x3f)); g->name = "x87 D8 reg"; break;
    case 1: g->b[g->len++] = 0xdc; g->b[g->len++] = (u8)(0xc0 | (rnd() & 0x3f)); g->name = "x87 DC reg"; break;
    case 2: g->b[g->len++] = 0xde; g->b[g->len++] = (u8)(0xc0 | (rnd() & 0x3f)); if ((g->b[1] & 0x38) == 0x18 || (g->b[1] & 0x38) == 0x10) g->b[1] = 0xd9; g->name = "x87 DE reg"; break;
    case 3: { static const u8 o[] = { 0xe0, 0xe1, 0xe4, 0xe5, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4,
                                     0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff, 0xd0 };
              g->b[g->len++] = 0xd9; g->b[g->len++] = o[rnd() % sizeof o]; g->name = "x87 D9 misc"; break; }
    case 4: g->b[g->len++] = 0xd9; g->b[g->len++] = (u8)((rnd() & 1 ? 0xc0 : 0xc8) | i); g->name = "fld/fxch st(i)"; break;
    case 5: g->b[g->len++] = 0xdd; g->b[g->len++] = (u8)(0xc0 | (rnd() & 0x3f)); if ((g->b[1] & 0x38) == 0x08 || (g->b[1] & 0x38) >= 0x30) g->b[1] = 0xd0 | (u8)i; g->name = "x87 DD reg"; break;
    case 6: { static const u8 o[] = { 0xc0, 0xc8, 0xd0, 0xd8, 0xe8, 0xf0 };   /* fcmov*, fucomi, fcomi */
              g->b[g->len++] = (rnd() & 1) ? 0xda : 0xdb; g->b[g->len++] = (u8)(o[rnd() % 6] | i);
              if (g->b[0] == 0xda && g->b[1] >= 0xe8) g->b[1] = 0xe9;      /* DA E9 FUCOMPP */
              g->name = "fcmov/fcomi"; g->defined = CF | ZF | PF; break; }
    case 7: { /* memory loads/stores/arith: [EBX+disp8] */
        static const u8 ops[][2] = { {0xd8,0},{0xd8,1},{0xd8,2},{0xd8,4},{0xd8,5},{0xd8,6},{0xd8,7},{0xdc,0},{0xdc,4},{0xdc,7},
                                     {0xd9,0},{0xd9,2},{0xd9,3},{0xdd,0},{0xdd,2},{0xdd,3},{0xdb,0},{0xdb,2},{0xdb,3},{0xdb,5},{0xdb,7},
                                     {0xdf,0},{0xdf,3},{0xdf,5},{0xdf,7},{0xda,0},{0xda,6},{0xde,1},{0xdf,4},{0xdf,6},{0xdd,1},{0xdb,1},{0xdf,1} };
        int k = (int)(rnd() % (sizeof ops / 2));
        g->b[g->len++] = ops[k][0]; g->b[g->len++] = (u8)(0x43 | ops[k][1] << 3); g->b[g->len++] = (u8)(rnd() % 100);
        g_mem = 1; g->name = "x87 mem"; break;
    }
    case 8: g->b[g->len++] = 0xdf; g->b[g->len++] = (u8)((rnd() & 1 ? 0xe8 : 0xf0) | i); g->name = "fcomip/fucomip"; g->defined = CF | ZF | PF; break;
    default: g->b[g->len++] = 0xdf; g->b[g->len++] = 0xe0; g->name = "fnstsw ax"; break;
    }
}

/* shift/rotate flag rules given the actual count (host and SDM agree on these) */
static u32 shift_defined(const Gen *g, const St *in) {
    int p = g->b[0] == 0x66;
    int width = p ? 16 : 32;
    u8 opc = g->b[p], m = g->b[p + 1];
    if (opc == 0x0f) {   /* SHLD/SHRD */
        u8 o2 = m;
        int cnt = (o2 & 1) ? (int)(in->r[1] & 31) : g->b[g->len - 1] & 31;
        if (!cnt) return ALL;
        if (cnt > width) return 0x2000;      /* result and flags undefined */
        return (cnt == 1 ? OF : 0) | CF | SF | ZF | PF;
    }
    int op = (g->b[p + 1] >> 3) & 7;
    int cnt = opc == 0xd3 ? (int)(in->r[1] & 31) : opc == 0xd1 ? 1 : g->b[g->len - 1] & 31;
    if (!cnt) return ALL;
    if (op == 2 || op == 3) {   /* RCL/RCR */
        return (cnt == 1 ? OF : 0) | CF;
    }
    if (op < 2) {               /* ROL/ROR: CF defined, OF only for 1 */
        return (cnt == 1 ? OF : 0) | CF;
    }
    if (cnt > width && width == 16) return SF | ZF | PF;
    return (cnt == 1 ? OF : 0) | CF | SF | ZF | PF;
}

int main(int argc, char **argv) {
    long iters = argc > 1 ? atol(argv[1]) : 200000;
    if (argc > 2) rng = (u32)strtoul(argv[2], 0, 0);
    nat = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ram = calloc(1, 1 << 20);
    Host h = { 0, 0, log_cb, 0, 0, 0, 0 };
    machine_init(&mach, ram, 1 << 20, h);
    mach.cpu.cr4 = 0x600;
    cpu_set_flat32(&mach.cpu);
    long fails = 0, done = 0;
    static long per_fail[32];
    for (long it = 0; it < iters && fails < 40; it++) {
        Gen g;
        int kind = (int)(rnd() % 4);
        g_fpu = kind == 0;
        g_sse = kind == 1;
        g_mem = 0;
        if (g_sse) { gen_sse(&g); sse_setup(); }
        else if (g_fpu) { if (rnd() & 1) gen_mmx(&g); else gen_fpu(&g); }
        else { g_mem = rnd() & 1; if (g_mem) gen_mem(&g); else { g_string = 0; gen(&g); } }
        if (g_fpu) {
            memset(fpu_in, 0, sizeof fpu_in);
            u16 cw = (u16)(0x37f & ~0x0f00) | (u16)((rnd() & 3) << 10) | (u16)(((rnd() & 1) ? 3 : 2) << 8);
            int depth = (int)(rnd() % 9), topv = (int)(rnd() & 7);
            u16 tag = 0xffff;
            for (int k = 0; k < depth; k++) tag &= (u16)~(3u << (2 * ((topv + k) & 7)));
            memcpy(fpu_in, &cw, 2);
            u16 sw = (u16)(topv << 11);
            memcpy(fpu_in + 4, &sw, 2);
            memcpy(fpu_in + 8, &tag, 2);
            for (int k = 0; k < 8; k++) rand_f80(fpu_in + 28 + 10 * k);
        }
        St in;
        for (int i = 0; i < 8; i++) {
            u32 v = rnd();
            switch (rnd() & 7) { case 0: v = 0; break; case 1: v = 0xffffffffu; break; case 2: v = 0x80000000u; break; case 3: v &= 0xff; break; default: break; }
            in.r[i] = v;
        }
        in.r[4] = 0;
        in.fl = (rnd() & ALL) | 0x202;
        if (g.skip_regs == 1) {   /* DIV/IDIV: EDX:EAX / EBX without overflow */
            in.r[3] = rnd() | 1;
            if (g.b[0] == 0x66 || g.b[1] == 0x66) in.r[2] = in.r[2] % (in.r[3] & 0xffff ? in.r[3] & 0xffff : 1);
            in.r[2] = 0; in.r[0] &= 0x7fffffff;
        }
        if (g.skip_regs == 3) { int p = g.b[0] == 0x66; int src = g.b[p + 2] & 7; if (!in.r[src] || (p && !(in.r[src] & 0xffff))) in.r[src] |= 1u << (rnd() & 15); }
        if (g_mem) {
            for (int i = 0; i < BUFSZ; i++) init_buf[i] = (u8)rnd();
            if (g_fpu) for (int k = 0; k < 8; k++) rand_f80(init_buf + 256 + 16 * k);
            memcpy(nbuf, init_buf, BUFSZ);
            in.r[3] = (u32)(uintptr_t)nbuf + 256 + (rnd() & 255);
            in.r[6] = rnd() & 31;
            if (g_sse) in.r[3] = (u32)(uintptr_t)nbuf + 256 + 16 * (rnd() % 16);
            if (g.skip_regs == 4) in.r[1] = (rnd() & 2047) - 1024;          /* BT bit offset: +-1024 bits */
            if (g_string) {
                in.r[6] = (u32)(uintptr_t)nbuf + 1024 + (rnd() & 511);
                in.r[7] = (u32)(uintptr_t)nbuf + 2560 + (rnd() & 511);
                in.r[1] = rnd() & 63;
            }
        }
        nat_in = in;
        build_native(g.b, g.len);
        ((void (*)(void))nat)();
        St emu;
        if (run_emu(g.b, g.len, &in, &emu)) {
            fails++;
            fprintf(stderr, "EMU FAILED to execute %s:", g.name);
            for (int i = 0; i < g.len; i++) fprintf(stderr, " %02x", g.b[i]);
            fprintf(stderr, "\n");
            continue;
        }
        u32 def = g.defined == 0x1000 ? shift_defined(&g, &in) : g.defined;
        if (def == 0x2000) continue;
        int bad = 0;
        if (g_mem && memcmp(nbuf, ram + GUEST_BUF, BUFSZ)) bad = 2;
        if (g_fpu) {
            /* control, status (incl. TOP and C0-C3), tag and the 8 registers; ignore pointer fields */
            u16 a, b2;
            memcpy(&a, fpu_out, 2); memcpy(&b2, fpu_out_emu, 2); if (a != b2) bad = 3;
            memcpy(&a, fpu_out + 4, 2); memcpy(&b2, fpu_out_emu + 4, 2); if ((a & 0x7fff) != (b2 & 0x7fff)) bad = 3;
            memcpy(&a, fpu_out + 8, 2); memcpy(&b2, fpu_out_emu + 8, 2); if (a != b2) bad = 3;
            if (memcmp(fpu_out + 28, fpu_out_emu + 28, 80)) bad = 3;
        }
        if (g_sse) {
            u32 m1, m2; memcpy(&m1, sse_out + 24, 4); memcpy(&m2, sse_out_emu + 24, 4);
            if (memcmp(sse_out + 160, sse_out_emu + 160, 128) || (m1 & ~0x3fu) != (m2 & ~0x3fu)) bad = 4;
            for (int k = 0; k < 8; k++) if (memcmp(sse_out + 32 + 16 * k, sse_out_emu + 32 + 16 * k, 10)) bad = 4;
            if (sse_out[4] != sse_out_emu[4] || ((sse_out[2] ^ sse_out_emu[2]) & 0x38) != 0) bad = 4;   /* FTW, TOP */
            else if ((m1 ^ m2) & 0x3fu) sse_flag_diffs++;
        }
        for (int i = 0; i < 8; i++) if (i != 4 && nat_out.r[i] != emu.r[i]) bad = 1;
        if ((nat_out.fl ^ emu.fl) & def) bad = 1;
        done++;
        if (bad) {
            fails++;
            per_fail[0]++;
            fprintf(stderr, "MISMATCH %-16s bytes:", g.name);
            for (int i = 0; i < g.len; i++) fprintf(stderr, " %02x", g.b[i]);
            fprintf(stderr, "\n  in : fl=%03x", in.fl & ALL);
            for (int i = 0; i < 8; i++) if (i != 4) fprintf(stderr, " r%d=%08x", i, in.r[i]);
            fprintf(stderr, "\n  cpu: fl=%03x", nat_out.fl & ALL);
            for (int i = 0; i < 8; i++) if (i != 4) fprintf(stderr, " r%d=%08x", i, nat_out.r[i]);
            fprintf(stderr, "\n  emu: fl=%03x", emu.fl & ALL);
            for (int i = 0; i < 8; i++) if (i != 4) fprintf(stderr, " r%d=%08x", i, emu.r[i]);
            fprintf(stderr, "   (defined flags %03x)%s\n", def, bad == 2 ? "  MEMORY DIFFERS" : bad == 3 ? "  FPU DIFFERS" : bad == 4 ? "  SSE DIFFERS" : "");
            if (bad == 4) {
                u32 m1, m2; memcpy(&m1, sse_in + 24, 4); fprintf(stderr, "  mxcsr in %08x", m1);
                memcpy(&m1, sse_out + 24, 4); memcpy(&m2, sse_out_emu + 24, 4); fprintf(stderr, " cpu %08x emu %08x\n", m1, m2);
                for (int k = 0; k < 8; k++) {
                    const u32 *a = (const u32 *)(const void *)(sse_out + 160 + 16 * k), *e = (const u32 *)(const void *)(sse_out_emu + 160 + 16 * k);
                    const u32 *iv = (const u32 *)(const void *)(sse_in + 160 + 16 * k);
                    if (memcmp(a, e, 16) || k < 2)
                        fprintf(stderr, "  xmm%d in %08x_%08x_%08x_%08x cpu %08x_%08x_%08x_%08x emu %08x_%08x_%08x_%08x\n", k,
                                iv[3], iv[2], iv[1], iv[0], a[3], a[2], a[1], a[0], e[3], e[2], e[1], e[0]);
                }
            }
            if (bad == 3) {
                u16 w[6]; memcpy(w, fpu_out, 2); memcpy(w + 1, fpu_out + 4, 2); memcpy(w + 2, fpu_out + 8, 2);
                memcpy(w + 3, fpu_out_emu, 2); memcpy(w + 4, fpu_out_emu + 4, 2); memcpy(w + 5, fpu_out_emu + 8, 2);
                fprintf(stderr, "  fpu cpu: cw %04x sw %04x tag %04x | emu: cw %04x sw %04x tag %04x\n", w[0], w[1], w[2], w[3], w[4], w[5]);
                for (int k = 0; k < 8; k++) if (memcmp(fpu_out + 28 + 10 * k, fpu_out_emu + 28 + 10 * k, 10)) {
                    fprintf(stderr, "  st%d cpu:", k); for (int j = 9; j >= 0; j--) fprintf(stderr, "%02x", fpu_out[28 + 10 * k + j]);
                    fprintf(stderr, " emu:"); for (int j = 9; j >= 0; j--) fprintf(stderr, "%02x", fpu_out_emu[28 + 10 * k + j]);
                    fprintf(stderr, "\n"); }
            }
        }
    }
    printf("%ld instructions compared, %ld mismatches (%ld SSE results differ only in MXCSR exception flags)\n", done, fails, sse_flag_diffs);
    return fails ? 1 : 0;
}
