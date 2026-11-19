/* Internal interfaces shared by the integer core (cpu.c) and SSE (sse.c). */
#ifndef W32_CPU_INT_H
#define W32_CPU_INT_H
#include "emu.h"

enum { EXC_DE = 0, EXC_DB = 1, EXC_BP = 3, EXC_UD = 6, EXC_NM = 7, EXC_DF = 8,
       EXC_TS = 10, EXC_NP = 11, EXC_SS = 12, EXC_GP = 13, EXC_PF = 14, EXC_MF = 16,
       EXC_AC = 17, EXC_XM = 19 };

typedef struct {
    u64 ip;             /* offset of the next byte to fetch (CS-relative) */
    u64 start;
    int osz, asz;       /* operand size 2/4/8, address size 4/8 */
    int rex, rexw, rexr, rexx, rexb;
    int seg;            /* override, -1 = none */
    int rep;            /* 0, 0xF2, 0xF3 */
    int p66;
    int mod, reg, rm;   /* reg includes REX.R; rm includes REX.B */
    int ea_seg, rip_rel;
    int opc;            /* opcode: byte, 0x100|0F xx, 0x200|0F38 xx, 0x300|0F3A xx */
    u64 ea;
} Dec;

static const u64 MASK[9] = { 0, 0xffull, 0xffffull, 0, 0xffffffffull, 0, 0, 0, ~0ull };
#define SIGNBIT(sz) (1ull << ((sz) * 8 - 1))

__attribute__((noreturn)) void cpu_exc(Cpu *c, int vec, int has_err, u32 err);
#define UD(c) cpu_exc((c), EXC_UD, 0, 0)
#define GP0(c) cpu_exc((c), EXC_GP, 1, 0)

u64 lread(Cpu *c, u64 la, int sz, int acc);
void lwrite(Cpu *c, u64 la, u64 v, int sz);
void fetch_page(Cpu *c, u64 la);
u8 *host_span(Cpu *c, int seg, u64 off, u32 len, int write);
void vread128(Cpu *c, int seg, u64 off, Xmm *x, int aligned);
void vwrite128(Cpu *c, int seg, u64 off, const Xmm *x, int aligned);

void decode_modrm(Cpu *c, Dec *d);
void set_flags_raw(Cpu *c, u64 arith);    /* replace the six arithmetic flags */
u64 get_flags(Cpu *c);
int sse_exec(Cpu *c, Dec *d, int op);
int sse_int_binop(int op, Xmm *x, const Xmm *s);
int sse4_exec(Cpu *c, Dec *d, int map);
void x87_exec(Cpu *c, Dec *d, u8 b);
void x87_reset(Cpu *c);
u8 x87_abridged_tag(Cpu *c);
void x87_set_abridged_tag(Cpu *c, u8 t);     /* 0F xx; returns 0 if not an SSE opcode */
void fx_save(Cpu *c, int seg, u64 off);
void fx_restore(Cpu *c, int seg, u64 off);

/* ---- memory and fetch fast paths (TLB hit), slow paths in cpu.c ---- */

static inline u64 seg_lin(Cpu *c, int seg, u64 off) {
    if (c->mode64) return seg >= FS ? off + c->seg[seg].base : off;
    return (u32)(off + c->seg[seg].base);
}

static inline u64 ld_mem(const u8 *p, int sz) {
    switch (sz) {
    case 1: return *p;
    case 2: { u16 v; __builtin_memcpy(&v, p, 2); return v; }
    case 4: { u32 v; __builtin_memcpy(&v, p, 4); return v; }
    default: { u64 v; __builtin_memcpy(&v, p, 8); return v; }
    }
}

static inline void st_mem(u8 *p, u64 v, int sz) {
    switch (sz) {
    case 1: *p = (u8)v; break;
    case 2: { u16 x = (u16)v; __builtin_memcpy(p, &x, 2); break; }
    case 4: { u32 x = (u32)v; __builtin_memcpy(p, &x, 4); break; }
    default: __builtin_memcpy(p, &v, 8); break;
    }
}

static inline u64 vread(Cpu *c, int seg, u64 off, int sz) {
    u64 la = seg_lin(c, seg, off);
    const TlbEntry *e = &c->tlb_r[(u32)(la >> 12) & (TLB_SIZE - 1)];
    if (__builtin_expect(e->tag == ((la & ~0xfffull) | c->tlb_mode) && (u32)(la & 0xfff) <= 0x1000u - (u32)sz, 1))
        return ld_mem(e->host + (la & 0xfff), sz);
    return lread(c, la, sz, 0);
}

static inline void vwrite(Cpu *c, int seg, u64 off, u64 v, int sz) {
    u64 la = seg_lin(c, seg, off);
    const TlbEntry *e = &c->tlb_w[(u32)(la >> 12) & (TLB_SIZE - 1)];
    if (__builtin_expect(e->tag == ((la & ~0xfffull) | c->tlb_mode) && (u32)(la & 0xfff) <= 0x1000u - (u32)sz, 1)) {
        st_mem(e->host + (la & 0xfff), v, sz);
        return;
    }
    lwrite(c, la, v, sz);
}

static inline u8 fetch8(Cpu *c, Dec *d) {
    u64 la = c->mode64 ? d->ip : (u32)(d->ip + c->seg[CS].base);
    if (__builtin_expect((la & ~0xfffull) != c->fetch_la, 0)) fetch_page(c, la);
    d->ip++;
    return c->fetch_host[la & 0xfff];
}

static inline u16 fetch16(Cpu *c, Dec *d) {
    u64 la = c->mode64 ? d->ip : (u32)(d->ip + c->seg[CS].base);
    if (__builtin_expect((la & ~0xfffull) == c->fetch_la && (la & 0xfff) <= 0xffe, 1)) {
        d->ip += 2;
        return (u16)ld_mem(c->fetch_host + (la & 0xfff), 2);
    }
    u16 v = fetch8(c, d);
    return (u16)(v | (u16)fetch8(c, d) << 8);
}

static inline u32 fetch32(Cpu *c, Dec *d) {
    u64 la = c->mode64 ? d->ip : (u32)(d->ip + c->seg[CS].base);
    if (__builtin_expect((la & ~0xfffull) == c->fetch_la && (la & 0xfff) <= 0xffc, 1)) {
        d->ip += 4;
        return (u32)ld_mem(c->fetch_host + (la & 0xfff), 4);
    }
    u32 v = fetch16(c, d);
    return v | (u32)fetch16(c, d) << 16;
}

static inline u64 ea_off(const Dec *d) {
    u64 a = d->rip_rel ? d->ea + d->ip : d->ea;
    return d->asz == 4 ? (u32)a : d->asz == 2 ? (a & 0xffff) : a;
}

static inline u64 get_reg(Cpu *c, const Dec *d, int n, int sz) {
    switch (sz) {
    case 1: if (!d->rex && n >= 4 && n < 8) return (c->r[n - 4] >> 8) & 0xff;
            return c->r[n] & 0xff;
    case 2: return c->r[n] & 0xffff;
    case 4: return (u32)c->r[n];
    default: return c->r[n];
    }
}

static inline void set_reg(Cpu *c, const Dec *d, int n, int sz, u64 v) {
    switch (sz) {
    case 1:
        if (!d->rex && n >= 4 && n < 8) c->r[n - 4] = (c->r[n - 4] & ~0xff00ull) | ((v & 0xff) << 8);
        else c->r[n] = (c->r[n] & ~0xffull) | (v & 0xff);
        break;
    case 2: c->r[n] = (c->r[n] & ~0xffffull) | (v & 0xffff); break;
    case 4: c->r[n] = (u32)v; break;
    default: c->r[n] = v; break;
    }
}

static inline u64 rd_rm(Cpu *c, Dec *d, int sz) {
    if (d->mod == 3) return get_reg(c, d, d->rm, sz);
    return vread(c, d->ea_seg, ea_off(d), sz);
}

static inline void wr_rm(Cpu *c, Dec *d, int sz, u64 v) {
    if (d->mod == 3) set_reg(c, d, d->rm, sz, v);
    else vwrite(c, d->ea_seg, ea_off(d), v, sz);
}

static inline u64 sext(u64 v, int sz) {
    switch (sz) {
    case 1: return (u64)(i64)(i8)v;
    case 2: return (u64)(i64)(i16)v;
    case 4: return (u64)(i64)(i32)v;
    default: return v;
    }
}
#endif
