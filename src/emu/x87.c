/* x87 FPU. Registers are kept as raw 80-bit values; every arithmetic operation
 * is carried out by the host's own x87 (all W32Eleven hosts are x86) under the
 * guest's control word, so results, rounding and condition codes are exact.
 * Only instructions present on a 387/486 are used on the host side.
 * Exceptions are always taken as masked (status bits are recorded; #MF is not
 * raised). */
#include "cpu_int.h"

typedef struct { u8 b[10]; } F80;

#define FSW_C0 0x0100
#define FSW_C1 0x0200
#define FSW_C2 0x0400
#define FSW_C3 0x4000
#define FSW_CC (FSW_C0 | FSW_C1 | FSW_C2 | FSW_C3)
#define FSW_TOP(c) (((c)->fsw >> 11) & 7)

static const F80 F_INDEF = { { 0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xff } };   /* QNaN indefinite */

/* ---------- register stack ---------- */

static int top(Cpu *c) { return FSW_TOP(c); }
static void set_top(Cpu *c, int t) { c->fsw = (u16)((c->fsw & ~0x3800u) | ((u32)(t & 7) << 11)); }
static int phys(Cpu *c, int i) { return (top(c) + i) & 7; }
static int empty(Cpu *c, int i) { return ((c->ftw >> (2 * phys(c, i))) & 3) == 3; }
static void tag_valid(Cpu *c, int p) { c->ftw = (u16)(c->ftw & ~(3u << (2 * p))); }
static void tag_empty(Cpu *c, int p) { c->ftw = (u16)(c->ftw | (3u << (2 * p))); }

static int g_uf;   /* this instruction read an empty register: results become QNaN indefinite */

static F80 st(Cpu *c, int i) {
    F80 v;
    if (empty(c, i)) { c->fsw |= 0x41; g_uf = 1; return F_INDEF; }   /* stack underflow: IE|SF */
    e_memcpy(v.b, c->fpr[phys(c, i)], 10);
    return v;
}
static void set_st(Cpu *c, int i, F80 v) {
    int p = phys(c, i);
    if (g_uf) v = F_INDEF;
    e_memcpy(c->fpr[p], v.b, 10);
    tag_valid(c, p);
}
static void push(Cpu *c, F80 v) {
    int t = (top(c) - 1) & 7;
    if (((c->ftw >> (2 * t)) & 3) != 3) { c->fsw |= 0x41 | FSW_C1; v = F_INDEF; }   /* overflow */
    set_top(c, t);
    set_st(c, 0, v);
}
static int full(Cpu *c) { return ((c->ftw >> (2 * ((top(c) - 1) & 7))) & 3) != 3; }
/* load-type push: on overflow the operand is not examined */
#define PUSH_LOAD(expr) do { if (full(c)) push(c, F_INDEF); else push(c, (expr)); } while (0)

static void pop(Cpu *c) { tag_empty(c, phys(c, 0)); set_top(c, top(c) + 1); }

/* ---------- host x87 execution ---------- */

static u16 g_host_cw;
static u16 g_opsw = 0xffff;   /* status captured right after the arithmetic instruction (0xffff: none) */
#if defined(__x86_64__)
#define OPSW "g_opsw(%%rip)"
#else
#define OPSW "g_opsw"
#endif

static void host_enter(Cpu *c) {
    u16 cw = (u16)(c->fcw | 0x3f);              /* exceptions masked on the host */
    __asm__ volatile("fnstcw %0\n\tfnclex\n\tfldcw %1" : "=m"(g_host_cw) : "m"(cw));
    g_opsw = 0xffff;
}
static u16 host_leave(Cpu *c) {
    u16 sw;
    __asm__ volatile("fnstsw %0\n\tfldcw %1" : "=m"(sw) : "m"(g_host_cw));
    c->fsw |= sw & 0x3f;                        /* sticky exception flags */
    u16 cc = g_opsw != 0xffff ? g_opsw : sw;
    c->fsw = (u16)((c->fsw & ~FSW_C1) | (cc & FSW_C1));   /* C1: rounded up */
    if (g_opsw != 0xffff) sw = (u16)((sw & ~FSW_CC) | (g_opsw & FSW_CC));
    if (c->fsw & 0x3f & ~c->fcw) c->fsw |= 0x80; /* ES (summary) for unmasked ones */
    return sw;
}

enum { OP_ADD, OP_MUL, OP_COM, OP_COMP, OP_SUB, OP_SUBR, OP_DIV, OP_DIVR };

/* r = a op b for the eight D8-style operations (compares return status) */
static F80 arith(Cpu *c, int op, F80 a, F80 b, u16 *sw) {
    F80 r = a;
    host_enter(c);
    switch (op) {
    case OP_ADD: __asm__ volatile("fldt %2\n\tfldt %1\n\tfadd %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case OP_MUL: __asm__ volatile("fldt %2\n\tfldt %1\n\tfmul %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case OP_SUB: __asm__ volatile("fldt %2\n\tfldt %1\n\tfsub %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case OP_SUBR: __asm__ volatile("fldt %2\n\tfldt %1\n\tfsubr %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case OP_DIV: __asm__ volatile("fldt %2\n\tfldt %1\n\tfdiv %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case OP_DIVR: __asm__ volatile("fldt %2\n\tfldt %1\n\tfdivr %%st(1), %%st\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    default: __asm__ volatile("fldt %1\n\tfldt %0\n\tfcompp" : : "m"(a), "m"(b)); break;     /* FCOM: compare a with b */
    }
    u16 s = host_leave(c);
    if (sw) *sw = s;
    return r;
}

/* ST0 op m (kind 0 m32fp, 1 m64fp, 2 m32int, 3 m16int) with the host's own memory form */
static F80 arith_mem(Cpu *c, int op, F80 a, const void *m, int kind, u16 *sw) {
    F80 r = a;
    host_enter(c);
    switch (kind * 8 + op) {
    case 0: __asm__ volatile("fldt %1\n\tfadds %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 1: __asm__ volatile("fldt %1\n\tfmuls %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 2: __asm__ volatile("fldt %0\n\tfcoms %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 3: __asm__ volatile("fldt %0\n\tfcoms %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 4: __asm__ volatile("fldt %1\n\tfsubs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 5: __asm__ volatile("fldt %1\n\tfsubrs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 6: __asm__ volatile("fldt %1\n\tfdivs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 7: __asm__ volatile("fldt %1\n\tfdivrs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 8: __asm__ volatile("fldt %1\n\tfaddl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 9: __asm__ volatile("fldt %1\n\tfmull %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 10: __asm__ volatile("fldt %0\n\tfcoml %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 11: __asm__ volatile("fldt %0\n\tfcoml %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 12: __asm__ volatile("fldt %1\n\tfsubl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 13: __asm__ volatile("fldt %1\n\tfsubrl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 14: __asm__ volatile("fldt %1\n\tfdivl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 15: __asm__ volatile("fldt %1\n\tfdivrl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 16: __asm__ volatile("fldt %1\n\tfiaddl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 17: __asm__ volatile("fldt %1\n\tfimull %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 18: __asm__ volatile("fldt %0\n\tficoml %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 19: __asm__ volatile("fldt %0\n\tficoml %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 20: __asm__ volatile("fldt %1\n\tfisubl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 21: __asm__ volatile("fldt %1\n\tfisubrl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 22: __asm__ volatile("fldt %1\n\tfidivl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 23: __asm__ volatile("fldt %1\n\tfidivrl %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 24: __asm__ volatile("fldt %1\n\tfiadds %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 25: __asm__ volatile("fldt %1\n\tfimuls %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 26: __asm__ volatile("fldt %0\n\tficoms %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 27: __asm__ volatile("fldt %0\n\tficoms %1\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 28: __asm__ volatile("fldt %1\n\tfisubs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 29: __asm__ volatile("fldt %1\n\tfisubrs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 30: __asm__ volatile("fldt %1\n\tfidivs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    case 31: __asm__ volatile("fldt %1\n\tfidivrs %2\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(*(const u8 (*)[8])m)); break;
    }
    u16 s = host_leave(c);
    if (sw) *sw = s;
    return r;
}

static u16 ucompare(Cpu *c, F80 a, F80 b) {
    host_enter(c);
    __asm__ volatile("fldt %1\n\tfldt %0\n\tfucompp" : : "m"(a), "m"(b));
    return host_leave(c);
}

static void set_cc(Cpu *c, u16 sw, u16 mask) { c->fsw = (u16)((c->fsw & ~mask) | (sw & mask)); }

/* unary operations on ST0, results replace ST0 (and may push/pop) */
static void unary(Cpu *c, u8 code) {
    if (code == 0xe5) {                                   /* FXAM: never faults; C1 = sign of the register */
        F80 raw; e_memcpy(raw.b, c->fpr[phys(c, 0)], 10);
        if (empty(c, 0)) { c->fsw = (u16)((c->fsw & ~FSW_CC) | FSW_C0 | FSW_C3 | ((raw.b[9] & 0x80) ? FSW_C1 : 0)); return; }
    }
    if ((code == 0xf2 || code == 0xfb || code == 0xf4) && !empty(c, 0) && ((c->ftw >> (2 * ((top(c) - 1) & 7))) & 3) != 3) {
        set_st(c, 0, F_INDEF);                            /* stack overflow: both results indefinite */
        push(c, F_INDEF);
        return;
    }
    F80 a = st(c, 0), r = a, r2 = a;
    F80 b = (code == 0xf1 || code == 0xf3 || code == 0xf9 || code == 0xfd || code == 0xf5 || code == 0xf8) ? st(c, 1) : a;
    u16 sw;
    host_enter(c);
    switch (code) {
    case 0xe0: __asm__ volatile("fldt %1\n\tfchs\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xe1: __asm__ volatile("fldt %1\n\tfabs\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xf0: __asm__ volatile("fldt %1\n\tf2xm1\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xfa: __asm__ volatile("fldt %1\n\tfsqrt\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xfc: __asm__ volatile("fldt %1\n\tfrndint\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xfe: __asm__ volatile("fldt %1\n\tfsin\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xff: __asm__ volatile("fldt %1\n\tfcos\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a)); break;
    case 0xf1: __asm__ volatile("fldt %2\n\tfldt %1\n\tfyl2x\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xf9: __asm__ volatile("fldt %2\n\tfldt %1\n\tfyl2xp1\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xf3: __asm__ volatile("fldt %2\n\tfldt %1\n\tfpatan\n\tfnstsw " OPSW "\n\tfstpt %0" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xfd: __asm__ volatile("fldt %2\n\tfldt %1\n\tfscale\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xf8: __asm__ volatile("fldt %2\n\tfldt %1\n\tfprem\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xf5: __asm__ volatile("fldt %2\n\tfldt %1\n\tfprem1\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstp %%st(0)" : "=m"(r) : "m"(a), "m"(b)); break;
    case 0xf2: __asm__ volatile("fldt %2\n\tfptan\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstpt %1" : "=m"(r2), "=m"(r) : "m"(a)); break;   /* r = tan, r2 = 1.0 */
    case 0xfb: __asm__ volatile("fldt %2\n\tfsincos\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstpt %1" : "=m"(r2), "=m"(r) : "m"(a)); break; /* r = sin, r2 = cos */
    case 0xf4: __asm__ volatile("fldt %2\n\tfxtract\n\tfnstsw " OPSW "\n\tfstpt %0\n\tfstpt %1" : "=m"(r2), "=m"(r) : "m"(a)); break; /* r = exp, r2 = signif */
    case 0xe4: __asm__ volatile("fldt %0\n\tftst\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a)); break;
    case 0xe5: __asm__ volatile("fldt %0\n\tfxam\n\tfnstsw " OPSW "\n\tfstp %%st(0)" : : "m"(a)); break;
    }
    sw = host_leave(c);
    switch (code) {
    case 0xf1: case 0xf9: case 0xf3: pop(c); set_st(c, 0, r); set_cc(c, sw, FSW_C1); break;
    case 0xe4: set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); c->fsw &= ~FSW_C1; break;
    case 0xe5: set_cc(c, sw, FSW_CC); break;
    case 0xf8: case 0xf5: set_st(c, 0, r); set_cc(c, sw, FSW_CC); break;
    case 0xf2: case 0xfb:
        if (sw & FSW_C2) set_cc(c, sw, FSW_C2);         /* out of range: operand unchanged */
        else { set_st(c, 0, r); push(c, r2); c->fsw &= ~(FSW_C2); }
        break;
    case 0xf4: set_st(c, 0, r); push(c, r2); break;
    case 0xfe: case 0xff:
        if (sw & FSW_C2) set_cc(c, sw, FSW_C2); else { set_st(c, 0, r); c->fsw &= ~FSW_C2; }
        break;
    default: set_st(c, 0, r); set_cc(c, sw, FSW_C1); break;
    }
}

/* ---------- memory formats ---------- */

static F80 from_f32(Cpu *c, u32 v) { F80 r; host_enter(c); __asm__ volatile("flds %1\n\tfstpt %0" : "=m"(r) : "m"(v)); host_leave(c); return r; }
static F80 from_f64(Cpu *c, u64 v) { F80 r; host_enter(c); __asm__ volatile("fldl %1\n\tfstpt %0" : "=m"(r) : "m"(v)); host_leave(c); return r; }
static F80 from_i16(Cpu *c, u16 v) { F80 r; host_enter(c); __asm__ volatile("filds %1\n\tfstpt %0" : "=m"(r) : "m"(v)); host_leave(c); return r; }
static F80 from_i32(Cpu *c, u32 v) { F80 r; host_enter(c); __asm__ volatile("fildl %1\n\tfstpt %0" : "=m"(r) : "m"(v)); host_leave(c); return r; }
static F80 from_i64(Cpu *c, u64 v) { F80 r; host_enter(c); __asm__ volatile("fildll %1\n\tfstpt %0" : "=m"(r) : "m"(v)); host_leave(c); return r; }
static u32 to_f32(Cpu *c, F80 a) { u32 v; host_enter(c); __asm__ volatile("fldt %1\n\tfstps %0" : "=m"(v) : "m"(a)); host_leave(c); return v; }
static u64 to_f64(Cpu *c, F80 a) { u64 v; host_enter(c); __asm__ volatile("fldt %1\n\tfstpl %0" : "=m"(v) : "m"(a)); host_leave(c); return v; }

/* FIST/FISTP (current rounding) or FISTTP (truncate) */
static u64 to_int(Cpu *c, F80 a, int sz, int trunc) {
    u16 saved = c->fcw;
    if (trunc) c->fcw |= 0xc00;
    u64 v = 0;
    host_enter(c);
    if (sz == 2) { u16 x; __asm__ volatile("fldt %1\n\tfistps %0" : "=m"(x) : "m"(a)); v = x; }
    else if (sz == 4) { u32 x; __asm__ volatile("fldt %1\n\tfistpl %0" : "=m"(x) : "m"(a)); v = x; }
    else { u64 x; __asm__ volatile("fldt %1\n\tfistpll %0" : "=m"(x) : "m"(a)); v = x; }
    host_leave(c);
    c->fcw = saved;
    return v;
}

static F80 from_bcd(Cpu *c, const u8 *p) {
    F80 r, in;
    e_memcpy(in.b, p, 10);
    host_enter(c);
    __asm__ volatile("fbld %1\n\tfstpt %0" : "=m"(r) : "m"(in));
    host_leave(c);
    return r;
}

static void to_bcd(Cpu *c, F80 a, u8 *p) {
    F80 out;
    host_enter(c);
    __asm__ volatile("fldt %1\n\tfbstp %0" : "=m"(out) : "m"(a));
    host_leave(c);
    e_memcpy(p, out.b, 10);
}

/* ---------- environment ---------- */

static u16 full_tag(Cpu *c) {
    u16 t = 0;
    for (int p = 0; p < 8; p++) {
        u32 tag = (c->ftw >> (2 * p)) & 3;
        if (tag != 3) {
            const u8 *r = c->fpr[p];
            u32 exp = (u32)(r[9] & 0x7f) << 8 | r[8];
            int mant_zero = 1;
            for (int k = 0; k < 8; k++) if (r[k]) mant_zero = 0;
            tag = (exp == 0 && mant_zero) ? 1 : (exp == 0x7fff || exp == 0 || !(r[7] & 0x80)) ? 2 : 0;
        }
        t |= (u16)(tag << (2 * p));
    }
    return t;
}

static void store_env(Cpu *c, int seg, u64 off) {
    vwrite(c, seg, off, c->fcw | 0xffff0000u, 4);
    vwrite(c, seg, off + 4, c->fsw | 0xffff0000u, 4);
    vwrite(c, seg, off + 8, full_tag(c) | 0xffff0000u, 4);
    for (int i = 12; i < 28; i += 4) vwrite(c, seg, off + (u64)i, 0, 4);
}

static void load_env(Cpu *c, int seg, u64 off) {
    c->fcw = (u16)vread(c, seg, off, 2);
    c->fsw = (u16)vread(c, seg, off + 4, 2);
    u16 t = (u16)vread(c, seg, off + 8, 2);
    c->ftw = 0;
    for (int p = 0; p < 8; p++) if (((t >> (2 * p)) & 3) == 3) c->ftw |= (u16)(3u << (2 * p));
}

void x87_reset(Cpu *c) { c->fcw = 0x37f; c->fsw = 0; c->ftw = 0xffff; }

/* abridged tag (FXSAVE): 1 = valid */
u8 x87_abridged_tag(Cpu *c) {
    u8 t = 0;
    for (int p = 0; p < 8; p++) if (((c->ftw >> (2 * p)) & 3) != 3) t |= (u8)(1u << p);
    return t;
}
void x87_set_abridged_tag(Cpu *c, u8 t) {
    c->ftw = 0;
    for (int p = 0; p < 8; p++) if (!((t >> p) & 1)) c->ftw |= (u16)(3u << (2 * p));
}

/* ---------- dispatch (D8-DF) ---------- */

static void fcmov(Cpu *c, int cond, int i) {
    F80 v = st(c, i), a = st(c, 0);
    if (g_uf) { set_st(c, 0, a); c->fsw &= ~FSW_C1; return; }   /* underflow: ST0 = indefinite */
    if (cond) set_st(c, 0, v);
}

static void comi(Cpu *c, u16 sw) {
    u64 f = ((sw & FSW_C0) ? 1u : 0) | ((sw & FSW_C2) ? 4u : 0) | ((sw & FSW_C3) ? 0x40u : 0);
    set_flags_raw(c, f);
}

void x87_exec(Cpu *c, Dec *d, u8 b) {
    g_uf = 0;
    decode_modrm(c, d);
    int r = d->reg & 7, mem = d->mod != 3, i = d->rm & 7;
    int seg = d->ea_seg;
    u64 off = mem ? ea_off(d) : 0;
    u16 sw;
    if (mem) {
        switch (b) {
        case 0xd8: case 0xdc: case 0xda: case 0xde: {
            int kind = b == 0xd8 ? 0 : b == 0xdc ? 1 : b == 0xda ? 2 : 3;
            u64 mv = vread(c, seg, off, kind == 1 ? 8 : kind == 3 ? 2 : 4);
            if (empty(c, 0)) {                    /* underflow: nothing is computed */
                st(c, 0);
                if (r == OP_COM || r == OP_COMP) { c->fsw |= FSW_C0 | FSW_C2 | FSW_C3; c->fsw &= ~FSW_C1; if (r == OP_COMP) pop(c); }
                else { set_st(c, 0, F_INDEF); c->fsw &= ~FSW_C1; }
                return;
            }
            F80 a = st(c, 0);
            if (r == OP_COM || r == OP_COMP) {
                arith_mem(c, OP_COM, a, &mv, kind, &sw);
                set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); c->fsw &= ~FSW_C1;
                if (r == OP_COMP) pop(c);
            } else set_st(c, 0, arith_mem(c, r, a, &mv, kind, 0));
            return;
        }
        case 0xd9:
            switch (r) {
            case 0: { u32 v = (u32)vread(c, seg, off, 4); PUSH_LOAD(from_f32(c, v)); return; }
            case 2: vwrite(c, seg, off, to_f32(c, st(c, 0)), 4); return;
            case 3: vwrite(c, seg, off, to_f32(c, st(c, 0)), 4); pop(c); return;
            case 4: load_env(c, seg, off); return;
            case 5: c->fcw = (u16)vread(c, seg, off, 2); return;
            case 6: store_env(c, seg, off); c->fcw |= 0x3f; return;
            case 7: vwrite(c, seg, off, c->fcw, 2); return;
            }
            break;
        case 0xdb:
            switch (r) {
            case 0: { u32 v = (u32)vread(c, seg, off, 4); PUSH_LOAD(from_i32(c, v)); return; }
            case 1: vwrite(c, seg, off, to_int(c, st(c, 0), 4, 1), 4); pop(c); return;
            case 2: vwrite(c, seg, off, to_int(c, st(c, 0), 4, 0), 4); return;
            case 3: vwrite(c, seg, off, to_int(c, st(c, 0), 4, 0), 4); pop(c); return;
            case 5: { F80 v; u64 lo = vread(c, seg, off, 8); u16 hi = (u16)vread(c, seg, off + 8, 2);
                      e_memcpy(v.b, &lo, 8); e_memcpy(v.b + 8, &hi, 2); push(c, v); return; }
            case 7: { F80 v = st(c, 0); u64 lo; u16 hi; e_memcpy(&lo, v.b, 8); e_memcpy(&hi, v.b + 8, 2);
                      vwrite(c, seg, off, lo, 8); vwrite(c, seg, off + 8, hi, 2); pop(c); return; }
            }
            break;
        case 0xdd:
            switch (r) {
            case 0: { u64 v = vread(c, seg, off, 8); PUSH_LOAD(from_f64(c, v)); return; }
            case 1: vwrite(c, seg, off, to_int(c, st(c, 0), 8, 1), 8); pop(c); return;
            case 2: vwrite(c, seg, off, to_f64(c, st(c, 0)), 8); return;
            case 3: vwrite(c, seg, off, to_f64(c, st(c, 0)), 8); pop(c); return;
            case 4: {   /* FRSTOR */
                load_env(c, seg, off);
                for (int k = 0; k < 8; k++) {
                    int p = (top(c) + k) & 7;
                    for (int j = 0; j < 10; j++) c->fpr[p][j] = (u8)vread(c, seg, off + 28 + 10 * (u64)k + (u64)j, 1);
                }
                return;
            }
            case 6: {   /* FNSAVE */
                store_env(c, seg, off);
                for (int k = 0; k < 8; k++) {
                    int p = (top(c) + k) & 7;
                    for (int j = 0; j < 10; j++) vwrite(c, seg, off + 28 + 10 * (u64)k + (u64)j, c->fpr[p][j], 1);
                }
                x87_reset(c);
                return;
            }
            case 7: vwrite(c, seg, off, c->fsw, 2); return;
            }
            break;
        case 0xdf:
            switch (r) {
            case 0: { u16 v = (u16)vread(c, seg, off, 2); PUSH_LOAD(from_i16(c, v)); return; }
            case 1: vwrite(c, seg, off, to_int(c, st(c, 0), 2, 1), 2); pop(c); return;
            case 2: vwrite(c, seg, off, to_int(c, st(c, 0), 2, 0), 2); return;
            case 3: vwrite(c, seg, off, to_int(c, st(c, 0), 2, 0), 2); pop(c); return;
            case 4: { u8 p[10]; for (int j = 0; j < 10; j++) p[j] = (u8)vread(c, seg, off + (u64)j, 1); PUSH_LOAD(from_bcd(c, p)); return; }
            case 5: { u64 v = vread(c, seg, off, 8); PUSH_LOAD(from_i64(c, v)); return; }
            case 6: { u8 p[10]; to_bcd(c, st(c, 0), p); for (int j = 0; j < 10; j++) vwrite(c, seg, off + (u64)j, p[j], 1); pop(c); return; }
            case 7: vwrite(c, seg, off, to_int(c, st(c, 0), 8, 0), 8); pop(c); return;
            }
            break;
        }
        UD(c);
    }
    /* register forms */
    switch (b) {
    case 0xd8: {
        F80 a = st(c, 0), s = st(c, i);
        if (r == OP_COM || r == OP_COMP) {
            arith(c, OP_COM, a, s, &sw);
            set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); c->fsw &= ~FSW_C1;
            if (r == OP_COMP) pop(c);
        } else set_st(c, 0, arith(c, r, a, s, 0));
        return;
    }
    case 0xdc: case 0xde: {
        /* ST(i) = ST(i) op ST(0); sub/div forms are swapped relative to D8 */
        static const int map[8] = { OP_ADD, OP_MUL, OP_COM, OP_COMP, OP_SUBR, OP_SUB, OP_DIVR, OP_DIV };
        if (b == 0xde && r == 3) {
            if (i != 1) UD(c);
            arith(c, OP_COM, st(c, 0), st(c, 1), &sw);                        /* FCOMPP */
            set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); c->fsw &= ~FSW_C1;
            pop(c); pop(c);
            return;
        }
        if (r == 2 || r == 3) {                                                /* aliases of FCOM/FCOMP */
            arith(c, OP_COM, st(c, 0), st(c, i), &sw);
            set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3);
            if (r == 3) pop(c);
            return;
        }
        set_st(c, i, arith(c, map[r], st(c, i), st(c, 0), 0));
        if (b == 0xde) pop(c);
        return;
    }
    case 0xd9:
        switch (r) {
        case 0: push(c, st(c, i)); return;
        case 1: { F80 a = st(c, 0), s = st(c, i); g_uf = 0; set_st(c, 0, s); set_st(c, i, a); c->fsw &= ~FSW_C1; return; }
        case 2: if (i == 0) return; break;                                     /* FNOP */
        case 3: set_st(c, i, st(c, 0)); pop(c); return;                        /* FSTP1 alias */
        case 4: if (i == 0 || i == 1 || i == 4 || i == 5) { unary(c, (u8)(0xe0 | i)); return; } break;
        case 5: {
            static const u8 k[7][10] = {
                { 0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f },                      /* 1 */
                { 0xfe, 0x8a, 0x1b, 0xcd, 0x4b, 0x78, 0x9a, 0xd4, 0x00, 0x40 },  /* log2(10) */
                { 0xbc, 0xf0, 0x17, 0x5c, 0x29, 0x3b, 0xaa, 0xb8, 0xff, 0x3f },  /* log2(e) */
                { 0x35, 0xc2, 0x68, 0x21, 0xa2, 0xda, 0x0f, 0xc9, 0x00, 0x40 },  /* pi */
                { 0x99, 0xf7, 0xcf, 0xfb, 0x84, 0x9a, 0x20, 0x9a, 0xfd, 0x3f },  /* log10(2) */
                { 0xac, 0x79, 0xcf, 0xd1, 0xf7, 0x17, 0x72, 0xb1, 0xfe, 0x3f },  /* ln(2) */
                { 0 } };                                                        /* 0 */
            if (i == 7) break;
            F80 v;
            if (i == 0) {   /* host constants honour rounding control; compute on the host */
                e_memcpy(v.b, k[0], 10);
            } else {
                host_enter(c);
                switch (i) {
                case 1: __asm__ volatile("fldl2t\n\tfstpt %0" : "=m"(v)); break;
                case 2: __asm__ volatile("fldl2e\n\tfstpt %0" : "=m"(v)); break;
                case 3: __asm__ volatile("fldpi\n\tfstpt %0" : "=m"(v)); break;
                case 4: __asm__ volatile("fldlg2\n\tfstpt %0" : "=m"(v)); break;
                case 5: __asm__ volatile("fldln2\n\tfstpt %0" : "=m"(v)); break;
                default: e_memcpy(v.b, k[6], 10); break;
                }
                host_leave(c);
            }
            push(c, v);
            return;
        }
        case 6:
            if (i == 6) { set_top(c, top(c) - 1); c->fsw &= ~FSW_C1; return; }   /* FDECSTP */
            if (i == 7) { set_top(c, top(c) + 1); c->fsw &= ~FSW_C1; return; }   /* FINCSTP */
            unary(c, (u8)(0xf0 | i));
            return;
        case 7: unary(c, (u8)(0xf8 | i)); return;
        }
        break;
    case 0xda: {
        u64 f = get_flags(c);
        switch (r) {
        case 0: fcmov(c, (f & F_CF) != 0, i); return;
        case 1: fcmov(c, (f & F_ZF) != 0, i); return;
        case 2: fcmov(c, (f & (F_CF | F_ZF)) != 0, i); return;
        case 3: fcmov(c, (f & F_PF) != 0, i); return;
        case 5:
            if (i == 1) { sw = ucompare(c, st(c, 0), st(c, 1)); set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); pop(c); pop(c); return; }
            break;
        }
        break;
    }
    case 0xdb: {
        u64 f = get_flags(c);
        switch (r) {
        case 0: fcmov(c, !(f & F_CF), i); return;
        case 1: fcmov(c, !(f & F_ZF), i); return;
        case 2: fcmov(c, !(f & (F_CF | F_ZF)), i); return;
        case 3: fcmov(c, !(f & F_PF), i); return;
        case 4:
            if (i == 2) { c->fsw &= 0x7f00; return; }                          /* FNCLEX */
            if (i == 3) { x87_reset(c); return; }                              /* FNINIT */
            if (i == 0 || i == 1 || i == 4) return;                            /* FENI/FDISI/FSETPM: no-ops */
            break;
        case 5: comi(c, ucompare(c, st(c, 0), st(c, i))); return;             /* FUCOMI */
        case 6: arith(c, OP_COM, st(c, 0), st(c, i), &sw); comi(c, sw); return; /* FCOMI */
        }
        break;
    }
    case 0xdd:
        switch (r) {
        case 0: tag_empty(c, phys(c, i)); return;                              /* FFREE */
        case 1: { F80 a = st(c, 0), s = st(c, i); set_st(c, 0, s); set_st(c, i, a); return; }
        case 2: set_st(c, i, st(c, 0)); return;                                /* FST ST(i) */
        case 3: set_st(c, i, st(c, 0)); pop(c); return;                        /* FSTP ST(i) */
        case 4: case 5:
            sw = ucompare(c, st(c, 0), st(c, i));
            set_cc(c, sw, FSW_C0 | FSW_C2 | FSW_C3); c->fsw &= ~FSW_C1;
            if (r == 5) pop(c);
            return;
        }
        break;
    case 0xdf:
        switch (r) {
        case 0: tag_empty(c, phys(c, i)); pop(c); return;                      /* FFREEP */
        case 1: { F80 a = st(c, 0), s = st(c, i); set_st(c, 0, s); set_st(c, i, a); return; }
        case 2: case 3: set_st(c, i, st(c, 0)); pop(c); return;
        case 4: if (i == 0) { c->r[0] = (c->r[0] & ~0xffffull) | c->fsw; return; } break;   /* FNSTSW AX */
        case 5: comi(c, ucompare(c, st(c, 0), st(c, i))); pop(c); return;     /* FUCOMIP */
        case 6: arith(c, OP_COM, st(c, 0), st(c, i), &sw); comi(c, sw); pop(c); return;   /* FCOMIP */
        }
        break;
    }
    emu_log(c->m, "x87 opcode %02x /%d mod %d rm %d at %llx is not implemented", b, r, d->mod, i, (unsigned long long)d->start);
    UD(c);
}
