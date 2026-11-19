/* SSE / SSE2 (and FXSAVE) execution.
 *
 * Floating point: operands and results are handled as raw bit patterns (NaN
 * propagation, signalling NaNs, signed zeros, DAZ/FTZ follow SSE rules); the
 * arithmetic runs on the host FPU under MXCSR.RC and, on x87 hosts, at the
 * precision of the SSE type so every result is rounded once. MXCSR exception
 * flags are not modelled.
 */
#include "cpu_int.h"

#define F_CF_ 1u
#define F_PF_ 4u
#define F_ZF_ 0x40u

static int is_nan_d(double x) { return x != x; }

/* round to integer per MXCSR.RC (0 nearest-even, 1 down, 2 up, 3 truncate) */
static double round_mode(double x, int rc) {
    if (is_nan_d(x)) return x;
    double ax = x < 0 ? -x : x;
    if (ax >= 4503599627370496.0) return x;      /* 2^52: already integral */
    i64 t = (i64)x;
    double fr = x - (double)t;
    switch (rc) {
    case 0:
        if (fr > 0.5 || (fr == 0.5 && (t & 1))) t++;
        else if (fr < -0.5 || (fr == -0.5 && (t & 1))) t--;
        break;
    case 1: if (fr < 0) t--; break;
    case 2: if (fr > 0) t++; break;
    default: break;
    }
    double r = (double)t;
    if (r == 0) { u64 xb; __builtin_memcpy(&xb, &x, 8); if (xb >> 63) r = -0.0; }
    return r;
}

static u64 cvt_d2i(Cpu *c, double x, int sz, int trunc) {
    if (!trunc) x = round_mode(x, (c->mxcsr >> 13) & 3);
    if (sz == 8) {
        if (is_nan_d(x) || x >= 9223372036854775808.0 || x < -9223372036854775808.0) return 0x8000000000000000ull;
        return (u64)(i64)x;
    }
    if (is_nan_d(x) || x >= 2147483648.0 || x < -2147483648.0) return 0x80000000u;
    return (u32)(i32)x;
}

#if defined(__i386__)
/* i386 hosts: x87 FSQRT (precision control is extended, so a double result can
 * in rare cases differ in the last bit from SSE2's correctly rounded SQRTSD). */
static double sqrt_d(double x) { double r; __asm__("fsqrt" : "=t"(r) : "0"(x)); return r; }
static float sqrt_f(float x) { return (float)sqrt_d(x); }
#else
static float sqrt_f(float x) { return __builtin_sqrtf(x); }
static double sqrt_d(double x) { return __builtin_sqrt(x); }
#endif

static int cmp_pred_d(double a, double b, int p) {
    int un = is_nan_d(a) || is_nan_d(b);
    switch (p & 7) {
    case 0: return !un && a == b;
    case 1: return !un && a < b;
    case 2: return !un && a <= b;
    case 3: return un;
    case 4: return un || a != b;
    case 5: return un || !(a < b);
    case 6: return un || !(a <= b);
    default: return !un;
    }
}

static void get_src(Cpu *c, Dec *d, Xmm *x, int aligned) {
    if (d->mod == 3) *x = c->xmm[d->rm];
    else vread128(c, d->ea_seg, ea_off(d), x, aligned);
}

static u64 get_src64(Cpu *c, Dec *d) {
    return d->mod == 3 ? c->xmm[d->rm].q[0] : vread(c, d->ea_seg, ea_off(d), 8);
}

static u32 get_src32(Cpu *c, Dec *d) {
    return d->mod == 3 ? c->xmm[d->rm].d[0] : (u32)vread(c, d->ea_seg, ea_off(d), 4);
}

static float f32(u32 v) { float f; __builtin_memcpy(&f, &v, 4); return f; }
static double f64(u64 v) { double f; __builtin_memcpy(&f, &v, 8); return f; }

/* ---- IEEE single/double operations with SSE semantics (see header) ---- */
#if defined(__i386__)
typedef u16 FpEnv;
static inline FpEnv fp_begin(const Cpu *c, int dbl) {
    FpEnv old;
    u16 cw = (u16)(0x007fu | ((c->mxcsr >> 13) & 3u) << 10 | (dbl ? 2u : 0u) << 8);   /* masked, RC, PC */
    __asm__ volatile("fnstcw %0" : "=m"(old) : : "memory");
    if (old != cw) __asm__ volatile("fldcw %0" : : "m"(cw) : "memory");
    return old;
}
static inline void fp_end(FpEnv old) { __asm__ volatile("fldcw %0" : : "m"(old) : "memory"); }
#elif defined(__x86_64__)
typedef u32 FpEnv;
static inline FpEnv fp_begin(const Cpu *c, int dbl) {
    FpEnv old;
    (void)dbl;
    __asm__ volatile("stmxcsr %0" : "=m"(old) : : "memory");
    u32 n = (old & ~0x6000u) | (c->mxcsr & 0x6000u);
    if (n != old) __asm__ volatile("ldmxcsr %0" : : "m"(n) : "memory");
    return old;
}
static inline void fp_end(FpEnv old) { __asm__ volatile("ldmxcsr %0" : : "m"(old) : "memory"); }
#else
typedef int FpEnv;
static inline FpEnv fp_begin(const Cpu *c, int dbl) { (void)c; (void)dbl; return 0; }
static inline void fp_end(FpEnv old) { (void)old; }
#endif

static u32 fbits(float f) { u32 v; __builtin_memcpy(&v, &f, 4); return v; }
static u64 dbits(double f) { u64 v; __builtin_memcpy(&v, &f, 8); return v; }
static int isnan_s(u32 v) { return (v & 0x7fffffffu) > 0x7f800000u; }
static int isnan_d(u64 v) { return (v & 0x7fffffffffffffffull) > 0x7ff0000000000000ull; }
static u32 daz_s(const Cpu *c, u32 v) { return (c->mxcsr & 0x40) && !(v & 0x7f800000u) ? v & 0x80000000u : v; }
static u64 daz_d(const Cpu *c, u64 v) { return (c->mxcsr & 0x40) && !(v & 0x7ff0000000000000ull) ? v & 0x8000000000000000ull : v; }
static u32 ftz_s(const Cpu *c, u32 v) { return (c->mxcsr & 0x8000) && !(v & 0x7f800000u) ? v & 0x80000000u : v; }
static u64 ftz_d(const Cpu *c, u64 v) { return (c->mxcsr & 0x8000) && !(v & 0x7ff0000000000000ull) ? v & 0x8000000000000000ull : v; }

enum { FADD = 0x58, FMUL = 0x59, FSUB = 0x5c, FMIN = 0x5d, FDIV = 0x5e, FMAX = 0x5f, FSQRT = 0x51 };

/* a op b (FSQRT: sqrt b); MIN/MAX return b unless a is strictly less/greater */
static u32 fop_s(const Cpu *c, int op, u32 a, u32 b) {
    a = daz_s(c, a); b = daz_s(c, b);
    if (op == FMIN || op == FMAX) {
        float fa = f32(a), fb = f32(b);
        return (op == FMIN ? fa < fb : fa > fb) ? a : b;
    }
    if (op == FSQRT) a = b;
    if (isnan_s(a)) return a | 0x00400000u;
    if (isnan_s(b)) return b | 0x00400000u;
    /* computed at double precision, rounded to single once by the store: correct for
     * + - * / sqrt since 53 >= 2 * 24 + 2, including denormal results */
    FpEnv env = fp_begin(c, 1);
    volatile float va = f32(a), vb = f32(b), vr;
    switch (op) {
    case FADD: vr = va + vb; break;
    case FSUB: vr = va - vb; break;
    case FMUL: vr = va * vb; break;
    case FDIV: vr = va / vb; break;
    default: vr = sqrt_f(va); break;
    }
    u32 r = fbits(vr);
    fp_end(env);
    return ftz_s(c, r);
}

static u64 fop_d(const Cpu *c, int op, u64 a, u64 b) {
    a = daz_d(c, a); b = daz_d(c, b);
    if (op == FMIN || op == FMAX) {
        double fa = f64(a), fb = f64(b);
        return (op == FMIN ? fa < fb : fa > fb) ? a : b;
    }
    if (op == FSQRT) a = b;
    if (isnan_d(a)) return a | 0x0008000000000000ull;
    if (isnan_d(b)) return b | 0x0008000000000000ull;
    FpEnv env = fp_begin(c, 1);
    volatile double va = f64(a), vb = f64(b), vr;
    switch (op) {
    case FADD: vr = va + vb; break;
    case FSUB: vr = va - vb; break;
    case FMUL: vr = va * vb; break;
    case FDIV: vr = va / vb; break;
    default: vr = sqrt_d(va); break;
    }
    u64 r = dbits(vr);
    fp_end(env);
    return ftz_d(c, r);
}

/* float -> double (exact) and double -> float (rounded per MXCSR) */
static u64 cvt_s2d(const Cpu *c, u32 a) {
    a = daz_s(c, a);
    if (isnan_s(a)) return (u64)(a >> 31) << 63 | 0x7ff8000000000000ull | (u64)(a & 0x7fffffu) << 29;
    return dbits((double)f32(a));
}
static u32 cvt_d2s(const Cpu *c, u64 a) {
    a = daz_d(c, a);
    if (isnan_d(a)) return (u32)(a >> 63) << 31 | 0x7fc00000u | (u32)((a >> 29) & 0x3fffffu);
    FpEnv env = fp_begin(c, 0);
    volatile double va = f64(a);
    volatile float vr = (float)va;
    u32 r = fbits(vr);
    fp_end(env);
    return ftz_s(c, r);
}
/* integer -> float / double, rounded per MXCSR */
static u32 cvt_i2s(const Cpu *c, i64 v) {
    FpEnv env = fp_begin(c, 0);
    volatile i64 vv = v;
    volatile float vr = (float)vv;
    u32 r = fbits(vr);
    fp_end(env);
    return r;
}
static u64 cvt_i2d(const Cpu *c, i64 v) {
    FpEnv env = fp_begin(c, 1);
    volatile i64 vv = v;
    volatile double vr = (double)vv;
    u64 r = dbits(vr);
    fp_end(env);
    return r;
}
/* float / double -> integer (sz 4 or 8), rounded per MXCSR or truncated */
static u64 cvt_s2i(Cpu *c, u32 a, int sz, int trunc) { return cvt_d2i(c, f64(cvt_s2d(c, a)), sz, trunc); }
static u64 cvt_dd2i(Cpu *c, u64 a, int sz, int trunc) { return cvt_d2i(c, f64(daz_d(c, a)), sz, trunc); }
/* COMIS/UCOMIS flags */
static u64 fcmp_flags(double a, double b) {
    if (is_nan_d(a) || is_nan_d(b)) return F_ZF_ | F_PF_ | F_CF_;
    if (a < b) return F_CF_;
    if (a == b) return F_ZF_;
    return 0;
}

static i16 sat16(i32 v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (i16)v; }
static i8 sat8(i32 v) { return v > 127 ? 127 : v < -128 ? -128 : (i8)v; }
static u8 satu8(i32 v) { return v > 255 ? 255 : v < 0 ? 0 : (u8)v; }
static u16 satu16(i32 v) { return v > 65535 ? 65535 : v < 0 ? 0 : (u16)v; }

static void shift_imm(Xmm *x, int grp, int r, u32 n) {
    int i;
    switch (grp * 8 + r) {
    case 0x71 * 8 + 2: for (i = 0; i < 8; i++) x->w[i] = n > 15 ? 0 : (u16)(x->w[i] >> n); break;
    case 0x71 * 8 + 4: for (i = 0; i < 8; i++) x->sw[i] = (i16)(x->sw[i] >> (n > 15 ? 15 : n)); break;
    case 0x71 * 8 + 6: for (i = 0; i < 8; i++) x->w[i] = n > 15 ? 0 : (u16)(x->w[i] << n); break;
    case 0x72 * 8 + 2: for (i = 0; i < 4; i++) x->d[i] = n > 31 ? 0 : x->d[i] >> n; break;
    case 0x72 * 8 + 4: for (i = 0; i < 4; i++) x->sd[i] = x->sd[i] >> (n > 31 ? 31 : n); break;
    case 0x72 * 8 + 6: for (i = 0; i < 4; i++) x->d[i] = n > 31 ? 0 : x->d[i] << n; break;
    case 0x73 * 8 + 2: for (i = 0; i < 2; i++) x->q[i] = n > 63 ? 0 : x->q[i] >> n; break;
    case 0x73 * 8 + 6: for (i = 0; i < 2; i++) x->q[i] = n > 63 ? 0 : x->q[i] << n; break;
    case 0x73 * 8 + 3: {   /* PSRLDQ */
        Xmm t; e_memset(&t, 0, sizeof t);
        if (n < 16) for (i = 0; i + (int)n < 16; i++) t.b[i] = x->b[i + n];
        *x = t;
        break;
    }
    case 0x73 * 8 + 7: {   /* PSLLDQ */
        Xmm t; e_memset(&t, 0, sizeof t);
        if (n < 16) for (i = (int)n; i < 16; i++) t.b[i] = x->b[i - n];
        *x = t;
        break;
    }
    }
}

void fx_save(Cpu *c, int seg, u64 off) {
    Xmm z; e_memset(&z, 0, sizeof z);
    Xmm h; e_memset(&h, 0, sizeof h);
    h.w[0] = c->fcw; h.w[1] = c->fsw; h.b[4] = x87_abridged_tag(c);
    Xmm stv[8];
    for (int k = 0; k < 8; k++) {   /* ST(k) in 16-byte slots */
        e_memset(&stv[k], 0, 16);
        e_memcpy(stv[k].b, c->fpr[(((c->fsw >> 11) & 7) + k) & 7], 10);
    }
    Xmm h2; e_memset(&h2, 0, sizeof h2);
    h2.d[2] = c->mxcsr; h2.d[3] = 0xffff;
    int n = c->mode64 ? 16 : 8;
    if (seg_lin(c, seg, off) & 15) GP0(c);
    u8 *p = host_span(c, seg, off, 160 + 16u * (u32)n, 1);
    if (p) {   /* whole area in one page: same bytes, one translation */
        __builtin_memcpy(p, &h, 16);
        __builtin_memcpy(p + 16, &h2, 16);
        e_memcpy(p + 32, stv, 128);
        e_memcpy(p + 160, c->xmm, 16u * (u32)n);
        return;
    }
    vwrite128(c, seg, off, &h, 1);
    vwrite128(c, seg, off + 16, &h2, 1);
    for (int i = 0; i < 8; i++) vwrite128(c, seg, off + 32 + 16 * (u64)i, &stv[i], 1);
    (void)z;
    for (int i = 0; i < n; i++) vwrite128(c, seg, off + 160 + 16 * (u64)i, &c->xmm[i], 1);
}

void fx_restore(Cpu *c, int seg, u64 off) {
    Xmm h, h2;
    int n = c->mode64 ? 16 : 8;
    Xmm regs[16], stv[8];
    if (seg_lin(c, seg, off) & 15) GP0(c);
    u8 *p = host_span(c, seg, off, 160 + 16u * (u32)n, 0);
    if (p) {
        __builtin_memcpy(&h, p, 16);
        __builtin_memcpy(&h2, p + 16, 16);
        if (h2.d[2] & 0xffff0000u) GP0(c);
        e_memcpy(stv, p + 32, 128);
        e_memcpy(regs, p + 160, 16u * (u32)n);
    } else {
        vread128(c, seg, off, &h, 1);
        vread128(c, seg, off + 16, &h2, 1);
        if (h2.d[2] & 0xffff0000u) GP0(c);
        for (int i = 0; i < 8; i++) vread128(c, seg, off + 32 + 16 * (u64)i, &stv[i], 1);
        for (int i = 0; i < n; i++) vread128(c, seg, off + 160 + 16 * (u64)i, &regs[i], 1);
    }
    c->fcw = h.w[0]; c->fsw = h.w[1];
    x87_set_abridged_tag(c, h.b[4]);
    for (int k = 0; k < 8; k++) e_memcpy(c->fpr[(((c->fsw >> 11) & 7) + k) & 7], stv[k].b, 10);
    c->mxcsr = h2.d[2];
    for (int i = 0; i < n; i++) c->xmm[i] = regs[i];
}

/* Packed integer SSE2 operations (66 0F 60-6D, 74-76, D1-FE): x = x op s.
 * Shared by the interpreter and the JIT. Returns 0 for opcodes not handled. */
int sse3_exec(Cpu *c, Dec *d, int op, int pfx);

int sse_int_binop(int op, Xmm *x, const Xmm *sp) {
    Xmm s = *sp;
    int i;
    switch (op) {
    case 0x60: case 0x61: case 0x62: case 0x68: case 0x69: case 0x6a: case 0x6c: case 0x6d: {
        Xmm t;
        int hi = op >= 0x68;
        switch (op & 0xf7) {
        case 0x60: for (i = 0; i < 8; i++) { t.b[2 * i] = x->b[i + 8 * hi]; t.b[2 * i + 1] = s.b[i + 8 * hi]; } break;
        case 0x61: for (i = 0; i < 4; i++) { t.w[2 * i] = x->w[i + 4 * hi]; t.w[2 * i + 1] = s.w[i + 4 * hi]; } break;
        case 0x62: for (i = 0; i < 2; i++) { t.d[2 * i] = x->d[i + 2 * hi]; t.d[2 * i + 1] = s.d[i + 2 * hi]; } break;
        default: { int h = op == 0x6d; t.q[0] = x->q[h]; t.q[1] = s.q[h]; break; }
        }
        *x = t;
        return 1;
    }
    case 0x63: case 0x67: case 0x6b: {
        Xmm t;
        if (op == 0x63) for (i = 0; i < 8; i++) t.sb[i] = sat8(x->sw[i]), t.sb[i + 8] = sat8(s.sw[i]);
        else if (op == 0x67) for (i = 0; i < 8; i++) t.b[i] = satu8(x->sw[i]), t.b[i + 8] = satu8(s.sw[i]);
        else for (i = 0; i < 4; i++) t.sw[i] = sat16(x->sd[i]), t.sw[i + 4] = sat16(s.sd[i]);
        *x = t;
        return 1;
    }
    case 0x64: case 0x65: case 0x66: case 0x74: case 0x75: case 0x76:
        switch (op) {
        case 0x64: for (i = 0; i < 16; i++) x->b[i] = x->sb[i] > s.sb[i] ? 0xff : 0; break;
        case 0x65: for (i = 0; i < 8; i++) x->w[i] = x->sw[i] > s.sw[i] ? 0xffff : 0; break;
        case 0x66: for (i = 0; i < 4; i++) x->d[i] = x->sd[i] > s.sd[i] ? 0xffffffffu : 0; break;
        case 0x74: for (i = 0; i < 16; i++) x->b[i] = x->b[i] == s.b[i] ? 0xff : 0; break;
        case 0x75: for (i = 0; i < 8; i++) x->w[i] = x->w[i] == s.w[i] ? 0xffff : 0; break;
        default: for (i = 0; i < 4; i++) x->d[i] = x->d[i] == s.d[i] ? 0xffffffffu : 0; break;
        }
        return 1;
    default: break;
    }
    if (op >= 0xd0) {
        u64 cnt = s.q[0];
        switch (op) {
        case 0xd1: for (i = 0; i < 8; i++) x->w[i] = cnt > 15 ? 0 : (u16)(x->w[i] >> cnt); break;
        case 0xd2: for (i = 0; i < 4; i++) x->d[i] = cnt > 31 ? 0 : x->d[i] >> cnt; break;
        case 0xd3: for (i = 0; i < 2; i++) x->q[i] = cnt > 63 ? 0 : x->q[i] >> cnt; break;
        case 0xd4: for (i = 0; i < 2; i++) x->q[i] += s.q[i]; break;
        case 0xd5: for (i = 0; i < 8; i++) x->w[i] = (u16)(x->sw[i] * s.sw[i]); break;
        case 0xd8: for (i = 0; i < 16; i++) x->b[i] = satu8((i32)x->b[i] - s.b[i]); break;
        case 0xd9: for (i = 0; i < 8; i++) x->w[i] = satu16((i32)x->w[i] - s.w[i]); break;
        case 0xda: for (i = 0; i < 16; i++) x->b[i] = x->b[i] < s.b[i] ? x->b[i] : s.b[i]; break;
        case 0xdb: x->q[0] &= s.q[0]; x->q[1] &= s.q[1]; break;
        case 0xdc: for (i = 0; i < 16; i++) x->b[i] = satu8((i32)x->b[i] + s.b[i]); break;
        case 0xdd: for (i = 0; i < 8; i++) x->w[i] = satu16((i32)x->w[i] + s.w[i]); break;
        case 0xde: for (i = 0; i < 16; i++) x->b[i] = x->b[i] > s.b[i] ? x->b[i] : s.b[i]; break;
        case 0xdf: x->q[0] = ~x->q[0] & s.q[0]; x->q[1] = ~x->q[1] & s.q[1]; break;
        case 0xe0: for (i = 0; i < 16; i++) x->b[i] = (u8)((x->b[i] + s.b[i] + 1) >> 1); break;
        case 0xe1: for (i = 0; i < 8; i++) x->sw[i] = (i16)(x->sw[i] >> (cnt > 15 ? 15 : cnt)); break;
        case 0xe2: for (i = 0; i < 4; i++) x->sd[i] = x->sd[i] >> (cnt > 31 ? 31 : cnt); break;
        case 0xe3: for (i = 0; i < 8; i++) x->w[i] = (u16)((x->w[i] + s.w[i] + 1) >> 1); break;
        case 0xe4: for (i = 0; i < 8; i++) x->w[i] = (u16)(((u32)x->w[i] * s.w[i]) >> 16); break;
        case 0xe5: for (i = 0; i < 8; i++) x->w[i] = (u16)(((i32)x->sw[i] * s.sw[i]) >> 16); break;
        case 0xe8: for (i = 0; i < 16; i++) x->sb[i] = sat8((i32)x->sb[i] - s.sb[i]); break;
        case 0xe9: for (i = 0; i < 8; i++) x->sw[i] = sat16((i32)x->sw[i] - s.sw[i]); break;
        case 0xea: for (i = 0; i < 8; i++) x->sw[i] = x->sw[i] < s.sw[i] ? x->sw[i] : s.sw[i]; break;
        case 0xeb: x->q[0] |= s.q[0]; x->q[1] |= s.q[1]; break;
        case 0xec: for (i = 0; i < 16; i++) x->sb[i] = sat8((i32)x->sb[i] + s.sb[i]); break;
        case 0xed: for (i = 0; i < 8; i++) x->sw[i] = sat16((i32)x->sw[i] + s.sw[i]); break;
        case 0xee: for (i = 0; i < 8; i++) x->sw[i] = x->sw[i] > s.sw[i] ? x->sw[i] : s.sw[i]; break;
        case 0xef: x->q[0] ^= s.q[0]; x->q[1] ^= s.q[1]; break;
        case 0xf1: for (i = 0; i < 8; i++) x->w[i] = cnt > 15 ? 0 : (u16)(x->w[i] << cnt); break;
        case 0xf2: for (i = 0; i < 4; i++) x->d[i] = cnt > 31 ? 0 : x->d[i] << cnt; break;
        case 0xf3: for (i = 0; i < 2; i++) x->q[i] = cnt > 63 ? 0 : x->q[i] << cnt; break;
        case 0xf4: x->q[0] = (u64)x->d[0] * s.d[0]; x->q[1] = (u64)x->d[2] * s.d[2]; break;
        case 0xf5: for (i = 0; i < 4; i++)
                       x->sd[i] = (i32)x->sw[2 * i] * s.sw[2 * i] + (i32)x->sw[2 * i + 1] * s.sw[2 * i + 1];
                   break;
        case 0xf6: {
            u32 a = 0, b = 0;
            for (i = 0; i < 8; i++) {
                a += (u32)(x->b[i] > s.b[i] ? x->b[i] - s.b[i] : s.b[i] - x->b[i]);
                b += (u32)(x->b[i + 8] > s.b[i + 8] ? x->b[i + 8] - s.b[i + 8] : s.b[i + 8] - x->b[i + 8]);
            }
            x->q[0] = a; x->q[1] = b;
            break;
        }
        case 0xf8: for (i = 0; i < 16; i++) x->b[i] = (u8)(x->b[i] - s.b[i]); break;
        case 0xf9: for (i = 0; i < 8; i++) x->w[i] = (u16)(x->w[i] - s.w[i]); break;
        case 0xfa: for (i = 0; i < 4; i++) x->d[i] -= s.d[i]; break;
        case 0xfb: for (i = 0; i < 2; i++) x->q[i] -= s.q[i]; break;
        case 0xfc: for (i = 0; i < 16; i++) x->b[i] = (u8)(x->b[i] + s.b[i]); break;
        case 0xfd: for (i = 0; i < 8; i++) x->w[i] = (u16)(x->w[i] + s.w[i]); break;
        case 0xfe: for (i = 0; i < 4; i++) x->d[i] += s.d[i]; break;
        default: return 0;
        }
        return 1;
    }
    return 0;
}

/* Scalar helpers for the JIT (operands already fetched; no faults).
 * code = opcode | prefix << 8 (0 none, 1 66, 2 F2, 3 F3) | imm8 << 16 */
void sse_scalar_op(Cpu *c, u32 code, Xmm *x, const u8 *src) {
    int op = code & 0xff, pfx = (code >> 8) & 3, imm = (int)(code >> 16) & 0xff;
    u32 s32; u64 s64;
    __builtin_memcpy(&s32, src, 4);
    __builtin_memcpy(&s64, src, 8);
    switch (op) {
    case 0x2e: case 0x2f:
        if (pfx == 1) set_flags_raw(c, fcmp_flags(f64(daz_d(c, x->q[0])), f64(daz_d(c, s64))));
        else set_flags_raw(c, fcmp_flags(f64(cvt_s2d(c, x->d[0])), f64(cvt_s2d(c, s32))));
        return;
    case 0xc2:
        if (pfx == 2) x->q[0] = cmp_pred_d(f64(daz_d(c, x->q[0])), f64(daz_d(c, s64)), imm) ? ~0ull : 0;
        else x->d[0] = cmp_pred_d(f64(cvt_s2d(c, x->d[0])), f64(cvt_s2d(c, s32)), imm) ? 0xffffffffu : 0;
        return;
    default:   /* 0x51 SQRT, 0x58-0x5f arithmetic, MIN, MAX */
        if (pfx == 2) x->q[0] = fop_d(c, op, x->q[0], s64);
        else x->d[0] = fop_s(c, op, x->d[0], s32);
        return;
    }
}

/* packed ops for generated code, exactly as sse_exec does them: code = opcode | prefix << 8 | imm << 16;
 * src: 16 bytes (8 for CVTPS2PD) */
void sse_packed_op(Cpu *c, u32 code, Xmm *x, const u8 *src) {
    int op = code & 0xff, pfx = (code >> 8) & 3, imm = (int)(code >> 16) & 0xff, i;
    Xmm s;
    __builtin_memcpy(&s, src, op == 0x5a && pfx == 0 ? 8 : 16);
    switch (op) {
    case 0x58: case 0x59: case 0x5c: case 0x5d: case 0x5e: case 0x5f:
        if (pfx == 1) for (i = 0; i < 2; i++) x->q[i] = fop_d(c, op, x->q[i], s.q[i]);
        else for (i = 0; i < 4; i++) x->d[i] = fop_s(c, op, x->d[i], s.d[i]);
        return;
    case 0x5a:
        if (pfx == 1) { x->d[0] = cvt_d2s(c, s.q[0]); x->d[1] = cvt_d2s(c, s.q[1]); x->q[1] = 0; }
        else { u64 v = s.q[0]; x->q[0] = cvt_s2d(c, (u32)v); x->q[1] = cvt_s2d(c, (u32)(v >> 32)); }
        return;
    case 0x5b:
        if (pfx == 0) for (i = 0; i < 4; i++) x->d[i] = cvt_i2s(c, s.sd[i]);
        else for (i = 0; i < 4; i++) x->d[i] = (u32)cvt_s2i(c, s.d[i], 4, pfx == 3);
        return;
    case 0xc6: {
        Xmm t;
        if (pfx == 1) { t.q[0] = x->q[imm & 1]; t.q[1] = s.q[(imm >> 1) & 1]; }
        else { t.d[0] = x->d[imm & 3]; t.d[1] = x->d[(imm >> 2) & 3]; t.d[2] = s.d[(imm >> 4) & 3]; t.d[3] = s.d[(imm >> 6) & 3]; }
        *x = t;
        return;
    }
    }
}

/* CVT(T)SS2SI / CVT(T)SD2SI: code = opcode | prefix << 8 | dest size << 16 */
u64 sse_cvt_int(Cpu *c, u32 code, const u8 *src) {
    int op = code & 0xff, pfx = (code >> 8) & 3, dsz = (int)(code >> 16);
    u32 s32; u64 s64;
    __builtin_memcpy(&s32, src, 4);
    __builtin_memcpy(&s64, src, 8);
    return pfx == 3 ? cvt_s2i(c, s32, dsz, op == 0x2c) : cvt_dd2i(c, s64, dsz, op == 0x2c);
}

/* ---------------- MMX (0F xx without prefix): 64-bit registers aliasing x87 ----------------
 * MMn is the mantissa of physical x87 register n. Any MMX instruction sets TOP = 0 and marks
 * all registers valid; writes set the exponent field to all ones; EMMS empties the tags. */
static void mmx_enter(Cpu *c) {
    if (c->cr0 & 4) UD(c);                   /* CR0.EM */
    if (c->cr0 & 8) cpu_exc(c, EXC_NM, 0, 0);
    c->fsw &= (u16)~0x3800u;
    c->ftw = 0;
}
static u64 mm_get(Cpu *c, int i) { u64 v; __builtin_memcpy(&v, c->fpr[i & 7], 8); return v; }
static void mm_set(Cpu *c, int i, u64 v) {
    __builtin_memcpy(c->fpr[i & 7], &v, 8);
    c->fpr[i & 7][8] = 0xff; c->fpr[i & 7][9] = 0xff;
}
static u64 mm_src(Cpu *c, Dec *d) { return d->mod == 3 ? mm_get(c, d->rm) : vread(c, d->ea_seg, ea_off(d), 8); }

/* 64-bit form of a packed-integer op, computed with the 128-bit helper */
static int mmx_binop(int op, u64 a, u64 b, u64 *out) {
    Xmm x, s;
    e_memset(&x, 0, sizeof x); e_memset(&s, 0, sizeof s);
    if (op == 0x6c || op == 0x6d || op == 0xd6 || op == 0xe6 || op == 0xd0) return 0;
    if (op == 0x63 || op == 0x67 || op == 0x6b) {        /* packs: dst words then src words */
        x.q[0] = a; x.q[1] = b; s = x;
    } else if (op >= 0x68 && op <= 0x6a) {               /* unpack high halves */
        x.q[0] = a >> 32; s.q[0] = b >> 32; op -= 8;
    } else {
        x.q[0] = a; s.q[0] = b;
    }
    if (!sse_int_binop(op, &x, &s)) return 0;
    *out = x.q[0];
    return 1;
}

static int mmx_exec(Cpu *c, Dec *d, int op) {
    int i;
    if (op == 0x77) { if (c->cr0 & 4) UD(c); if (c->cr0 & 8) cpu_exc(c, EXC_NM, 0, 0); c->fsw &= (u16)~0x3800u; c->ftw = 0xffff; return 1; }  /* EMMS: TOP = 0, all empty */
    decode_modrm(c, d);
    mmx_enter(c);
    int r = d->reg & 7;
    switch (op) {
    case 0x6e: mm_set(c, r, rd_rm(c, d, d->rexw ? 8 : 4)); return 1;              /* MOVD/MOVQ mm, r/m */
    case 0x7e: wr_rm(c, d, d->rexw ? 8 : 4, d->rexw ? mm_get(c, r) : (u32)mm_get(c, r)); return 1;
    case 0x6f: mm_set(c, r, mm_src(c, d)); return 1;                              /* MOVQ mm, mm/m64 */
    case 0x7f: case 0xe7:                                                          /* MOVQ / MOVNTQ m64, mm */
        if (d->mod == 3) { if (op == 0xe7) UD(c); mm_set(c, d->rm, mm_get(c, r)); }
        else vwrite(c, d->ea_seg, ea_off(d), mm_get(c, r), 8);
        return 1;
    case 0x70: {                                                                   /* PSHUFW */
        u64 sv = mm_src(c, d);
        u8 imm = fetch8(c, d);
        u64 v = 0;
        for (i = 0; i < 4; i++) v |= ((sv >> (16 * ((imm >> (2 * i)) & 3))) & 0xffff) << (16 * i);
        mm_set(c, r, v);
        return 1;
    }
    case 0x71: case 0x72: case 0x73: {
        if (d->mod != 3) UD(c);
        u8 imm = fetch8(c, d);
        if (!(r == 2 || r == 6 || (r == 4 && op != 0x73))) UD(c);
        Xmm x; e_memset(&x, 0, sizeof x);
        x.q[0] = mm_get(c, d->rm);
        shift_imm(&x, op, r, imm);
        mm_set(c, d->rm, x.q[0]);
        return 1;
    }
    case 0xc4: {                                                                   /* PINSRW */
        u64 v = d->mod == 3 ? get_reg(c, d, d->rm, 4) : vread(c, d->ea_seg, ea_off(d), 2);
        u8 imm = fetch8(c, d);
        int sh = 16 * (imm & 3);
        mm_set(c, r, (mm_get(c, r) & ~(0xffffull << sh)) | ((v & 0xffff) << sh));
        return 1;
    }
    case 0xc5: {                                                                   /* PEXTRW */
        if (d->mod != 3) UD(c);
        u8 imm = fetch8(c, d);
        set_reg(c, d, d->reg, 4, (mm_get(c, d->rm) >> (16 * (imm & 3))) & 0xffff);
        return 1;
    }
    case 0xd7: {                                                                   /* PMOVMSKB */
        if (d->mod != 3) UD(c);
        u64 v = mm_get(c, d->rm);
        u32 m = 0;
        for (i = 0; i < 8; i++) m |= (u32)((v >> (8 * i + 7)) & 1) << i;
        set_reg(c, d, d->reg, 4, m);
        return 1;
    }
    default: {
        u64 out;
        u64 b = mm_src(c, d);
        if (!mmx_binop(op, mm_get(c, r), b, &out)) {
            emu_log(c->m, "MMX opcode 0f %02x at %llx is not implemented", op, (unsigned long long)d->start);
            UD(c);
        }
        mm_set(c, r, out);
        return 1;
    }
    }
}

/* CVTPI2PS/PD, CVT(T)PS2PI, CVT(T)PD2PI: x is the XMM register operand */
static void mmx_cvt(Cpu *c, Dec *d, int op, int pfx, Xmm *x) {
    int i;
    if (op == 0x2a) {
        u64 v;
        if (d->mod == 3) { mmx_enter(c); v = mm_get(c, d->rm); } else v = vread(c, d->ea_seg, ea_off(d), 8);
        i32 lo = (i32)(u32)v, hi = (i32)(u32)(v >> 32);
        if (pfx == 0) { x->d[0] = cvt_i2s(c, lo); x->d[1] = cvt_i2s(c, hi); }
        else { x->q[0] = cvt_i2d(c, lo); x->q[1] = cvt_i2d(c, hi); }
        return;
    }
    Xmm s;
    if (pfx == 0) { e_memset(&s, 0, sizeof s); s.q[0] = d->mod == 3 ? c->xmm[d->rm].q[0] : vread(c, d->ea_seg, ea_off(d), 8); }
    else get_src(c, d, &s, 1);
    mmx_enter(c);
    u64 v = 0;
    for (i = 0; i < 2; i++) {
        u64 r = pfx == 0 ? cvt_s2i(c, s.d[i], 4, op == 0x2c) : cvt_dd2i(c, s.q[i], 4, op == 0x2c);
        v |= (r & 0xffffffffull) << (32 * i);
    }
    mm_set(c, d->reg, v);
}

/* SSSE3 instructions on MMX registers (0F 38 00-0B/1C-1E, 0F 3A 0F without prefix) */
static int mmx_ssse3(Cpu *c, Dec *d, int map, int op) {
    int i;
    if (!((map == 0x38 && (op <= 0x0b || (op >= 0x1c && op <= 0x1e))) || (map == 0x3a && op == 0x0f))) return 0;
    decode_modrm(c, d);
    mmx_enter(c);
    int r = d->reg & 7;
    Xmm x, s, t;
    e_memset(&x, 0, sizeof x); e_memset(&s, 0, sizeof s); e_memset(&t, 0, sizeof t);
    x.q[0] = mm_get(c, r);
    s.q[0] = mm_src(c, d);
    if (map == 0x3a) {                                   /* PALIGNR: (x:s) >> imm bytes */
        u8 imm = fetch8(c, d);
        u8 cat[16];
        __builtin_memcpy(cat, &s.q[0], 8); __builtin_memcpy(cat + 8, &x.q[0], 8);
        for (i = 0; i < 8; i++) t.b[i] = (u32)i + imm < 16 ? cat[i + imm] : 0;
        mm_set(c, r, t.q[0]);
        return 1;
    }
    switch (op) {
    case 0x00: for (i = 0; i < 8; i++) t.b[i] = (s.b[i] & 0x80) ? 0 : x.b[s.b[i] & 7]; break;
    case 0x01: for (i = 0; i < 2; i++) { t.w[i] = (u16)(x.w[2 * i] + x.w[2 * i + 1]); t.w[i + 2] = (u16)(s.w[2 * i] + s.w[2 * i + 1]); } break;
    case 0x02: t.d[0] = x.d[0] + x.d[1]; t.d[1] = s.d[0] + s.d[1]; break;
    case 0x03: for (i = 0; i < 2; i++) { t.sw[i] = sat16((i32)x.sw[2 * i] + x.sw[2 * i + 1]); t.sw[i + 2] = sat16((i32)s.sw[2 * i] + s.sw[2 * i + 1]); } break;
    case 0x04: for (i = 0; i < 4; i++) t.sw[i] = sat16((i32)x.b[2 * i] * s.sb[2 * i] + (i32)x.b[2 * i + 1] * s.sb[2 * i + 1]); break;
    case 0x05: for (i = 0; i < 2; i++) { t.w[i] = (u16)(x.w[2 * i] - x.w[2 * i + 1]); t.w[i + 2] = (u16)(s.w[2 * i] - s.w[2 * i + 1]); } break;
    case 0x06: t.d[0] = x.d[0] - x.d[1]; t.d[1] = s.d[0] - s.d[1]; break;
    case 0x07: for (i = 0; i < 2; i++) { t.sw[i] = sat16((i32)x.sw[2 * i] - x.sw[2 * i + 1]); t.sw[i + 2] = sat16((i32)s.sw[2 * i] - s.sw[2 * i + 1]); } break;
    case 0x08: for (i = 0; i < 8; i++) t.sb[i] = s.sb[i] < 0 ? (i8)-x.sb[i] : s.sb[i] ? x.sb[i] : 0; break;
    case 0x09: for (i = 0; i < 4; i++) t.sw[i] = s.sw[i] < 0 ? (i16)-x.sw[i] : s.sw[i] ? x.sw[i] : 0; break;
    case 0x0a: for (i = 0; i < 2; i++) t.sd[i] = s.sd[i] < 0 ? (i32)(0u - (u32)x.sd[i]) : s.sd[i] ? x.sd[i] : 0; break;
    case 0x0b: for (i = 0; i < 4; i++) t.sw[i] = (i16)((((i32)x.sw[i] * s.sw[i] >> 14) + 1) >> 1); break;
    case 0x1c: for (i = 0; i < 8; i++) t.b[i] = (u8)(s.sb[i] < 0 ? -s.sb[i] : s.sb[i]); break;
    case 0x1d: for (i = 0; i < 4; i++) t.w[i] = (u16)(s.sw[i] < 0 ? -s.sw[i] : s.sw[i]); break;
    default:   for (i = 0; i < 2; i++) t.d[i] = s.sd[i] < 0 ? 0u - (u32)s.sd[i] : (u32)s.sd[i]; break;
    }
    mm_set(c, r, t.q[0]);
    return 1;
}

static void check_sse(Cpu *c) {
    if (c->cr0 & 4) UD(c);                  /* CR0.EM */
    if (!(c->cr4 & 0x200)) UD(c);          /* CR4.OSFXSR */
    if (c->cr0 & 8) cpu_exc(c, EXC_NM, 0, 0);
}

int sse_exec(Cpu *c, Dec *d, int op) {
    int pfx = d->rep == 0xf3 ? 3 : d->rep == 0xf2 ? 2 : d->p66 ? 1 : 0;   /* 0 ps, 1 pd/66, 2 sd, 3 ss */
    int i;
    Xmm s, *x;

    if (sse3_exec(c, d, op, pfx)) return 1;

    if (op == 0xae) {
        decode_modrm(c, d);
        int r = d->reg & 7;
        if (d->mod == 3) {
            if (r >= 5) return 1;           /* LFENCE / MFENCE / SFENCE */
            return 0;
        }
        switch (r) {
        case 0: if (c->cr0 & 8) cpu_exc(c, EXC_NM, 0, 0); fx_save(c, d->ea_seg, ea_off(d)); return 1;
        case 1: if (c->cr0 & 8) cpu_exc(c, EXC_NM, 0, 0); fx_restore(c, d->ea_seg, ea_off(d)); return 1;
        case 2: {
            check_sse(c);
            u32 v = (u32)vread(c, d->ea_seg, ea_off(d), 4);
            if (v & 0xffff0000u) GP0(c);
            c->mxcsr = v;
            return 1;
        }
        case 3: check_sse(c); vwrite(c, d->ea_seg, ea_off(d), c->mxcsr, 4); return 1;
        case 7: return 1;                   /* CLFLUSH */
        default: return 0;
        }
    }

    /* opcodes handled here */
    int known = (op >= 0x10 && op <= 0x17) || op == 0x28 || op == 0x29 || (op >= 0x2a && op <= 0x2f) ||
                (op >= 0x50 && op <= 0x7f) || op == 0xc2 || (op >= 0xc4 && op <= 0xc6) || op >= 0xd0;
    if (!known) return 0;
    /* MMX register forms (no prefix) */
    if (((op >= 0x60 && op <= 0x7f) || op >= 0xd0 || op == 0xc4 || op == 0xc5) && pfx == 0) return mmx_exec(c, d, op);
    check_sse(c);
    decode_modrm(c, d);
    x = &c->xmm[d->reg];

    switch (op) {
    case 0x10: case 0x11:
        if (pfx >= 2) {   /* MOVSS / MOVSD */
            int sz = pfx == 3 ? 4 : 8;
            if (op == 0x10) {
                if (d->mod == 3) {
                    if (sz == 4) x->d[0] = c->xmm[d->rm].d[0]; else x->q[0] = c->xmm[d->rm].q[0];
                } else {
                    u64 v = vread(c, d->ea_seg, ea_off(d), sz);
                    e_memset(x, 0, sizeof *x);
                    if (sz == 4) x->d[0] = (u32)v; else x->q[0] = v;
                }
            } else {
                if (d->mod == 3) {
                    if (sz == 4) c->xmm[d->rm].d[0] = x->d[0]; else c->xmm[d->rm].q[0] = x->q[0];
                } else vwrite(c, d->ea_seg, ea_off(d), sz == 4 ? x->d[0] : x->q[0], sz);
            }
        } else {           /* MOVUPS / MOVUPD */
            if (op == 0x10) { get_src(c, d, &s, 0); *x = s; }
            else if (d->mod == 3) c->xmm[d->rm] = *x;
            else vwrite128(c, d->ea_seg, ea_off(d), x, 0);
        }
        return 1;
    case 0x12: case 0x13: case 0x16: case 0x17: {
        int high = op >= 0x16;
        if (pfx >= 2) {
            emu_log(c->m, "SSE3 opcode 0f %02x is not implemented", op);
            UD(c);
        }
        if (op == 0x12 || op == 0x16) {
            if (d->mod == 3) {
                if (pfx == 1) UD(c);
                if (high) x->q[1] = c->xmm[d->rm].q[0];   /* MOVLHPS */
                else x->q[0] = c->xmm[d->rm].q[1];        /* MOVHLPS */
            } else {
                x->q[high] = vread(c, d->ea_seg, ea_off(d), 8);
            }
        } else {
            if (d->mod == 3) UD(c);
            vwrite(c, d->ea_seg, ea_off(d), x->q[high], 8);
        }
        return 1;
    }
    case 0x14: case 0x15:
        get_src(c, d, &s, 1);
        if (pfx == 1) {
            if (op == 0x14) x->q[1] = s.q[0];
            else { x->q[0] = x->q[1]; x->q[1] = s.q[1]; }
        } else {
            Xmm t;
            int h = op == 0x15 ? 2 : 0;
            t.d[0] = x->d[h]; t.d[1] = s.d[h]; t.d[2] = x->d[h + 1]; t.d[3] = s.d[h + 1];
            *x = t;
        }
        return 1;
    case 0x28: get_src(c, d, &s, 1); *x = s; return 1;
    case 0x29: case 0x2b:
        if (d->mod == 3) { if (op == 0x2b) UD(c); c->xmm[d->rm] = *x; }
        else vwrite128(c, d->ea_seg, ea_off(d), x, 1);
        return 1;
    case 0x2a: {
        if (pfx < 2) { mmx_cvt(c, d, op, pfx, x); return 1; }
        int ssz = d->rexw ? 8 : 4;
        i64 v = d->mod == 3 ? (i64)sext(get_reg(c, d, d->rm, ssz), ssz)
                            : (i64)sext(vread(c, d->ea_seg, ea_off(d), ssz), ssz);
        if (pfx == 3) x->d[0] = cvt_i2s(c, v); else x->q[0] = cvt_i2d(c, v);
        return 1;
    }
    case 0x2c: case 0x2d: {
        if (pfx < 2) { mmx_cvt(c, d, op, pfx, x); return 1; }
        int dsz = d->rexw ? 8 : 4;
        u64 r = pfx == 3 ? cvt_s2i(c, get_src32(c, d), dsz, op == 0x2c) : cvt_dd2i(c, get_src64(c, d), dsz, op == 0x2c);
        set_reg(c, d, d->reg, dsz, r);
        return 1;
    }
    case 0x2e: case 0x2f:
        if (pfx == 1) set_flags_raw(c, fcmp_flags(f64(daz_d(c, x->q[0])), f64(daz_d(c, get_src64(c, d)))));
        else set_flags_raw(c, fcmp_flags(f64(cvt_s2d(c, x->d[0])), f64(cvt_s2d(c, get_src32(c, d)))));
        return 1;
    case 0x50: {
        if (d->mod != 3) UD(c);
        Xmm *r = &c->xmm[d->rm];
        u32 m = 0;
        if (pfx == 1) m = (u32)((r->q[0] >> 63) | ((r->q[1] >> 63) << 1));
        else for (i = 0; i < 4; i++) m |= (r->d[i] >> 31) << i;
        set_reg(c, d, d->reg, 4, m);
        return 1;
    }
    case 0x51: case 0x52: case 0x53:
        if (pfx == 3) {
            u32 v = get_src32(c, d);
            if (op == 0x51) x->d[0] = fop_s(c, FSQRT, 0, v);
            else x->f[0] = op == 0x52 ? 1.0f / sqrt_f(f32(v)) : 1.0f / f32(v);         /* approximations */
        } else if (pfx == 2) {
            if (op != 0x51) UD(c);
            x->q[0] = fop_d(c, FSQRT, 0, get_src64(c, d));
        } else if (pfx == 1) {
            if (op != 0x51) UD(c);
            get_src(c, d, &s, 1);
            x->q[0] = fop_d(c, FSQRT, 0, s.q[0]); x->q[1] = fop_d(c, FSQRT, 0, s.q[1]);
        } else {
            get_src(c, d, &s, 1);
            for (i = 0; i < 4; i++) {
                if (op == 0x51) x->d[i] = fop_s(c, FSQRT, 0, s.d[i]);
                else x->f[i] = op == 0x52 ? 1.0f / sqrt_f(s.f[i]) : 1.0f / s.f[i];
            }
        }
        return 1;
    case 0x54: case 0x55: case 0x56: case 0x57:
        if (pfx >= 2) UD(c);
        get_src(c, d, &s, 1);
        for (i = 0; i < 2; i++) {
            u64 a = x->q[i], b = s.q[i];
            x->q[i] = op == 0x54 ? a & b : op == 0x55 ? ~a & b : op == 0x56 ? a | b : a ^ b;
        }
        return 1;
    case 0x58: case 0x59: case 0x5c: case 0x5d: case 0x5e: case 0x5f:
        if (pfx == 3) x->d[0] = fop_s(c, op, x->d[0], get_src32(c, d));
        else if (pfx == 2) x->q[0] = fop_d(c, op, x->q[0], get_src64(c, d));
        else if (pfx == 1) {
            get_src(c, d, &s, 1);
            for (i = 0; i < 2; i++) x->q[i] = fop_d(c, op, x->q[i], s.q[i]);
        } else {
            get_src(c, d, &s, 1);
            for (i = 0; i < 4; i++) x->d[i] = fop_s(c, op, x->d[i], s.d[i]);
        }
        return 1;
    case 0x5a:
        if (pfx == 3) x->q[0] = cvt_s2d(c, get_src32(c, d));                 /* CVTSS2SD */
        else if (pfx == 2) x->d[0] = cvt_d2s(c, get_src64(c, d));            /* CVTSD2SS */
        else if (pfx == 1) {                                                  /* CVTPD2PS */
            get_src(c, d, &s, 1);
            x->d[0] = cvt_d2s(c, s.q[0]); x->d[1] = cvt_d2s(c, s.q[1]); x->q[1] = 0;
        } else {                                                              /* CVTPS2PD */
            u64 v = get_src64(c, d);
            x->q[0] = cvt_s2d(c, (u32)v); x->q[1] = cvt_s2d(c, (u32)(v >> 32));
        }
        return 1;
    case 0x5b:
        get_src(c, d, &s, 1);
        if (pfx == 0) for (i = 0; i < 4; i++) x->d[i] = cvt_i2s(c, s.sd[i]);                /* CVTDQ2PS */
        else if (pfx == 1 || pfx == 3)                                                       /* CVT(T)PS2DQ */
            for (i = 0; i < 4; i++) x->d[i] = (u32)cvt_s2i(c, s.d[i], 4, pfx == 3);
        else UD(c);
        return 1;
    case 0x60: case 0x61: case 0x62: case 0x63: case 0x64: case 0x65: case 0x66: case 0x67:
    case 0x68: case 0x69: case 0x6a: case 0x6b: case 0x6c: case 0x6d: case 0x74: case 0x75: case 0x76:
        get_src(c, d, &s, 1);
        sse_int_binop(op, x, &s);
        return 1;
    case 0x6e: {
        int sz = d->rexw ? 8 : 4;
        u64 v = rd_rm(c, d, sz);
        e_memset(x, 0, sizeof *x);
        x->q[0] = v;
        return 1;
    }
    case 0x6f:
        if (pfx != 1 && pfx != 3) UD(c);
        get_src(c, d, &s, pfx == 1);
        *x = s;
        return 1;
    case 0x70: {
        get_src(c, d, &s, 1);
        u8 imm = fetch8(c, d);
        Xmm t = s;
        if (pfx == 1) for (i = 0; i < 4; i++) t.d[i] = s.d[(imm >> (2 * i)) & 3];
        else if (pfx == 2) for (i = 0; i < 4; i++) t.w[i] = s.w[(imm >> (2 * i)) & 3];
        else for (i = 0; i < 4; i++) t.w[4 + i] = s.w[4 + ((imm >> (2 * i)) & 3)];
        *x = t;
        return 1;
    }
    case 0x71: case 0x72: case 0x73: {
        if (d->mod != 3) UD(c);
        u8 imm = fetch8(c, d);
        shift_imm(&c->xmm[d->rm], op, d->reg & 7, imm);
        return 1;
    }
    case 0x7e:
        if (pfx == 3) {    /* MOVQ xmm, xmm/m64 */
            u64 v = get_src64(c, d);
            x->q[0] = v; x->q[1] = 0;
        } else {           /* MOVD/MOVQ r/m, xmm */
            int sz = d->rexw ? 8 : 4;
            wr_rm(c, d, sz, sz == 8 ? x->q[0] : x->d[0]);
        }
        return 1;
    case 0x7f:
        if (pfx != 1 && pfx != 3) UD(c);
        if (d->mod == 3) c->xmm[d->rm] = *x;
        else vwrite128(c, d->ea_seg, ea_off(d), x, pfx == 1);
        return 1;
    case 0xc2: {
        int scalar = pfx >= 2;
        Xmm t;
        if (pfx == 3) t.d[0] = get_src32(c, d);
        else if (pfx == 2) t.q[0] = get_src64(c, d);
        else get_src(c, d, &t, 1);
        u8 p = fetch8(c, d);
        if (pfx == 1 || pfx == 2) {
            for (i = 0; i < (scalar ? 1 : 2); i++) x->q[i] = cmp_pred_d(f64(daz_d(c, x->q[i])), f64(daz_d(c, t.q[i])), p) ? ~0ull : 0;
        } else {
            for (i = 0; i < (scalar ? 1 : 4); i++) x->d[i] = cmp_pred_d(f64(cvt_s2d(c, x->d[i])), f64(cvt_s2d(c, t.d[i])), p) ? 0xffffffffu : 0;
        }
        return 1;
    }
    case 0xc4: {
        u64 v = d->mod == 3 ? get_reg(c, d, d->rm, 4) : vread(c, d->ea_seg, ea_off(d), 2);
        u8 imm = fetch8(c, d);
        x->w[imm & 7] = (u16)v;
        return 1;
    }
    case 0xc5: {
        if (d->mod != 3) UD(c);
        u8 imm = fetch8(c, d);
        set_reg(c, d, d->reg, 4, c->xmm[d->rm].w[imm & 7]);
        return 1;
    }
    case 0xc6: {
        get_src(c, d, &s, 1);
        u8 imm = fetch8(c, d);
        Xmm t;
        if (pfx == 1) { t.q[0] = x->q[imm & 1]; t.q[1] = s.q[(imm >> 1) & 1]; }
        else {
            t.d[0] = x->d[imm & 3]; t.d[1] = x->d[(imm >> 2) & 3];
            t.d[2] = s.d[(imm >> 4) & 3]; t.d[3] = s.d[(imm >> 6) & 3];
        }
        *x = t;
        return 1;
    }
    case 0xd6:
        if (pfx == 3 && d->mod == 3) {               /* MOVQ2DQ xmm, mm */
            mmx_enter(c);
            x->q[0] = mm_get(c, d->rm); x->q[1] = 0;
            return 1;
        }
        if (pfx == 2 && d->mod == 3) {               /* MOVDQ2Q mm, xmm */
            mmx_enter(c);
            mm_set(c, d->reg, c->xmm[d->rm].q[0]);
            return 1;
        }
        if (pfx != 1) UD(c);
        if (d->mod == 3) { c->xmm[d->rm].q[0] = x->q[0]; c->xmm[d->rm].q[1] = 0; }
        else vwrite(c, d->ea_seg, ea_off(d), x->q[0], 8);
        return 1;
    case 0xd7: {
        if (d->mod != 3) UD(c);
        u32 m = 0;
        for (i = 0; i < 16; i++) m |= (u32)(c->xmm[d->rm].b[i] >> 7) << i;
        set_reg(c, d, d->reg, 4, m);
        return 1;
    }
    case 0xe6: {
        get_src(c, d, &s, 1);
        if (pfx == 3) {   /* CVTDQ2PD */
            i32 a = s.sd[0], b = s.sd[1];
            x->fd[0] = a; x->fd[1] = b;
        } else {          /* CVTTPD2DQ (66) / CVTPD2DQ (F2) */
            u32 a = (u32)cvt_dd2i(c, s.q[0], 4, pfx == 1), b = (u32)cvt_dd2i(c, s.q[1], 4, pfx == 1);
            x->d[0] = a; x->d[1] = b; x->q[1] = 0;
        }
        return 1;
    }
    case 0xe7:
        if (d->mod == 3) UD(c);
        vwrite128(c, d->ea_seg, ea_off(d), x, 1);
        return 1;
    case 0xf7: UD(c);
    default:
        break;
    }

    if (op >= 0xd0) {
        if (pfx != 1) UD(c);
        get_src(c, d, &s, 1);
        if (!sse_int_binop(op, x, &s)) {
            emu_log(c->m, "SSE opcode 66 0f %02x at %llx is not implemented", op, (unsigned long long)d->start);
            UD(c);
        }
        return 1;
    }
    emu_log(c->m, "SSE opcode 0f %02x (prefix %d) at %llx is not implemented", op, pfx,
            (unsigned long long)d->start);
    UD(c);
}

/* ================= SSE3 / SSSE3 / SSE4.1 / SSE4.2 ================= */

static double round_imm(Cpu *c, double v, int imm) {
    int rc = (imm & 4) ? (int)((c->mxcsr >> 13) & 3) : (imm & 3);
    return round_mode(v, rc);
}
/* ROUNDPS/SS, ROUNDPD/SD on raw bits: NaNs are quieted, the sign of zero is kept */
static u32 round_s(Cpu *c, u32 a, int imm) {
    a = daz_s(c, a);
    if (isnan_s(a)) return a | 0x00400000u;
    return fbits((float)round_imm(c, f32(a), imm));
}
static u64 round_d(Cpu *c, u64 a, int imm) {
    a = daz_d(c, a);
    if (isnan_d(a)) return a | 0x0008000000000000ull;
    return dbits(round_imm(c, f64(a), imm));
}

static u32 crc32c_byte(u32 crc, u8 b) {
    crc ^= b;
    for (int i = 0; i < 8; i++) crc = (crc & 1) ? (crc >> 1) ^ 0x82f63b78u : crc >> 1;
    return crc;
}

/* PCMPESTRI/PCMPESTRM/PCMPISTRI/PCMPISTRM */
static void pcmpstr(Cpu *c, Dec *d, int op, const Xmm *a, const Xmm *b, int imm) {
    int words = imm & 1, sgn = (imm >> 1) & 1, agg = (imm >> 2) & 3, pol = (imm >> 4) & 3;
    int n = words ? 8 : 16;
    i32 ea[16], eb[16];
    for (int i = 0; i < n; i++) {
        if (words) { ea[i] = sgn ? a->sw[i] : a->w[i]; eb[i] = sgn ? b->sw[i] : b->w[i]; }
        else { ea[i] = sgn ? a->sb[i] : a->b[i]; eb[i] = sgn ? b->sb[i] : b->b[i]; }
    }
    int la, lb;
    if (op == 0x60 || op == 0x61) {          /* explicit lengths in rAX / rDX */
        i64 ra = d->rexw ? (i64)c->r[0] : (i64)(i32)c->r[0];
        i64 rb = d->rexw ? (i64)c->r[2] : (i64)(i32)c->r[2];
        if (ra < 0) ra = -ra;
        if (rb < 0) rb = -rb;
        la = ra > n ? n : (int)ra;
        lb = rb > n ? n : (int)rb;
    } else {                                 /* implicit: up to the first zero element */
        la = n; lb = n;
        for (int i = 0; i < n; i++) if (!ea[i]) { la = i; break; }
        for (int i = 0; i < n; i++) if (!eb[i]) { lb = i; break; }
    }
    u32 r1 = 0;
    for (int j = 0; j < n; j++) {
        int bit = 0;
        switch (agg) {
        case 0:   /* equal any */
            if (j < lb) for (int i = 0; i < la; i++) if (ea[i] == eb[j]) { bit = 1; break; }
            break;
        case 1:   /* ranges */
            if (j < lb) for (int i = 0; i + 1 < la; i += 2) if (eb[j] >= ea[i] && eb[j] <= ea[i + 1]) { bit = 1; break; }
            break;
        case 2: { /* equal each */
            int va = j < la, vb = j < lb;
            bit = (!va && !vb) ? 1 : (va && vb) ? ea[j] == eb[j] : 0;
            break;
        }
        default: { /* equal ordered */
            bit = 1;
            for (int k = 0; k < n - j; k++) {
                if (k >= la) break;                       /* A exhausted: matches */
                if (j + k >= lb) { bit = 0; break; }      /* B exhausted first */
                if (ea[k] != eb[j + k]) { bit = 0; break; }
            }
            break;
        }
        }
        r1 |= (u32)bit << j;
    }
    u32 full = (1u << n) - 1, r2 = r1;
    if (pol == 1) r2 = ~r1 & full;
    else if (pol == 3) r2 = r1 ^ (((1u << lb) - 1) & full);
    u64 fl = (r2 ? F_CF_ : 0) | (lb < n ? F_ZF_ : 0) | (la < n ? 0x80u : 0) | ((r2 & 1) ? 0x800u : 0);
    set_flags_raw(c, fl);
    if (op & 1) {            /* index -> ECX */
        u32 idx = (u32)n;
        if (r2) idx = (imm & 0x40) ? 31u - (u32)__builtin_clz(r2) : (u32)__builtin_ctz(r2);
        c->r[1] = idx;
    } else {                 /* mask -> XMM0 */
        Xmm m;
        e_memset(&m, 0, sizeof m);
        if (imm & 0x40) {
            for (int i = 0; i < n; i++) if ((r2 >> i) & 1) { if (words) m.w[i] = 0xffff; else m.b[i] = 0xff; }
        } else m.d[0] = r2;
        c->xmm[0] = m;
    }
}

/* 66 0F 38 xx, F2 0F 38 F0/F1 (CRC32), 66 0F 3A xx */
int sse4_exec(Cpu *c, Dec *d, int map) {
    u8 op = fetch8(c, d);
    d->opc = (map == 0x38 ? 0x200 : 0x300) | op;
    int i;
    Xmm s, t, *x;
    if (map == 0x38 && (op == 0xf0 || op == 0xf1)) {
        if (d->rep != 0xf2) UD(c);                       /* MOVBE is not advertised */
        decode_modrm(c, d);
        int ssz = op == 0xf0 ? 1 : d->osz;
        u64 v = rd_rm(c, d, ssz);
        u32 crc = (u32)c->r[d->reg];
        for (int k = 0; k < ssz; k++) crc = crc32c_byte(crc, (u8)(v >> (8 * k)));
        c->r[d->reg] = crc;                              /* 32- or 64-bit destination: zero-extended */
        return 1;
    }
    if (!d->p66 && !d->rep && mmx_ssse3(c, d, map, op)) return 1;
    if (!d->p66 || d->rep) {
        emu_log(c->m, "0f %02x %02x without 66 prefix (MMX form) is not implemented", map, op);
        UD(c);
    }
    check_sse(c);
    decode_modrm(c, d);
    x = &c->xmm[d->reg];

    if (map == 0x38) {
        /* PMOVSX / PMOVZX: narrower memory source */
        if ((op >= 0x20 && op <= 0x25) || (op >= 0x30 && op <= 0x35)) {
            static const int srcb[6] = { 8, 4, 2, 8, 4, 8 };
            int k = op & 7, zx = op >= 0x30;
            if (d->mod == 3) s = c->xmm[d->rm];
            else {
                e_memset(&s, 0, sizeof s);
                u64 v = vread(c, d->ea_seg, ea_off(d), srcb[k]);
                s.q[0] = v;
            }
            switch (k) {
            case 0: for (i = 0; i < 8; i++) t.w[i] = zx ? s.b[i] : (u16)s.sb[i]; break;
            case 1: for (i = 0; i < 4; i++) t.d[i] = zx ? s.b[i] : (u32)s.sb[i]; break;
            case 2: for (i = 0; i < 2; i++) t.q[i] = zx ? s.b[i] : (u64)(i64)s.sb[i]; break;
            case 3: for (i = 0; i < 4; i++) t.d[i] = zx ? s.w[i] : (u32)(i32)s.sw[i]; break;
            case 4: for (i = 0; i < 2; i++) t.q[i] = zx ? s.w[i] : (u64)(i64)s.sw[i]; break;
            default: for (i = 0; i < 2; i++) t.q[i] = zx ? s.d[i] : (u64)(i64)s.sd[i]; break;
            }
            *x = t;
            return 1;
        }
        get_src(c, d, &s, 1);
        t = *x;
        switch (op) {
        case 0x00: for (i = 0; i < 16; i++) t.b[i] = (s.b[i] & 0x80) ? 0 : x->b[s.b[i] & 15]; break;
        case 0x01: for (i = 0; i < 4; i++) { t.w[i] = (u16)(x->w[2 * i] + x->w[2 * i + 1]); t.w[i + 4] = (u16)(s.w[2 * i] + s.w[2 * i + 1]); } break;
        case 0x02: for (i = 0; i < 2; i++) { t.d[i] = x->d[2 * i] + x->d[2 * i + 1]; t.d[i + 2] = s.d[2 * i] + s.d[2 * i + 1]; } break;
        case 0x03: for (i = 0; i < 4; i++) { t.sw[i] = sat16((i32)x->sw[2 * i] + x->sw[2 * i + 1]); t.sw[i + 4] = sat16((i32)s.sw[2 * i] + s.sw[2 * i + 1]); } break;
        case 0x04: for (i = 0; i < 8; i++) t.sw[i] = sat16((i32)x->b[2 * i] * s.sb[2 * i] + (i32)x->b[2 * i + 1] * s.sb[2 * i + 1]); break;
        case 0x05: for (i = 0; i < 4; i++) { t.w[i] = (u16)(x->w[2 * i] - x->w[2 * i + 1]); t.w[i + 4] = (u16)(s.w[2 * i] - s.w[2 * i + 1]); } break;
        case 0x06: for (i = 0; i < 2; i++) { t.d[i] = x->d[2 * i] - x->d[2 * i + 1]; t.d[i + 2] = s.d[2 * i] - s.d[2 * i + 1]; } break;
        case 0x07: for (i = 0; i < 4; i++) { t.sw[i] = sat16((i32)x->sw[2 * i] - x->sw[2 * i + 1]); t.sw[i + 4] = sat16((i32)s.sw[2 * i] - s.sw[2 * i + 1]); } break;
        case 0x08: for (i = 0; i < 16; i++) t.sb[i] = s.sb[i] < 0 ? (i8)-x->sb[i] : s.sb[i] ? x->sb[i] : 0; break;
        case 0x09: for (i = 0; i < 8; i++) t.sw[i] = s.sw[i] < 0 ? (i16)-x->sw[i] : s.sw[i] ? x->sw[i] : 0; break;
        case 0x0a: for (i = 0; i < 4; i++) t.d[i] = s.sd[i] < 0 ? (u32)0 - x->d[i] : s.sd[i] ? x->d[i] : 0; break;
        case 0x0b: for (i = 0; i < 8; i++) t.sw[i] = (i16)((((i32)x->sw[i] * s.sw[i] >> 14) + 1) >> 1); break;
        case 0x10: for (i = 0; i < 16; i++) if (c->xmm[0].b[i] & 0x80) t.b[i] = s.b[i]; break;
        case 0x14: for (i = 0; i < 4; i++) if (c->xmm[0].d[i] >> 31) t.d[i] = s.d[i]; break;
        case 0x15: for (i = 0; i < 2; i++) if (c->xmm[0].q[i] >> 63) t.q[i] = s.q[i]; break;
        case 0x17: {
            u64 z = (x->q[0] & s.q[0]) | (x->q[1] & s.q[1]);
            u64 cz = (~x->q[0] & s.q[0]) | (~x->q[1] & s.q[1]);
            set_flags_raw(c, (z ? 0 : F_ZF_) | (cz ? 0 : F_CF_));
            return 1;
        }
        case 0x1c: for (i = 0; i < 16; i++) t.b[i] = (u8)(s.sb[i] < 0 ? -s.sb[i] : s.sb[i]); break;
        case 0x1d: for (i = 0; i < 8; i++) t.w[i] = (u16)(s.sw[i] < 0 ? -s.sw[i] : s.sw[i]); break;
        case 0x1e: for (i = 0; i < 4; i++) t.d[i] = s.sd[i] < 0 ? (u32)0 - (u32)s.sd[i] : (u32)s.sd[i]; break;
        case 0x28: t.q[0] = (u64)((i64)x->sd[0] * s.sd[0]); t.q[1] = (u64)((i64)x->sd[2] * s.sd[2]); break;
        case 0x29: for (i = 0; i < 2; i++) t.q[i] = x->q[i] == s.q[i] ? ~0ull : 0; break;
        case 0x2a: if (d->mod == 3) UD(c); t = s; break;
        case 0x2b: for (i = 0; i < 4; i++) { t.w[i] = satu16(x->sd[i] < 0 ? 0 : x->sd[i] > 65535 ? 65535 : x->sd[i]); t.w[i + 4] = satu16(s.sd[i] < 0 ? 0 : s.sd[i] > 65535 ? 65535 : s.sd[i]); } break;
        case 0x37: for (i = 0; i < 2; i++) t.q[i] = x->sq[i] > s.sq[i] ? ~0ull : 0; break;
        case 0x38: for (i = 0; i < 16; i++) t.sb[i] = x->sb[i] < s.sb[i] ? x->sb[i] : s.sb[i]; break;
        case 0x39: for (i = 0; i < 4; i++) t.sd[i] = x->sd[i] < s.sd[i] ? x->sd[i] : s.sd[i]; break;
        case 0x3a: for (i = 0; i < 8; i++) t.w[i] = x->w[i] < s.w[i] ? x->w[i] : s.w[i]; break;
        case 0x3b: for (i = 0; i < 4; i++) t.d[i] = x->d[i] < s.d[i] ? x->d[i] : s.d[i]; break;
        case 0x3c: for (i = 0; i < 16; i++) t.sb[i] = x->sb[i] > s.sb[i] ? x->sb[i] : s.sb[i]; break;
        case 0x3d: for (i = 0; i < 4; i++) t.sd[i] = x->sd[i] > s.sd[i] ? x->sd[i] : s.sd[i]; break;
        case 0x3e: for (i = 0; i < 8; i++) t.w[i] = x->w[i] > s.w[i] ? x->w[i] : s.w[i]; break;
        case 0x3f: for (i = 0; i < 4; i++) t.d[i] = x->d[i] > s.d[i] ? x->d[i] : s.d[i]; break;
        case 0x40: for (i = 0; i < 4; i++) t.d[i] = x->d[i] * s.d[i]; break;
        case 0x41: {
            u32 mi = 0;
            for (i = 1; i < 8; i++) if (s.w[i] < s.w[mi]) mi = (u32)i;
            e_memset(&t, 0, sizeof t);
            t.w[0] = s.w[mi]; t.w[1] = (u16)mi;
            break;
        }
        default:
            emu_log(c->m, "SSE4 opcode 66 0f 38 %02x at %llx is not implemented", op, (unsigned long long)d->start);
            UD(c);
        }
        *x = t;
        return 1;
    }

    /* 0F 3A: all forms take imm8 after the ModRM */
    switch (op) {
    case 0x14: case 0x15: case 0x16: case 0x17: {      /* PEXTRB/W/D/Q, EXTRACTPS */
        u8 imm = fetch8(c, d);
        u64 v; int sz;
        if (op == 0x14) { v = x->b[imm & 15]; sz = 1; }
        else if (op == 0x15) { v = x->w[imm & 7]; sz = 2; }
        else if (op == 0x16 && d->rexw) { v = x->q[imm & 1]; sz = 8; }
        else { v = x->d[imm & 3]; sz = 4; }
        if (d->mod == 3) set_reg(c, d, d->rm, sz == 8 ? 8 : 4, v);   /* register forms zero-extend */
        else vwrite(c, d->ea_seg, ea_off(d), v, sz);
        return 1;
    }
    case 0x20: case 0x22: {                            /* PINSRB, PINSRD/Q */
        int sz = op == 0x20 ? 1 : d->rexw ? 8 : 4;
        u64 v = d->mod == 3 ? get_reg(c, d, d->rm, sz == 1 ? 4 : sz) : vread(c, d->ea_seg, ea_off(d), sz);
        u8 imm = fetch8(c, d);
        if (sz == 1) x->b[imm & 15] = (u8)v;
        else if (sz == 4) x->d[imm & 3] = (u32)v;
        else x->q[imm & 1] = v;
        return 1;
    }
    case 0x21: {                                       /* INSERTPS */
        u32 v;
        u8 imm;
        if (d->mod == 3) { s = c->xmm[d->rm]; imm = fetch8(c, d); v = s.d[(imm >> 6) & 3]; }
        else { v = (u32)vread(c, d->ea_seg, ea_off(d), 4); imm = fetch8(c, d); }
        x->d[(imm >> 4) & 3] = v;
        for (i = 0; i < 4; i++) if ((imm >> i) & 1) x->d[i] = 0;
        return 1;
    }
    case 0x60: case 0x61: case 0x62: case 0x63: {      /* PCMPESTRM/I, PCMPISTRM/I (unaligned allowed) */
        get_src(c, d, &s, 0);
        u8 imm = fetch8(c, d);
        Xmm a = *x;
        pcmpstr(c, d, op, &a, &s, imm);
        return 1;
    }
    default: break;
    }
    get_src(c, d, &s, 1);
    u8 imm = fetch8(c, d);
    t = *x;
    switch (op) {
    case 0x08: for (i = 0; i < 4; i++) t.d[i] = round_s(c, s.d[i], imm); break;
    case 0x09: for (i = 0; i < 2; i++) t.q[i] = round_d(c, s.q[i], imm); break;
    case 0x0a: t.d[0] = round_s(c, s.d[0], imm); break;
    case 0x0b: t.q[0] = round_d(c, s.q[0], imm); break;
    case 0x0c: for (i = 0; i < 4; i++) if ((imm >> i) & 1) t.d[i] = s.d[i]; break;
    case 0x0d: for (i = 0; i < 2; i++) if ((imm >> i) & 1) t.q[i] = s.q[i]; break;
    case 0x0e: for (i = 0; i < 8; i++) if ((imm >> i) & 1) t.w[i] = s.w[i]; break;
    case 0x0f: {                                       /* PALIGNR: (x:s) >> imm bytes */
        u8 cat[32];
        e_memcpy(cat, s.b, 16); e_memcpy(cat + 16, x->b, 16);
        for (i = 0; i < 16; i++) t.b[i] = (u32)(i + imm) < 32 ? cat[i + imm] : 0;
        break;
    }
    case 0x40: {                                       /* DPPS */
        u32 p[4];
        for (i = 0; i < 4; i++) p[i] = ((imm >> (4 + i)) & 1) ? fop_s(c, FMUL, x->d[i], s.d[i]) : 0;
        u32 sum = fop_s(c, FADD, fop_s(c, FADD, p[0], p[1]), fop_s(c, FADD, p[2], p[3]));
        for (i = 0; i < 4; i++) t.d[i] = ((imm >> i) & 1) ? sum : 0;
        break;
    }
    case 0x41: {                                       /* DPPD */
        u64 p0 = (imm & 0x10) ? fop_d(c, FMUL, x->q[0], s.q[0]) : 0, p1 = (imm & 0x20) ? fop_d(c, FMUL, x->q[1], s.q[1]) : 0;
        u64 sum = fop_d(c, FADD, p0, p1);
        t.q[0] = (imm & 1) ? sum : 0; t.q[1] = (imm & 2) ? sum : 0;
        break;
    }
    case 0x42: {                                       /* MPSADBW */
        int ao = ((imm >> 2) & 1) * 4, bo = (imm & 3) * 4;
        for (i = 0; i < 8; i++) {
            u32 sum = 0;
            for (int k = 0; k < 4; k++) {
                int a = x->b[ao + i + k], b = s.b[bo + k];
                sum += (u32)(a > b ? a - b : b - a);
            }
            t.w[i] = (u16)sum;
        }
        break;
    }
    default:
        emu_log(c->m, "SSE4 opcode 66 0f 3a %02x at %llx is not implemented", op, (unsigned long long)d->start);
        UD(c);
    }
    *x = t;
    return 1;
}

/* SSE3 forms inside the 0F map; returns 1 if handled */
int sse3_exec(Cpu *c, Dec *d, int op, int pfx) {
    Xmm s, t, *x;
    int i;
    int is3 = (op == 0xd0 && (pfx == 1 || pfx == 2)) || ((op == 0x7c || op == 0x7d) && (pfx == 1 || pfx == 2)) ||
              (op == 0x12 && pfx >= 2) || (op == 0x16 && pfx == 3) || (op == 0xf0 && pfx == 2);
    if (!is3) return 0;
    check_sse(c);
    decode_modrm(c, d);
    x = &c->xmm[d->reg];
    if (op == 0xf0) { if (d->mod == 3) UD(c); get_src(c, d, &s, 0); *x = s; return 1; }   /* LDDQU */
    if (op == 0x12 && pfx == 2) {                       /* MOVDDUP */
        u64 v = d->mod == 3 ? c->xmm[d->rm].q[0] : vread(c, d->ea_seg, ea_off(d), 8);
        x->q[0] = x->q[1] = v;
        return 1;
    }
    get_src(c, d, &s, 1);
    if (op == 0x12) { t.d[0] = t.d[1] = s.d[0]; t.d[2] = t.d[3] = s.d[2]; *x = t; return 1; }   /* MOVSLDUP */
    if (op == 0x16) { t.d[0] = t.d[1] = s.d[1]; t.d[2] = t.d[3] = s.d[3]; *x = t; return 1; }   /* MOVSHDUP */
    if (op == 0xd0) {
        if (pfx == 2) for (i = 0; i < 4; i++) t.d[i] = fop_s(c, (i & 1) ? FADD : FSUB, x->d[i], s.d[i]);
        else { t.q[0] = fop_d(c, FSUB, x->q[0], s.q[0]); t.q[1] = fop_d(c, FADD, x->q[1], s.q[1]); }
        *x = t;
        return 1;
    }
    int fo = op == 0x7d ? FSUB : FADD;
    if (pfx == 2) {
        t.d[0] = fop_s(c, fo, x->d[0], x->d[1]);
        t.d[1] = fop_s(c, fo, x->d[2], x->d[3]);
        t.d[2] = fop_s(c, fo, s.d[0], s.d[1]);
        t.d[3] = fop_s(c, fo, s.d[2], s.d[3]);
    } else {
        t.q[0] = fop_d(c, fo, x->q[0], x->q[1]);
        t.q[1] = fop_d(c, fo, s.q[0], s.q[1]);
    }
    *x = t;
    return 1;
}
