/* Freestanding runtime for the bare-metal i386 image: the 64-bit division
 * helpers GCC calls on 32-bit targets. Implemented here (instead of linking
 * the multilib libgcc, which is built for i686) so the image stays i386-only. */
#include <stdint.h>

uint64_t __udivmoddi4(uint64_t n, uint64_t d, uint64_t *rem);
uint64_t __udivdi3(uint64_t n, uint64_t d);
uint64_t __umoddi3(uint64_t n, uint64_t d);
int64_t __divdi3(int64_t n, int64_t d);
int64_t __moddi3(int64_t n, int64_t d);
int64_t __divmoddi4(int64_t n, int64_t d, int64_t *rem);

uint64_t __udivmoddi4(uint64_t n, uint64_t d, uint64_t *rem) {
    if (!(d >> 32)) {
        uint32_t dd = (uint32_t)d, hi = (uint32_t)(n >> 32), lo = (uint32_t)n;
        uint32_t qhi = hi / dd, r = hi % dd, qlo;
        __asm__("divl %4" : "=a"(qlo), "=d"(r) : "a"(lo), "d"(r), "rm"(dd));
        if (rem) *rem = r;
        return ((uint64_t)qhi << 32) | qlo;
    }
    uint64_t q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= 1ull << i; }
    }
    if (rem) *rem = r;
    return q;
}

uint64_t __udivdi3(uint64_t n, uint64_t d) { return __udivmoddi4(n, d, 0); }
uint64_t __umoddi3(uint64_t n, uint64_t d) { uint64_t r; __udivmoddi4(n, d, &r); return r; }

int64_t __divmoddi4(int64_t n, int64_t d, int64_t *rem) {
    int nn = n < 0, nd = d < 0;
    uint64_t un = nn ? (uint64_t)0 - (uint64_t)n : (uint64_t)n;
    uint64_t ud = nd ? (uint64_t)0 - (uint64_t)d : (uint64_t)d;
    uint64_t r, q = __udivmoddi4(un, ud, &r);
    if (rem) *rem = nn ? (int64_t)((uint64_t)0 - r) : (int64_t)r;
    return (nn ^ nd) ? (int64_t)((uint64_t)0 - q) : (int64_t)q;
}

int64_t __divdi3(int64_t n, int64_t d) { return __divmoddi4(n, d, 0); }
int64_t __moddi3(int64_t n, int64_t d) { int64_t r; __divmoddi4(n, d, &r); return r; }

/* GCC may emit calls to these even in freestanding code */
void *memcpy(void *d, const void *s, unsigned n);
void *memset(void *d, int v, unsigned n);
void *memmove(void *d, const void *s, unsigned n);
int memcmp(const void *a, const void *b, unsigned n);

void *memcpy(void *d, const void *s, unsigned n) {
    void *ret = d;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) :: "memory");
    return ret;
}

void *memset(void *d, int v, unsigned n) {
    void *ret = d;
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(v) : "memory");
    return ret;
}

void *memmove(void *d, const void *s, unsigned n) {
    unsigned char *dp = d;
    const unsigned char *sp = s;
    if (dp <= sp || dp >= sp + n) return memcpy(d, s, n);
    while (n--) dp[n] = sp[n];
    return d;
}

int memcmp(const void *a, const void *b, unsigned n) {
    const unsigned char *x = a, *y = b;
    for (unsigned i = 0; i < n; i++) if (x[i] != y[i]) return x[i] - y[i];
    return 0;
}
