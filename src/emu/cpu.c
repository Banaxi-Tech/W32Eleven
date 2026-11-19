/* x86-64 integer core: memory/paging, decoder, interpreter, interrupts. */
#include "cpu_int.h"

#define CR0_PE 0x1ull
#define CR0_TS 0x8ull
#define CR0_ET 0x10ull
#define CR0_WP 0x10000ull
#define CR0_PG 0x80000000ull
#define CR4_PSE 0x10ull
#define CR4_PAE 0x20ull
#define EFER_SCE 0x1ull
#define EFER_LME 0x100ull
#define EFER_LMA 0x400ull
#define EFER_NXE 0x800ull

int jit_run_block(Cpu *c, u64 budget);
extern int jit_prof_on;
extern u32 jit_off;
void jit_invalidate_page(Machine *m, u64 pa);
int jit_page_is_code(Machine *m, u64 pa);
void jit_flush_lookup(Machine *m);
void jit_flush_code(Machine *m);

/* ================= exceptions ================= */

void cpu_exc(Cpu *c, int vec, int has_err, u32 err) {
    c->sys_access = 0;
    c->exc_vec = vec;
    c->exc_has_err = has_err;
    c->exc_err = err;
    __builtin_longjmp(c->jmp, 1);
}

/* ================= physical memory ================= */

static inline u8 *phys_page(Machine *m, u64 pa, int write) {
    u8 *fw;
    if (m->chipset && chipset_map(m, pa, write, &fw)) return fw;
    if (pa < m->ram_size) return m->ram + (uintptr_t)(pa & ~0xfffull);
    if (m->fb && pa >= m->fb_phys && pa < m->fb_phys + m->fb_size)
        return m->fb + (uintptr_t)((pa - m->fb_phys) & ~0xfffull);
    return 0;
}

static u64 mmio_read(Cpu *c, u64 pa, int sz) {
    Machine *m = c->m;
    u64 av;
    if (m->chipset && apic_mmio(m, pa, 0, &av, sz)) return av & MASK[sz];
    if (m->chipset) {
        u64 v = 0;
        int hit = 0;
        for (int i = 0; i < sz; i++) {
            const u8 *r = chipset_vga_rom(m, pa + (u64)i);
            if (r) { v |= (u64)*r << (8 * i); hit = 1; } else v |= 0xffull << (8 * i);
        }
        if (hit) return v;
    }
    return MASK[sz];        /* unbacked physical memory reads as all ones */
}

static void mmio_write(Cpu *c, u64 pa, u64 v, int sz) {
    if (c->m->chipset) apic_mmio(c->m, pa, 1, &v, sz);
}

static inline u64 pt_read(Machine *m, u64 pa) {
    if (pa + 8 > m->ram_size) return 0;
    u64 v; e_memcpy(&v, m->ram + (uintptr_t)pa, 8); return v;
}

static inline u32 pt_read32(Machine *m, u64 pa) {
    if (pa + 4 > m->ram_size) return 0;
    u32 v; e_memcpy(&v, m->ram + (uintptr_t)pa, 4); return v;
}

static inline void pt_set(Machine *m, u64 pa, u64 bits, int wide) {
    if (pa + 8 > m->ram_size) return;
    if (wide) { u64 v; e_memcpy(&v, m->ram + (uintptr_t)pa, 8); v |= bits; e_memcpy(m->ram + (uintptr_t)pa, &v, 8); }
    else { u32 v; e_memcpy(&v, m->ram + (uintptr_t)pa, 4); v |= (u32)bits; e_memcpy(m->ram + (uintptr_t)pa, &v, 4); }
}

/* A flush moves to a new tag epoch: entries of older epochs never match.
 * The arrays are cleared only when the 8-bit epoch wraps. */
u64 stat_glist_sum;
u32 stat_tlb_flush, stat_invlpg, stat_walks, stat_cr3;
u32 legacy_hist_pc[4096], legacy_hist_n[4096];

/* keep_global: CR3 write; entries of global pages move to the new epoch */
static void tlb_new_epoch(Cpu *c, int keep_global) {
    stat_tlb_flush++;
    { extern u64 stat_glist_sum; stat_glist_sum += c->nglist; }
    c->fetch_la = 1;
    if (c->m && c->m->jit) jit_flush_lookup(c->m);
    u32 old = c->tlb_epoch;
    c->tlb_epoch = (c->tlb_epoch + 1) & 0xff;
    if (!c->tlb_epoch) {
        e_memset(c->psc, 0, sizeof c->psc);
        e_memset(c->tlb_r, 0, sizeof c->tlb_r);
        e_memset(c->tlb_w, 0, sizeof c->tlb_w);
        e_memset(c->tlb_x, 0, sizeof c->tlb_x);
        keep_global = 0;
    }
    c->tlb_mode = (c->tlb_mode & 3) | c->tlb_epoch << 4;
    u32 n = 0, oldb = old << 4, newb = c->tlb_epoch << 4;
    u8 *gflag = &c->tlb_g[0][0];
    _Static_assert(__builtin_offsetof(Cpu, tlb_w) == __builtin_offsetof(Cpu, tlb_r) + sizeof(TlbEntry) * TLB_SIZE &&
                   __builtin_offsetof(Cpu, tlb_x) == __builtin_offsetof(Cpu, tlb_w) + sizeof(TlbEntry) * TLB_SIZE,
                   "TLB arrays are scanned as one");
    TlbEntry *all = (TlbEntry *)((u8 *)c + __builtin_offsetof(Cpu, tlb_r));
    if (!keep_global) {
        for (u32 k = 0; k < c->nglist; k++) gflag[c->glist[k]] = 0;
    } else {
        for (u32 k = 0; k < c->nglist; k++) {
            u32 g = c->glist[k], idx = g;   /* type * TLB_SIZE + index */
            u32 lo = (u32)all[idx].tag;
            if (gflag[idx] && lo && (lo & 0xff0) == oldb) {
                all[idx].tag = (all[idx].tag & ~0xff0ull) | newb;
                c->glist[n++] = (u16)g;
            } else gflag[idx] = 0;
        }
    }
    c->nglist = n;
}

void tlb_flush(Cpu *c) { tlb_new_epoch(c, 0); }

static void tlb_flush_page(Cpu *c, u64 la) {
    stat_invlpg++;
    e_memset(c->psc, 0, sizeof c->psc);
    u32 i = (u32)(la >> 12) & (TLB_SIZE - 1);
    c->tlb_r[i].tag = c->tlb_w[i].tag = c->tlb_x[i].tag = 0;
    c->fetch_la = 1;
    if (c->m->jit) jit_flush_lookup(c->m);
}

#define PTE_P 1ull
#define PTE_W 2ull
#define PTE_U 4ull
#define PTE_A 0x20ull
#define PTE_D 0x40ull
#define PTE_PS 0x80ull
#define PTE_NX (1ull << 63)
#define PTE_ADDR 0x000ffffffffff000ull

__attribute__((noreturn)) static void page_fault(Cpu *c, u64 la, u32 err) {
    c->cr2 = la;
    cpu_exc(c, EXC_PF, 1, err);
}

/* acc: 0 read, 1 write, 2 execute */
static u64 translate(Cpu *c, u64 la, int acc) {
    c->walk_global = 0;
    if (!(c->cr0 & CR0_PG)) return (u32)la;
    Machine *m = c->m;
    int user = c->cpl == 3 && !c->sys_access, write = acc == 1;
    int nxe = (c->efer & EFER_NXE) != 0;
    u32 err = (write ? 2 : 0) | (user ? 4 : 0) | (acc == 2 && nxe ? 0x10 : 0);
    if (c->efer & EFER_LMA || c->cr4 & CR4_PAE) {
        int lvl, top;
        u64 table, rw = 1, us = 1, nx = 0, pte = 0, pte_addr = 0, pa = 0;
        if (c->efer & EFER_LMA) {
            if ((u64)((i64)(la << 16) >> 16) != la) GP0(c);
            top = 0;
            table = c->cr3 & PTE_ADDR;
        } else {
            la = (u32)la;
            u64 pdpte = pt_read(m, (c->cr3 & 0xffffffe0ull) + ((la >> 30) & 3) * 8);
            if (!(pdpte & PTE_P)) page_fault(c, la, err);
            top = 2;
            table = pdpte & PTE_ADDR;
        }
        u32 pi = (u32)(la >> 21) & 255;
        u64 ptag = (la >> 21) | (1ull << 63);
        if (top == 0 && c->psc[pi].tag == ptag && c->psc[pi].epoch == c->tlb_epoch) {
            table = c->psc[pi].table;
            rw = c->psc[pi].bits & 1; us = (c->psc[pi].bits >> 1) & 1; nx = (c->psc[pi].bits >> 2) & 1;
            top = 3;
        }
        for (lvl = top; lvl < 4; lvl++) {
            int shift = 39 - 9 * lvl;
            pte_addr = table + ((la >> shift) & 511) * 8;
            pte = pt_read(m, pte_addr);
            if (!(pte & PTE_P)) page_fault(c, la, err);
            rw &= pte >> 1; us &= pte >> 2; nx |= pte >> 63;
            if (!(pte & PTE_A)) pt_set(m, pte_addr, PTE_A, 1);
            if ((lvl == 1 || lvl == 2) && (pte & PTE_PS)) {
                u64 pmask = (1ull << shift) - 1;
                pa = ((pte & PTE_ADDR) & ~pmask) | (la & pmask);
                break;
            }
            if (lvl == 3) { pa = (pte & PTE_ADDR) | (la & 0xfff); break; }
            table = pte & PTE_ADDR;
            if (lvl == 2 && (c->efer & EFER_LMA)) {
                c->psc[pi].tag = ptag; c->psc[pi].table = table; c->psc[pi].epoch = c->tlb_epoch;
                c->psc[pi].bits = (u32)(rw & 1) | (u32)(us & 1) << 1 | (u32)(nx & 1) << 2;
            }
        }
        if (write && !(rw & 1) && (user || (c->cr0 & CR0_WP))) page_fault(c, la, err | 1);
        if (user && !(us & 1)) page_fault(c, la, err | 1);
        if (acc == 2 && nxe && (nx & 1)) page_fault(c, la, err | 1);
        if (write && !(pte & PTE_D)) pt_set(m, pte_addr, PTE_D, 1);
        c->walk_global = (pte & 0x100) && (c->cr4 & 0x80);   /* G with CR4.PGE */
        return pa;
    }
    /* legacy 32-bit paging */
    la = (u32)la;
    u64 pde_addr = (c->cr3 & 0xfffff000ull) + ((la >> 22) & 1023) * 4;
    u32 pde = pt_read32(m, pde_addr);
    if (!(pde & PTE_P)) page_fault(c, la, err);
    if (!(pde & PTE_A)) pt_set(m, pde_addr, PTE_A, 0);
    if ((pde & PTE_PS) && (c->cr4 & CR4_PSE)) {
        if (write && !(pde & PTE_W) && (user || (c->cr0 & CR0_WP))) page_fault(c, la, err | 1);
        if (user && !(pde & PTE_U)) page_fault(c, la, err | 1);
        if (write && !(pde & PTE_D)) pt_set(m, pde_addr, PTE_D, 0);
        return (pde & 0xffc00000u) | (la & 0x3fffff);
    }
    u64 pte_addr = (pde & 0xfffff000u) + ((la >> 12) & 1023) * 4;
    u32 pte = pt_read32(m, pte_addr);
    if (!(pte & PTE_P)) page_fault(c, la, err);
    u32 rw = pde & pte & PTE_W, us = pde & pte & PTE_U;
    if (write && !rw && (user || (c->cr0 & CR0_WP))) page_fault(c, la, err | 1);
    if (user && !us) page_fault(c, la, err | 1);
    if (!(pte & PTE_A)) pt_set(m, pte_addr, PTE_A, 0);
    if (write && !(pte & PTE_D)) pt_set(m, pte_addr, PTE_D, 0);
    return (pte & 0xfffff000u) | (la & 0xfff);
}

/* Host pointer for a linear address, filling the TLB; NULL for MMIO (pa set). */
static u8 *lin_host(Cpu *c, u64 la, int acc, u64 *pa_out) {
    TlbEntry *t = acc == 2 ? c->tlb_x : acc == 1 ? c->tlb_w : c->tlb_r;
    TlbEntry *e = &t[(u32)(la >> 12) & (TLB_SIZE - 1)];
    u64 tag = (la & ~0xfffull) | c->tlb_mode;
    if (e->tag == tag) return e->host + (la & 0xfff);
    if (acc == 1 && e->tag == (tag | TLB_CODE)) {
        /* write to a page holding translated code: drop its blocks, keep the
         * cached translation (the TLB stays as it would be without the JIT) */
        jit_invalidate_page(c->m, (u64)(uintptr_t)(e->host - c->m->ram));
        e->tag = tag;
        return e->host + (la & 0xfff);
    }
    stat_walks++;
    u64 pa = translate(c, la, acc);
    u8 *p = phys_page(c->m, pa, acc == 1);
    if (pa_out) *pa_out = pa;
    if (!p) return 0;
    if (c->sys_access && c->cpl == 3) {
        /* supervisor-checked translation: do not cache it where user code could hit it */
        if (acc == 1 && c->m->jit && jit_page_is_code(c->m, pa)) jit_invalidate_page(c->m, pa);
        return p + (la & 0xfff);
    }
    /* a write to a page holding translated code drops its blocks first */
    if (acc == 1 && c->m->jit && jit_page_is_code(c->m, pa)) jit_invalidate_page(c->m, pa);
    e->tag = tag;
    e->host = p;
    {
        u32 a = acc == 2 ? 2u : acc == 1 ? 1u : 0u, i = (u32)(la >> 12) & (TLB_SIZE - 1);
        if (c->walk_global) {
            if (!c->tlb_g[a][i] && c->nglist < 3 * TLB_SIZE) { c->tlb_g[a][i] = 1; c->glist[c->nglist++] = (u16)(a * TLB_SIZE + i); }
        } else c->tlb_g[a][i] = 0;
    }
#if UINTPTR_MAX == 0xffffffffu
    e->addend = (u32)(uintptr_t)p - (u32)(la & ~0xfffull);
#endif
    return p + (la & 0xfff);
}

static inline u64 load(const u8 *p, int sz) {
    switch (sz) {
    case 1: return *p;
    case 2: { u16 v; __builtin_memcpy(&v, p, 2); return v; }
    case 4: { u32 v; __builtin_memcpy(&v, p, 4); return v; }
    default: { u64 v; __builtin_memcpy(&v, p, 8); return v; }
    }
}

static inline void store(u8 *p, u64 v, int sz) {
    switch (sz) {
    case 1: *p = (u8)v; break;
    case 2: { u16 x = (u16)v; __builtin_memcpy(p, &x, 2); break; }
    case 4: { u32 x = (u32)v; __builtin_memcpy(p, &x, 4); break; }
    default: __builtin_memcpy(p, &v, 8); break;
    }
}

u64 lread(Cpu *c, u64 la, int sz, int acc) {
    if ((la & 0xfff) + (u32)sz <= 0x1000) {
        TlbEntry *t = acc == 2 ? c->tlb_x : c->tlb_r;
        TlbEntry *e = &t[(u32)(la >> 12) & (TLB_SIZE - 1)];
        if (e->tag == ((la & ~0xfffull) | c->tlb_mode)) return load(e->host + (la & 0xfff), sz);
        u64 pa = 0;
        u8 *p = lin_host(c, la, acc, &pa);
        if (p) return load(p, sz);
        return mmio_read(c, pa, sz);
    }
    u64 v = 0;
    for (int i = 0; i < sz; i++) v |= lread(c, la + i, 1, acc) << (8 * i);
    return v;
}

void lwrite(Cpu *c, u64 la, u64 v, int sz) {
    if ((la & 0xfff) + (u32)sz <= 0x1000) {
        TlbEntry *e = &c->tlb_w[(u32)(la >> 12) & (TLB_SIZE - 1)];
        if (e->tag == ((la & ~0xfffull) | c->tlb_mode)) { store(e->host + (la & 0xfff), v, sz); return; }
        u64 pa = 0;
        u8 *p = lin_host(c, la, 1, &pa);
        if (p) store(p, v, sz);
        else mmio_write(c, pa, v, sz);
        return;
    }
    /* page-crossing: make sure both pages are writable before changing anything */
    u64 pa;
    lin_host(c, la, 1, &pa);
    lin_host(c, la + sz - 1, 1, &pa);
    for (int i = 0; i < sz; i++) lwrite(c, la + i, v >> (8 * i), 1);
}


/* Host pointer for [off, off+len) when it lies in one RAM page (faults like the
 * first access would); NULL when the caller must take the per-access path. */
u8 *host_span(Cpu *c, int seg, u64 off, u32 len, int write) {
    u64 la = seg_lin(c, seg, off);
    if ((la & 0xfff) + len > 0x1000) return 0;
    u64 pa;
    return lin_host(c, la, write, &pa);
}

void vread128(Cpu *c, int seg, u64 off, Xmm *x, int aligned) {
    u64 la = seg_lin(c, seg, off);
    if (aligned && (la & 15)) GP0(c);
    const TlbEntry *e = &c->tlb_r[(u32)(la >> 12) & (TLB_SIZE - 1)];
    if (e->tag == ((la & ~0xfffull) | c->tlb_mode) && (la & 0xfff) <= 0xff0) {
        __builtin_memcpy(x, e->host + (la & 0xfff), 16);
        return;
    }
    x->q[0] = lread(c, la, 8, 0);
    x->q[1] = lread(c, la + 8, 8, 0);
}

void vwrite128(Cpu *c, int seg, u64 off, const Xmm *x, int aligned) {
    u64 la = seg_lin(c, seg, off);
    if (aligned && (la & 15)) GP0(c);
    const TlbEntry *e = &c->tlb_w[(u32)(la >> 12) & (TLB_SIZE - 1)];
    if (e->tag == ((la & ~0xfffull) | c->tlb_mode) && (la & 0xfff) <= 0xff0) {
        __builtin_memcpy(e->host + (la & 0xfff), x, 16);
        return;
    }
    u64 pa;
    lin_host(c, la, 1, &pa);
    lin_host(c, la + 15, 1, &pa);
    lwrite(c, la, x->q[0], 8);
    lwrite(c, la + 8, x->q[1], 8);
}

/* ================= instruction fetch ================= */

void fetch_page(Cpu *c, u64 la) {
    u64 pa = 0;
    u8 *p = lin_host(c, la & ~0xfffull, 2, &pa);
    if (!p) emu_fatal(c, "instruction fetch from unbacked memory %llx", (unsigned long long)pa);
    c->fetch_host = p;
    c->fetch_la = la & ~0xfffull;
}

static u64 fetch64(Cpu *c, Dec *d) { u64 v = fetch32(c, d); return v | (u64)fetch32(c, d) << 32; }

/* immediate of operand size (Iz: 16 or 32, sign-extended to 64) */
static u64 fetch_iz(Cpu *c, Dec *d) {
    if (d->osz == 2) return fetch16(c, d);
    u32 v = fetch32(c, d);
    return d->osz == 8 ? (u64)(i64)(i32)v : v;
}

/* bytes of immediate that follow the ModRM/displacement of d->opc */
static int rip_imm_size(const Dec *d) {
    int op = d->opc, z = d->osz == 2 ? 2 : 4;
    if (op >= 0x300) return 1;
    if (op >= 0x200) return 0;
    if (op >= 0x100) {
        int b = op & 0xff;
        return ((b >= 0x70 && b <= 0x73) || b == 0xa4 || b == 0xac || b == 0xba || b == 0xc2 ||
                b == 0xc4 || b == 0xc5 || b == 0xc6) ? 1 : 0;
    }
    switch (op) {
    case 0x69: case 0x81: case 0xc7: return z;
    case 0x6b: case 0x80: case 0x83: case 0xc0: case 0xc1: case 0xc6: return 1;
    case 0xf6: return (d->reg & 7) < 2 ? 1 : 0;
    case 0xf7: return (d->reg & 7) < 2 ? z : 0;
    }
    return 0;
}

void decode_modrm(Cpu *c, Dec *d) {
    u8 m = fetch8(c, d);
    d->mod = m >> 6;
    d->reg = ((m >> 3) & 7) | (d->rexr << 3);
    int rm = m & 7;
    d->rip_rel = 0;
    if (d->mod == 3) { d->rm = rm | (d->rexb << 3); return; }
    if (d->asz == 2) {
        /* 16-bit forms: BX+SI, BX+DI, BP+SI, BP+DI, SI, DI, BP (disp16 if mod 0), BX */
        static const int base16[8] = { 3, 3, 5, 5, 6, 7, 5, 3 };
        static const int idx16[8] = { 6, 7, 6, 7, -1, -1, -1, -1 };
        u32 ea16 = 0;
        int defseg = (rm == 2 || rm == 3 || rm == 6) ? SS : DS;
        if (d->mod == 0 && rm == 6) { ea16 = fetch16(c, d); defseg = DS; }
        else {
            ea16 = (u32)c->r[base16[rm]];
            if (idx16[rm] >= 0) ea16 += (u32)c->r[idx16[rm]];
        }
        if (d->mod == 1) ea16 += (u32)(i32)(i8)fetch8(c, d);
        else if (d->mod == 2) ea16 += fetch16(c, d);
        d->ea = ea16 & 0xffff;
        d->ea_seg = d->seg >= 0 ? d->seg : defseg;
        return;
    }
    u64 ea = 0;
    int defseg = DS;
    if (rm == 4) {
        u8 sib = fetch8(c, d);
        int idx = ((sib >> 3) & 7) | (d->rexx << 3);
        int base = sib & 7;
        if (idx != 4) ea += c->r[idx] << (sib >> 6);
        if (base == 5 && d->mod == 0) {
            ea += (u64)(i64)(i32)fetch32(c, d);
        } else {
            base |= d->rexb << 3;
            ea += c->r[base];
            if (base == 4 || base == 5) defseg = SS;
        }
    } else if (rm == 5 && d->mod == 0) {
        ea = (u64)(i64)(i32)fetch32(c, d);
        if (c->mode64) d->rip_rel = 1;
    } else {
        int base = rm | (d->rexb << 3);
        ea = c->r[base];
        if (base == 5) defseg = SS;
    }
    if (d->mod == 1) ea += (u64)(i64)(i8)fetch8(c, d);
    else if (d->mod == 2) ea += (u64)(i64)(i32)fetch32(c, d);
    if (d->rip_rel) {
        /* relative to the end of the whole instruction, immediate included,
         * whether or not the handler has fetched the immediate yet */
        ea += d->ip + (u64)rip_imm_size(d);
        d->rip_rel = 0;
    }
    d->ea = ea;
    d->ea_seg = d->seg >= 0 ? d->seg : defseg;
}

/* ================= flags ================= */

static inline u32 parity(u64 v) {
    u32 x = (u32)v & 0xff;
    x ^= x >> 4;
    return ((0x6996u >> (x & 0xf)) & 1) ? 0 : F_PF;
}

static inline void lazy(Cpu *c, u32 op, u64 res, u64 a, u64 b, int sz) {
    c->lf_op = (u16)op; c->lf_res = res; c->lf_a = a; c->lf_b = b; c->lf_size = (u16)sz;
}

static u32 lf_cf(Cpu *c) {
    u64 m = MASK[c->lf_size & 15], ra = c->lf_res & m, aa = c->lf_a & m, ba = c->lf_b & m;
    switch (c->lf_op) {
    case LF_ADD: return ra < aa;
    case LF_ADC: return c->lf_c ? ra <= aa : ra < aa;
    case LF_SUB: return aa < ba;
    case LF_SBB: return c->lf_c ? aa <= ba : aa < ba;
    case LF_INC: case LF_DEC: return c->lf_c;
    case LF_LOGIC: return 0;
    case LF_SHL: {
        u32 bits = c->lf_size * 8, n = (u32)c->lf_b;
        return n <= bits ? (u32)(c->lf_a >> (bits - n)) & 1 : 0;
    }
    case LF_SHR: return (u32)(c->lf_a >> ((u32)c->lf_b - 1)) & 1;
    case LF_MUL: return c->lf_c;
    case LF_SAR: return (u32)(((i64)sext(c->lf_a, (int)c->lf_size)) >> ((u32)c->lf_b - 1)) & 1;
    default: return (u32)(c->rflags & F_CF);
    }
}

static u32 lf_of(Cpu *c) {
    u64 s = SIGNBIT(c->lf_size), a = c->lf_a, b = c->lf_b, r = c->lf_res;
    switch (c->lf_op) {
    case LF_ADD: case LF_ADC: case LF_INC: return (((a ^ r) & (b ^ r)) & s) != 0;
    case LF_SUB: case LF_SBB: case LF_DEC: return (((a ^ b) & (a ^ r)) & s) != 0;
    case LF_LOGIC: case LF_SAR: return 0;
    case LF_SHL: return (u32)((r >> (c->lf_size * 8 - 1)) & 1) ^ lf_cf(c);
    case LF_SHR: return (a & s) != 0;
    case LF_MUL: return c->lf_c;
    default: return (c->rflags & F_OF) != 0;
    }
}

static inline u32 lf_zf(Cpu *c) { return c->lf_op == LF_NONE ? (c->rflags & F_ZF) != 0 : (c->lf_res & MASK[c->lf_size & 15]) == 0; }
static inline u32 lf_sf(Cpu *c) {
    return c->lf_op == LF_NONE ? (c->rflags & F_SF) != 0 : (c->lf_res & SIGNBIT(c->lf_size)) != 0;
}

u64 get_flags(Cpu *c) {
    if (c->lf_op == LF_NONE) return c->rflags | 2;
    u64 f = c->rflags & ~(u64)F_ARITH;
    f |= lf_cf(c) ? F_CF : 0;
    f |= parity(c->lf_res);
    if (c->lf_op != LF_LOGIC && c->lf_op < LF_SHL) f |= (c->lf_a ^ c->lf_b ^ c->lf_res) & F_AF;
    f |= (c->lf_res & MASK[c->lf_size & 15]) == 0 ? F_ZF : 0;
    f |= (c->lf_res & SIGNBIT(c->lf_size)) ? F_SF : 0;
    f |= lf_of(c) ? F_OF : 0;
    f |= 2;
    c->rflags = f;
    c->lf_op = LF_NONE;
    return f;
}

void set_flags_raw(Cpu *c, u64 arith) {
    c->rflags = (get_flags(c) & ~(u64)F_ARITH) | (arith & F_ARITH);
}

static inline u64 szp(u64 r, int sz) {
    return (r == 0 ? F_ZF : 0) | ((r & SIGNBIT(sz)) ? F_SF : 0) | parity(r);
}

static int cond(Cpu *c, int cc) {
    int r;
    switch (cc >> 1) {
    case 0: r = lf_of(c); break;
    case 1: r = lf_cf(c); break;
    case 2: r = lf_zf(c); break;
    case 3: r = lf_cf(c) || lf_zf(c); break;
    case 4: r = lf_sf(c); break;
    case 5: r = (get_flags(c) & F_PF) != 0; break;
    case 6: r = lf_sf(c) != lf_of(c); break;
    default: r = lf_zf(c) || (lf_sf(c) != lf_of(c)); break;
    }
    return r ^ (cc & 1);
}

/* ================= arithmetic ================= */

static u64 alu(Cpu *c, int op, u64 a, u64 b, int sz) {
    u64 m = MASK[sz], r;
    a &= m; b &= m;
    switch (op & 7) {
    case 0: r = (a + b) & m; lazy(c, LF_ADD, r, a, b, sz); break;
    case 1: r = a | b; lazy(c, LF_LOGIC, r, a, b, sz); break;
    case 2: { u32 cf = lf_cf(c); r = (a + b + cf) & m; c->lf_c = cf; lazy(c, LF_ADC, r, a, b, sz); break; }
    case 3: { u32 cf = lf_cf(c); r = (a - b - cf) & m; c->lf_c = cf; lazy(c, LF_SBB, r, a, b, sz); break; }
    case 4: r = a & b; lazy(c, LF_LOGIC, r, a, b, sz); break;
    case 5: case 7: r = (a - b) & m; lazy(c, LF_SUB, r, a, b, sz); break;
    default: r = a ^ b; lazy(c, LF_LOGIC, r, a, b, sz); break;
    }
    return r;
}

static u64 inc_dec(Cpu *c, u64 a, int dec, int sz) {
    u64 m = MASK[sz];
    a &= m;
    u32 cf = lf_cf(c);
    u64 r = (dec ? a - 1 : a + 1) & m;
    c->lf_c = cf;
    lazy(c, dec ? LF_DEC : LF_INC, r, a, 1, sz);
    return r;
}

static void put_flags_cf_of(Cpu *c, u32 cf, u32 of) {
    u64 f = get_flags(c) & ~(u64)(F_CF | F_OF);
    c->rflags = f | (cf ? F_CF : 0) | (of ? F_OF : 0);
}

static void put_flags_full(Cpu *c, u64 r, int sz, u32 cf, u32 of) {
    set_flags_raw(c, szp(r, sz) | (cf ? F_CF : 0) | (of ? F_OF : 0));
}

static u64 shift(Cpu *c, int op, u64 a, u32 cnt, int sz) {
    int bits = sz * 8;
    u64 m = MASK[sz];
    a &= m;
    cnt &= sz == 8 ? 63 : 31;
    if (!cnt) return a;
    u64 r;
    u32 cf, of;
    switch (op & 7) {
    case 0: { /* ROL */
        u32 n = cnt % bits;
        r = n ? ((a << n) | (a >> (bits - n))) & m : a;
        cf = r & 1; of = ((r >> (bits - 1)) & 1) ^ cf;
        put_flags_cf_of(c, cf, of);
        return r;
    }
    case 1: { /* ROR */
        u32 n = cnt % bits;
        r = n ? ((a >> n) | (a << (bits - n))) & m : a;
        cf = (r >> (bits - 1)) & 1; of = cf ^ ((r >> (bits - 2)) & 1);
        put_flags_cf_of(c, cf, of);
        return r;
    }
    case 2: { /* RCL */
        u32 n = cnt % (bits + 1);
        cf = (get_flags(c) & F_CF) != 0;
        r = a;
        for (u32 i = 0; i < n; i++) { u32 nc = (r >> (bits - 1)) & 1; r = ((r << 1) | cf) & m; cf = nc; }
        of = ((r >> (bits - 1)) & 1) ^ cf;
        put_flags_cf_of(c, cf, of);
        return r;
    }
    case 3: { /* RCR */
        u32 n = cnt % (bits + 1);
        cf = (get_flags(c) & F_CF) != 0;
        r = a;
        of = ((r >> (bits - 1)) & 1) ^ cf;
        for (u32 i = 0; i < n; i++) { u32 nc = r & 1; r = (r >> 1) | ((u64)cf << (bits - 1)); cf = nc; }
        put_flags_cf_of(c, cf, of);
        return r;
    }
    case 4: case 6: r = (a << cnt) & m; lazy(c, LF_SHL, r, a, cnt, sz); return r;
    case 5: r = a >> cnt; lazy(c, LF_SHR, r, a, cnt, sz); return r;
    default: r = (u64)((i64)sext(a, sz) >> cnt) & m; lazy(c, LF_SAR, r, a, cnt, sz); return r;
    }
}

static u64 shld_shrd(Cpu *c, int right, u64 dst, u64 src, u32 cnt, int sz) {
    int bits = sz * 8;
    u64 m = MASK[sz];
    dst &= m; src &= m;
    cnt &= sz == 8 ? 63 : 31;
    if (!cnt) return dst;
    if ((int)cnt > bits) cnt %= bits;   /* 16-bit operands: architecturally undefined */
    if (!cnt) return dst;
    u64 r; u32 cf;
    if (!right) {
        r = ((dst << cnt) | (cnt == (u32)bits ? src : src >> (bits - cnt))) & m;
        cf = (dst >> (bits - cnt)) & 1;
    } else {
        r = ((dst >> cnt) | (cnt == (u32)bits ? src : src << (bits - cnt))) & m;
        cf = (dst >> (cnt - 1)) & 1;
    }
    u32 of = ((r ^ dst) >> (bits - 1)) & 1;
    put_flags_full(c, r, sz, cf, of);
    return r;
}

static void mul64(u64 a, u64 b, u64 *lo, u64 *hi) {
    u64 al = (u32)a, ah = a >> 32, bl = (u32)b, bh = b >> 32;
    u64 p0 = al * bl, p1 = al * bh, p2 = ah * bl, p3 = ah * bh;
    u64 mid = (p0 >> 32) + (u32)p1 + (u32)p2;
    *lo = (mid << 32) | (u32)p0;
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

static void imul64(i64 a, i64 b, u64 *lo, u64 *hi) {
    mul64((u64)a, (u64)b, lo, hi);
    if (a < 0) *hi -= (u64)b;
    if (b < 0) *hi -= (u64)a;
}

/* (hi:lo) / d with hi < d */
static void div128(u64 hi, u64 lo, u64 d, u64 *q, u64 *r) {
    if (!hi) { *q = lo / d; *r = lo % d; return; }
    for (int i = 0; i < 64; i++) {
        u64 top = hi >> 63;
        hi = (hi << 1) | (lo >> 63);
        lo <<= 1;
        if (top || hi >= d) { hi -= d; lo |= 1; }
    }
    *q = lo; *r = hi;
}

/* two/three operand IMUL truncated to sz, with CF/OF */
static u64 imul_trunc(Cpu *c, u64 a, u64 b, int sz) {
    u64 r; u32 of;
    if (sz == 8) {
        u64 lo, hi;
        imul64((i64)a, (i64)b, &lo, &hi);
        r = lo;
        of = hi != (u64)((i64)lo >> 63);
    } else {
        i64 p = (i64)sext(a, sz) * (i64)sext(b, sz);
        r = (u64)p & MASK[sz];
        of = (i64)sext(r, sz) != p;
    }
    lazy(c, LF_MUL, r, 0, 0, sz);
    c->lf_c = of;
    return r;
}

static void muldiv(Cpu *c, Dec *d, int r, u64 src, int sz) {
    u64 m = MASK[sz];
    switch (r) {
    case 4: case 5: {   /* MUL / IMUL */
        int sgn = r == 5;
        u32 of;
        if (sz == 1) {
            u64 al = c->r[0] & 0xff;
            u64 p = sgn ? (u64)((i64)(i8)al * (i64)(i8)src) : al * src;
            c->r[0] = (c->r[0] & ~0xffffull) | (p & 0xffff);
            of = sgn ? (i64)(i8)p != (i64)(i16)p : (p & 0xff00) != 0;
            lazy(c, LF_MUL, p & 0xff, 0, 0, 1); c->lf_c = of;
        } else if (sz == 8) {
            u64 lo, hi;
            if (sgn) imul64((i64)c->r[0], (i64)src, &lo, &hi); else mul64(c->r[0], src, &lo, &hi);
            c->r[0] = lo; c->r[2] = hi;
            of = sgn ? hi != (u64)((i64)lo >> 63) : hi != 0;
            lazy(c, LF_MUL, lo, 0, 0, 8); c->lf_c = of;
        } else {
            u64 a = c->r[0] & m;
            u64 p = sgn ? (u64)((i64)sext(a, sz) * (i64)sext(src, sz)) : a * src;
            u64 lo = p & m, hi = (p >> (sz * 8)) & m;
            set_reg(c, d, 0, sz, lo);
            set_reg(c, d, 2, sz, hi);
            of = sgn ? (i64)sext(lo, sz) != (i64)p : hi != 0;
            lazy(c, LF_MUL, lo, 0, 0, sz); c->lf_c = of;
        }
        return;
    }
    case 6: {   /* DIV */
        if (!src) cpu_exc(c, EXC_DE, 0, 0);
        if (sz == 1) {
            u64 n = c->r[0] & 0xffff, q = n / src;
            if (q > 0xff) cpu_exc(c, EXC_DE, 0, 0);
            c->r[0] = (c->r[0] & ~0xffffull) | q | ((n % src) << 8);
        } else if (sz == 8) {
            u64 q, r;
            if (c->r[2] >= src) cpu_exc(c, EXC_DE, 0, 0);
            div128(c->r[2], c->r[0], src, &q, &r);
            c->r[0] = q; c->r[2] = r;
        } else {
            int bits = sz * 8;
            u64 n = ((c->r[2] & m) << bits) | (c->r[0] & m);
            u64 q = n / src;
            if (q > m) cpu_exc(c, EXC_DE, 0, 0);
            set_reg(c, d, 0, sz, q);
            set_reg(c, d, 2, sz, n % src);
        }
        return;
    }
    default: {  /* IDIV */
        if (!src) cpu_exc(c, EXC_DE, 0, 0);
        if (sz == 8) {
            i64 dv = (i64)src;
            u64 hi = c->r[2], lo = c->r[0];
            int neg = (i64)hi < 0;
            if (neg) { lo = ~lo + 1; hi = ~hi + (lo == 0); }
            u64 ud = dv < 0 ? (u64)0 - (u64)dv : (u64)dv;
            if (hi >= ud) cpu_exc(c, EXC_DE, 0, 0);
            u64 q, r;
            div128(hi, lo, ud, &q, &r);
            int qneg = neg ^ (dv < 0);
            if (qneg ? q > 0x8000000000000000ull : q > 0x7fffffffffffffffull) cpu_exc(c, EXC_DE, 0, 0);
            c->r[0] = qneg ? (u64)0 - q : q;
            c->r[2] = neg ? (u64)0 - r : r;
        } else {
            int bits = sz * 8;
            i64 n;
            if (sz == 1) n = (i16)(c->r[0] & 0xffff);
            else n = (i64)sext(((c->r[2] & m) << bits) | (c->r[0] & m), sz * 2);
            i64 dv = (i64)sext(src, sz);
            if (n == (i64)0x8000000000000000ull && dv == -1) cpu_exc(c, EXC_DE, 0, 0);
            i64 q = n / dv, r = n % dv;
            i64 lim = (i64)1 << (bits - 1);
            if (q >= lim || q < -lim) cpu_exc(c, EXC_DE, 0, 0);
            if (sz == 1) c->r[0] = (c->r[0] & ~0xffffull) | ((u64)q & 0xff) | (((u64)r & 0xff) << 8);
            else { set_reg(c, d, 0, sz, (u64)q); set_reg(c, d, 2, sz, (u64)r); }
        }
        return;
    }
    }
}

static void group3(Cpu *c, Dec *d, int sz) {
    u64 m = MASK[sz];
    switch (d->reg & 7) {
    case 0: case 1: {
        u64 imm = sz == 1 ? fetch8(c, d) : fetch_iz(c, d);
        alu(c, 4, rd_rm(c, d, sz), imm, sz);
        return;
    }
    case 2: wr_rm(c, d, sz, ~rd_rm(c, d, sz) & m); return;
    case 3: { u64 a = rd_rm(c, d, sz); u64 r = alu(c, 5, 0, a, sz); wr_rm(c, d, sz, r); return; }
    default: muldiv(c, d, d->reg & 7, rd_rm(c, d, sz), sz); return;
    }
}


/* ================= stack ================= */

/* stack pointer width: 64 in long mode, else SS.B selects ESP or SP */
static inline u64 sp_mask(Cpu *c) {
    if (c->mode64) return ~0ull;
    return (c->seg[SS].attr & 0x4000) ? 0xffffffffull : 0xffffull;
}

static inline void set_sp(Cpu *c, u64 v) {
    u64 m = sp_mask(c);
    c->r[4] = m == 0xffffull ? (c->r[4] & ~0xffffull) | (v & 0xffff) : (v & m);
}

static inline void push(Cpu *c, u64 v, int sz) {
    u64 sp = (c->r[4] - (u64)sz) & sp_mask(c);
    vwrite(c, SS, sp, v, sz);
    set_sp(c, sp);
}

static inline u64 pop(Cpu *c, int sz) {
    u64 sp = c->r[4] & sp_mask(c);
    u64 v = vread(c, SS, sp, sz);
    set_sp(c, sp + (u64)sz);
    return v;
}

/* write an address-size register (string/loop counters): 16-bit forms keep the upper bits */
static inline void set_areg(Cpu *c, int n, u64 v, int asz) {
    if (asz == 8) c->r[n] = v;
    else if (asz == 4) c->r[n] = (u32)v;
    else c->r[n] = (c->r[n] & ~0xffffull) | (v & 0xffff);
}

static inline int stack_sz(Cpu *c, Dec *d) {
    if (c->mode64) return d->osz == 2 ? 2 : 8;
    return d->osz;
}

/* ================= segments & modes ================= */

static void update_mode(Cpu *c) {
    u16 a = c->seg[CS].attr;
    c->mode64 = (c->efer & EFER_LMA) && (a & 0x2000);
    c->def32 = c->mode64 ? 1 : (a & 0x4000) != 0;
    c->tlb_mode = (c->cpl == 3 ? 2u : 1u) | c->tlb_epoch << 4;   /* tagged with the privilege level */
}

static u64 sys_read(Cpu *c, u64 la, int sz) {
    c->sys_access = 1;
    u64 v = lread(c, la, sz, 0);
    c->sys_access = 0;
    return v;
}

static void sys_write(Cpu *c, u64 la, u64 v, int sz) {
    c->sys_access = 1;
    lwrite(c, la, v, sz);
    c->sys_access = 0;
}

static u64 read_desc(Cpu *c, u16 sel, int err_vec) {
    u64 base = (sel & 4) ? c->ldtr.base : c->gdtr.base;
    u32 limit = (sel & 4) ? c->ldtr.limit : c->gdtr.limit;
    u32 idx = sel & ~7u;
    if (idx + 7 > limit) cpu_exc(c, err_vec, 1, sel & ~3u);
    return sys_read(c, base + idx, 8);
}

static void load_seg(Cpu *c, int s, u16 sel) {
    if (!(c->cr0 & CR0_PE) || (c->rflags & 0x20000)) {
        /* real mode: base = selector * 16; limits and other attributes stay cached */
        Seg *g = &c->seg[s];
        g->sel = sel;
        g->base = (u64)sel << 4;
        if (s == CS) {
            g->attr = (u16)((g->attr & ~0x6000u) | 0x9b);   /* 16-bit code */
            c->cpl = 0;
            update_mode(c);
        }
        return;
    }
    if ((sel & ~3u) == 0) {
        if (s == CS || (s == SS && !c->mode64)) GP0(c);
        c->seg[s].sel = sel;
        c->seg[s].base = 0;
        c->seg[s].limit = 0;
        c->seg[s].attr = 0;
        return;
    }
    u64 d = read_desc(c, sel, EXC_GP);
    if (!(d & (1ull << 47))) cpu_exc(c, s == SS ? EXC_SS : EXC_NP, 1, sel & ~3u);
    Seg *g = &c->seg[s];
    g->sel = sel;
    g->base = ((d >> 16) & 0xffffff) | ((d >> 32) & 0xff000000ull);
    g->limit = (u32)((d & 0xffff) | ((d >> 32) & 0xf0000));
    g->attr = (u16)((d >> 40) & 0xf0ff);
    if (g->attr & 0x8000) g->limit = (g->limit << 12) | 0xfff;
    if (s == CS) {
        int ncpl = sel & 3;
        c->cpl = ncpl;
        update_mode(c);
    }
}

static void load_system_seg(Cpu *c, Seg *g, u16 sel) {
    if ((sel & ~3u) == 0) { g->sel = 0; g->base = 0; g->limit = 0; g->attr = 0; return; }
    u64 d = read_desc(c, sel, EXC_GP);
    g->sel = sel;
    g->base = ((d >> 16) & 0xffffff) | ((d >> 32) & 0xff000000ull);
    if (c->efer & EFER_LMA) {
        u64 hi = read_desc(c, (u16)(sel + 8), EXC_GP);
        g->base |= (hi & 0xffffffffull) << 32;
    }
    g->limit = (u32)((d & 0xffff) | ((d >> 32) & 0xf0000));
    g->attr = (u16)((d >> 40) & 0xf0ff);
    if (g->attr & 0x8000) g->limit = (g->limit << 12) | 0xfff;
}

/* cpl: privilege level of the instruction doing the write (IRET checks the CPL before it returns) */
static void set_rflags_at(Cpu *c, u64 v, int sz, int cpl) {
    u64 keep_mask = sz == 2 ? ~0xffffull : 0;
    u64 changeable = F_ARITH | F_TF | F_DF | 0x4000 /*NT*/ | 0x40000 /*AC*/ | 0x200000 /*ID*/;
    if (cpl == 0) changeable |= F_IF | 0x3000;
    u64 cur = get_flags(c);
    u64 nv = (cur & ~(changeable & ~keep_mask)) | (v & changeable & ~keep_mask);
    c->rflags = nv | 2;
    c->lf_op = LF_NONE;
}

static void set_rflags(Cpu *c, u64 v, int sz) { set_rflags_at(c, v, sz, c->cpl); }

static void write_cr0(Cpu *c, u64 v) {
    v |= CR0_ET;
    if ((v & CR0_PG) && !(v & CR0_PE)) GP0(c);
    if ((v & CR0_PG) && !(c->cr0 & CR0_PG) && (c->efer & EFER_LME)) {
        if (!(c->cr4 & CR4_PAE)) GP0(c);
        c->efer |= EFER_LMA;
    }
    if (!(v & CR0_PG) && (c->cr0 & CR0_PG)) {
        if (c->mode64) GP0(c);
        c->efer &= ~EFER_LMA;
    }
    if (c->m->jit && ((c->cr0 ^ v) & 0xc)) jit_flush_code(c->m);
    u64 old = c->cr0;
    c->cr0 = v;
    /* only paging-related bits invalidate translations (TS toggles are frequent) */
    if ((old ^ v) & (CR0_PG | CR0_PE | CR0_WP | (1ull << 30))) tlb_flush(c);
    update_mode(c);
}

/* ================= interrupts ================= */

/* recent interrupt/exception history, dumped when the CPU dies */
typedef struct { u64 icount, rip, cr2, rsp; u32 vec, err, kind, cpl; } EvRec;
static EvRec g_ev[256];
static u32 g_ev_n;

/* debugging: base of the PE image containing va (scans down for MZ/PE headers), 0 if none */
u64 cpu_image_base(Cpu *c, u64 va);
u64 cpu_image_base(Cpu *c, u64 va) {
    void *saved[5];
    e_memcpy(saved, c->jmp, sizeof saved);
    u64 found = 0;
    for (u64 a = va & ~0xfffull, n = 0; n < 16384 && !found; n++, a -= 0x1000) {
        u64 pa; u8 *p = 0;
        if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, a, 0, &pa);
        if (!p || p[0] != 'M' || p[1] != 'Z') continue;
        u32 lfa; e_memcpy(&lfa, p + 0x3c, 4);
        if (lfa < 0x1000 - 4 && p[lfa] == 'P' && p[lfa + 1] == 'E') found = a;
    }
    e_memcpy(c->jmp, saved, sizeof saved);
    return found;
}

u64 cpu_peek64(Cpu *c, u64 va, int *ok);
u64 cpu_peek64(Cpu *c, u64 va, int *ok) {
    void *saved[5];
    e_memcpy(saved, c->jmp, sizeof saved);
    u64 v = 0, pa; u8 *p = 0;
    if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, va, 0, &pa);
    e_memcpy(c->jmp, saved, sizeof saved);
    if (p) e_memcpy(&v, p, 8);
    *ok = p != 0;
    return v;
}

void cpu_dump_events(Cpu *c);
void cpu_dump_events(Cpu *c) {
    u32 n = g_ev_n < 256 ? g_ev_n : 256;
    emu_log(c->m, "last %u interrupts/exceptions (oldest first):", n);
    for (u32 i = g_ev_n - n; i < g_ev_n; i++) {
        EvRec *e = &g_ev[i & 255];
        emu_log(c->m, "  #%llu cpl%u %s vec %u err %x rip %llx rsp %llx cr2 %llx", (unsigned long long)e->icount, e->cpl,
                e->kind == 0 ? "exc" : e->kind == 1 ? "int" : "irq", e->vec, e->err,
                (unsigned long long)e->rip, (unsigned long long)e->rsp, (unsigned long long)e->cr2);
    }
}

static void do_interrupt(Cpu *c, int vec, u64 ret_ip, int has_err, u32 err, int soft) {
    {
        EvRec *e = &g_ev[g_ev_n++ & 255];
        e->icount = c->icount; e->rip = ret_ip; e->cr2 = c->cr2; e->rsp = c->r[4]; e->cpl = (u32)c->cpl;
        e->vec = (u32)vec; e->err = has_err ? err : 0; e->kind = soft ? 1 : (ret_ip == c->rip && !c->delivering) ? 2 : 0;
        if (c->delivering) e->kind = 0;
    }
    c->halted = 0;
    u64 fl = get_flags(c);
    int gate_type;
    if (c->efer & EFER_LMA) {
        u32 off = (u32)vec * 16;
        if (off + 15 > c->idtr.limit) cpu_exc(c, EXC_GP, 1, (u32)vec * 8 + 2);
        u64 lo = sys_read(c, c->idtr.base + off, 8);
        u64 hi = sys_read(c, c->idtr.base + off + 8, 8);
        int dpl = (lo >> 45) & 3;
        gate_type = (lo >> 40) & 0xf;
        if (!((lo >> 47) & 1)) cpu_exc(c, EXC_NP, 1, (u32)vec * 8 + 2);
        if (gate_type != 0xe && gate_type != 0xf) cpu_exc(c, EXC_GP, 1, (u32)vec * 8 + 2);
        if (soft && dpl < c->cpl) cpu_exc(c, EXC_GP, 1, (u32)vec * 8 + 2);
        u16 sel = (u16)(lo >> 16);
        u64 target = (lo & 0xffff) | ((lo >> 32) & 0xffff0000ull) | (hi << 32);
        int ist = (lo >> 32) & 7;
        u64 desc = read_desc(c, sel, EXC_GP);
        int new_cpl = ((desc >> 42) & 1) ? c->cpl : (int)((desc >> 45) & 3);   /* conforming keeps CPL */
        u64 rsp = c->r[4];
        if (new_cpl < c->cpl) rsp = sys_read(c, c->tr.base + 4 + 8 * (u64)new_cpl, 8);
        if (ist) rsp = sys_read(c, c->tr.base + 0x24 + 8 * (u64)(ist - 1), 8);
        rsp &= ~15ull;
        u64 frame[6];
        int n = 0;
        frame[n++] = c->seg[SS].sel;
        frame[n++] = c->r[4];
        frame[n++] = fl;
        frame[n++] = c->seg[CS].sel;
        frame[n++] = ret_ip;
        if (has_err) frame[n++] = err;
        u64 pa;
        c->sys_access = 1;
        lin_host(c, rsp - 8 * (u64)n, 1, &pa);      /* probe before writing anything */
        lin_host(c, rsp - 8, 1, &pa);
        c->sys_access = 0;
        for (int i = 0; i < n; i++) { rsp -= 8; sys_write(c, rsp, frame[i], 8); }
        int old_cpl = c->cpl;
        load_seg(c, CS, (u16)((sel & ~3u) | (u32)new_cpl));
        if (new_cpl != old_cpl) {
            c->seg[SS].sel = (u16)new_cpl; c->seg[SS].base = 0; c->seg[SS].attr = 0;
        }
        c->r[4] = rsp;
        c->rip = target;
    } else if (c->cr0 & CR0_PE) {
        u32 off = (u32)vec * 8;
        if (off + 7 > c->idtr.limit) cpu_exc(c, EXC_GP, 1, (u32)vec * 8 + 2);
        u64 g = lread(c, c->idtr.base + off, 8, 0);
        int dpl = (g >> 45) & 3;
        gate_type = (g >> 40) & 0xf;
        if (!((g >> 47) & 1)) cpu_exc(c, EXC_NP, 1, (u32)vec * 8 + 2);
        if (soft && dpl < c->cpl) cpu_exc(c, EXC_GP, 1, (u32)vec * 8 + 2);
        if (gate_type != 0xe && gate_type != 0xf)
            emu_fatal(c, "protected-mode gate type %d is not implemented", gate_type);
        u16 sel = (u16)(g >> 16);
        u32 target = (u32)((g & 0xffff) | ((g >> 32) & 0xffff0000ull));
        u64 desc = read_desc(c, sel, EXC_GP);
        if ((int)((desc >> 45) & 3) != c->cpl) emu_fatal(c, "protected-mode privilege change is not implemented");
        push(c, (u32)fl, 4);
        push(c, c->seg[CS].sel, 4);
        push(c, (u32)ret_ip, 4);
        if (has_err) push(c, err, 4);
        load_seg(c, CS, sel);
        c->rip = target;
    } else {
        /* real mode: IVT entry offset:segment, push FLAGS, CS, IP */
        if (c->m->trace_bios == 1 && soft && vec == 0x10 && ((c->r[0] >> 8) & 0xff) == 0x4f)
            emu_log(c->m, "INT10 VBE ax=%x bx=%x cx=%x es:di=%x:%x", (u32)c->r[0] & 0xffff, (u32)c->r[3] & 0xffff,
                    (u32)c->r[1] & 0xffff, c->seg[ES].sel, (u32)c->r[7] & 0xffff);
        if (c->m->trace_bios == 1 && soft && (vec == 0x13 || vec == 0x15)) {
            u32 ax = (u32)c->r[0] & 0xffff;
            if (vec == 0x13 && (ax >> 8) == 0x42) {
                u64 dap = c->seg[DS].base + (c->r[6] & 0xffff);
                emu_log(c->m, "INT13 42 dl=%x count=%u lba=%llu", (u32)c->r[2] & 0xff,
                        (u32)lread(c, dap + 2, 2, 0), (unsigned long long)lread(c, dap + 8, 8, 0));
            } else {
                emu_log(c->m, "INT%x ax=%x bx=%x cx=%x dx=%x", vec, ax, (u32)c->r[3] & 0xffff,
                        (u32)c->r[1] & 0xffff, (u32)c->r[2] & 0xffff);
            }
        }
        u32 off = (u32)vec * 4;
        if (off + 3 > c->idtr.limit) emu_fatal(c, "real-mode interrupt %d beyond IVT limit", vec);
        u32 ent = (u32)lread(c, c->idtr.base + off, 4, 0);
        push(c, fl & 0xffff, 2);
        push(c, c->seg[CS].sel, 2);
        push(c, ret_ip & 0xffff, 2);
        load_seg(c, CS, (u16)(ent >> 16));
        c->rip = ent & 0xffff;
        gate_type = 0xe;
        c->rflags &= ~(u64)0x40000;   /* AC */
    }
    c->rflags &= ~(u64)(F_TF | 0x4000 | 0x10000);
    if (gate_type == 0xe) c->rflags &= ~(u64)F_IF;
}

static void deliver_exception(Cpu *c) {
    if (c->fatal) return;
    int vec = c->exc_vec;
    if (c->delivering) {
        if (c->delivering >= 2 || vec == EXC_DF) {
            cpu_dump_events(c);
            {
                static const char *rn[16] = { "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15" };
                for (int i = 0; i < 16; i += 4)
                    emu_log(c->m, "  %s=%llx %s=%llx %s=%llx %s=%llx", rn[i], (unsigned long long)c->r[i],
                            rn[i + 1], (unsigned long long)c->r[i + 1], rn[i + 2], (unsigned long long)c->r[i + 2],
                            rn[i + 3], (unsigned long long)c->r[i + 3]);
                c->delivering = 3;   /* reads below must not recurse into exception delivery */
                void *saved_jmp[5];
                e_memcpy(saved_jmp, c->jmp, sizeof saved_jmp);
                char line[160]; u32 n = 0;
                for (int i = -32; i < 16 && n + 4 < sizeof line; i++) {
                    u64 la = c->rip + (u64)(i64)i;
                    u64 pa = 0;
                    u8 *p = 0;
                    if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, la, 0, &pa);
                    u32 b = p ? *p : 0x100;
                    const char *hx = "0123456789abcdef";
                    if (b > 0xff) { line[n++] = '?'; line[n++] = '?'; }
                    else { line[n++] = hx[b >> 4]; line[n++] = hx[b & 15]; }
                    line[n++] = i == -1 ? '|' : ' ';
                }
                line[n] = 0;
                emu_log(c->m, "  code[rip-32..rip+16]: %s", line);
                for (int i = 0; i < 12; i++) {
                    u64 pa = 0; u8 *p = 0;
                    if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, c->r[4] + 8 * (u64)i, 0, &pa);
                    u64 v = 0; if (p) e_memcpy(&v, p, 8);
                    emu_log(c->m, "  [rsp+%02x] %llx", 8 * i, (unsigned long long)v);
                }
                e_memcpy(c->jmp, saved_jmp, sizeof saved_jmp);
            }
            emu_fatal(c, "triple fault (vector %d at rip %llx)", vec, (unsigned long long)c->rip);
            return;
        }
        c->delivering = 2;
        c->exc_vec = EXC_DF; c->exc_has_err = 1; c->exc_err = 0;
        do_interrupt(c, EXC_DF, c->rip, 1, 0, 0);
        c->delivering = 0;
        return;
    }
    c->delivering = 1;
    do_interrupt(c, vec, c->rip, c->exc_has_err, c->exc_err, 0);
    c->delivering = 0;
}

void cpu_interrupt(Cpu *c, int vector) {
    if (__builtin_setjmp(c->jmp)) { deliver_exception(c); return; }
    do_interrupt(c, vector, c->rip, 0, 0, 0);
}

/* ================= MSR / CPUID ================= */

static u64 tsc(Cpu *c) { return machine_now(c->m); }

static u64 *mtrr_slot(Cpu *c, u32 msr) {
    if (msr == 0x2ff) return &c->mtrr_def;
    if (msr >= 0x200 && msr < 0x210) return &c->mtrr_var[msr - 0x200];
    static const u32 fix[11] = { 0x250, 0x258, 0x259, 0x268, 0x269, 0x26a, 0x26b, 0x26c, 0x26d, 0x26e, 0x26f };
    for (int i = 0; i < 11; i++) if (fix[i] == msr) return &c->mtrr_fix[i];
    return 0;
}

static int rdmsr(Cpu *c, u32 msr, u64 *v) {
    if (c->m->chipset) {
        u64 *slot = mtrr_slot(c, msr);
        if (slot) { *v = *slot; return 1; }
        switch (msr) {
        case 0xfe: *v = 0x508; return 1;            /* MTRRCAP: 8 variable, fixed, WC */
        case 0x179: case 0x17a: *v = 0; return 1;   /* MCG_CAP (no banks), MCG_STATUS */
        case 0x3a: *v = 1; return 1;                /* FEATURE_CONTROL: locked, no VMX */
        case 0x8b: *v = 0; return 1;                /* microcode revision */
        /* ARCH_CAPABILITIES: no speculative execution here, so RDCL_NO, SSB_NO, MDS_NO, TAA_NO
         * (also spares guests their Meltdown mitigation, e.g. Windows' KVA shadow) */
        case 0x10a: *v = 0x131; return 1;
        case 0x17: *v = 0; return 1;                /* platform ID */
        }
    }
    switch (msr) {
    case 0x10: *v = tsc(c); return 1;
    case 0x1b: *v = c->apic_base; return 1;
    case 0x174: *v = c->sysenter_cs; return 1;
    case 0x175: *v = c->sysenter_esp; return 1;
    case 0x176: *v = c->sysenter_eip; return 1;
    case 0x1a0: *v = c->misc_enable; return 1;
    case 0x277: *v = c->pat; return 1;
    case 0xc0000080: *v = c->efer; return 1;
    case 0xc0000081: *v = c->star; return 1;
    case 0xc0000082: *v = c->lstar; return 1;
    case 0xc0000083: *v = c->cstar; return 1;
    case 0xc0000084: *v = c->sfmask; return 1;
    case 0xc0000100: *v = c->seg[FS].base; return 1;
    case 0xc0000101: *v = c->seg[GS].base; return 1;
    case 0xc0000102: *v = c->gs_kernel; return 1;
    case 0xc0000103: *v = c->tsc_aux; return 1;
    }
    return 0;
}

static int wrmsr(Cpu *c, u32 msr, u64 v) {
    if (c->m->chipset) {
        u64 *slot = mtrr_slot(c, msr);
        if (slot) { *slot = v; return 1; }
        switch (msr) {
        case 0x17a: case 0x8b: case 0x79: return 1;   /* MCG_STATUS, microcode update trigger */
        }
    }
    switch (msr) {
    case 0x10: return 1;
    case 0x1b: c->apic_base = v; return 1;
    case 0x174: c->sysenter_cs = v; return 1;
    case 0x175: c->sysenter_esp = v; return 1;
    case 0x176: c->sysenter_eip = v; return 1;
    case 0x1a0: c->misc_enable = v; return 1;
    case 0x277: c->pat = v; return 1;
    case 0xc0000080:
        if (v & ~(EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE)) return 0;
        c->efer = (v & ~EFER_LMA) | (c->efer & EFER_LMA);
        tlb_flush(c);
        return 1;
    case 0xc0000081: c->star = v; return 1;
    case 0xc0000082: c->lstar = v; return 1;
    case 0xc0000083: c->cstar = v; return 1;
    case 0xc0000084: c->sfmask = v; return 1;
    case 0xc0000100: c->seg[FS].base = v; return 1;
    case 0xc0000101: c->seg[GS].base = v; return 1;
    case 0xc0000102: c->gs_kernel = v; return 1;
    case 0xc0000103: c->tsc_aux = v; return 1;
    }
    return 0;
}

static void cpuid(Cpu *c) {
    u32 leaf = (u32)c->r[0], a = 0, b = 0, cc = 0, d = 0;
    static const char brand[48] = "W32Eleven x86-64 interpreter";
    switch (leaf) {
    case 0:
        a = 1;
        b = 0x45323357; d = 0x65766c65; cc = 0x756d456e;   /* "W32Eleven" "Emu" */
        /* firmware boots run OSes that only accept known vendors (Windows: Intel/AMD/Centaur) */
        if (c->m->chipset) { b = 0x756e6547; d = 0x49656e69; cc = 0x6c65746e; a = 7; }   /* "GenuineIntel", leaves 0-7 */
        break;
    case 1:
        a = 0x000006f1;
        b = 8u << 8;                        /* CLFLUSH line 64 bytes */
        cc = 0;
        d = (1u << 0) | (1u << 3) | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 8) | (1u << 13) |
            (1u << 15) | (1u << 19) | (1u << 24) | (1u << 25) | (1u << 26);
        /* FPU PSE TSC MSR PAE CX8 PGE CMOV CLFSH FXSR SSE SSE2 (+ APIC on firmware boots) */
        if (c->m->chipset) {
            d |= 1u << 9; b |= 1u << 16;
            if (c->m->ncpu > 1) { b = (b & 0xffffu) | (u32)c->m->ncpu << 16 | (u32)c->cpu_index << 24; d |= 1u << 28; }   /* HTT: n logical CPUs, APIC ID */
            /* DE, MCE, SEP, MTRR, MCA, PAT, PSE36, MMX */
            d |= (1u << 2) | (1u << 7) | (1u << 11) | (1u << 12) | (1u << 14) | (1u << 16) | (1u << 17) | (1u << 23);
            /* x86-64-v2: SSE3, SSSE3, CX16, SSE4.1, SSE4.2, POPCNT */
            cc |= (1u << 0) | (1u << 9) | (1u << 13) | (1u << 19) | (1u << 20) | (1u << 23);
        }
        break;
    case 4:   /* cache topology (SMP only): one core per CPU, all in one package */
        if (c->m->chipset && c->m->ncpu > 1 && (u32)c->r[1] == 0) { a = (u32)(c->m->ncpu - 1) << 26 | 1u << 5 | 1u; b = 63; cc = 63; }
        break;
    case 6:   /* thermal/power: ARAT, the local APIC timer keeps running in every C-state */
        if (c->m->chipset) a = 1u << 2;
        break;
    case 7:   /* structured features: only IA32_ARCH_CAPABILITIES (EDX bit 29) */
        if (c->m->chipset && (u32)c->r[1] == 0) d = 1u << 29;
        break;
    case 0x80000000: a = 0x80000008; break;
    case 0x80000001: cc = 1u; d = (1u << 20) | (1u << 29);           /* LAHF64, NX, LM */
        if (c->m->chipset) { cc |= 1u << 8; d |= (1u << 11) | (1u << 27); }   /* PREFETCHW; SYSCALL, RDTSCP */
        break;
    case 0x80000002: case 0x80000003: case 0x80000004: {
        const u32 *p = (const u32 *)(const void *)(brand + 16 * (leaf - 0x80000002));
        a = p[0]; b = p[1]; cc = p[2]; d = p[3];
        break;
    }
    case 0x80000008: a = 0x3024; break;
    default: break;
    }
    c->r[0] = a; c->r[3] = b; c->r[1] = cc; c->r[2] = d;
}

/* ================= string instructions ================= */

static void string_op(Cpu *c, Dec *d, u8 b) {
    int sz;
    if (b >= 0x6c && b <= 0x6f) sz = (b & 1) ? (d->osz == 8 ? 4 : d->osz) : 1;
    else sz = (b & 1) ? d->osz : 1;
    u64 am = d->asz == 8 ? ~0ull : d->asz == 4 ? 0xffffffffull : 0xffffull;
    int down = (c->rflags & F_DF) != 0;
    u64 step = down ? (u64)0 - (u64)sz : (u64)sz;
    int sseg = d->seg >= 0 ? d->seg : DS;
    int budget = 1 << 16;
    for (;;) {
        if (d->rep) {
            if ((c->r[1] & am) == 0) return;
            if (budget <= 0) { d->ip = d->start; return; }
        }
        u64 si = c->r[6] & am, di = c->r[7] & am;
        /* bulk fast path for rep movs / rep stos */
        if (d->rep && !down && (b == 0xa4 || b == 0xa5 || b == 0xaa || b == 0xab)) {
            u64 cnt = c->r[1] & am;
            u64 dla = seg_lin(c, ES, di);
            u64 room = (0x1000 - (dla & 0xfff)) / (u64)sz;
            u64 sla = 0;
            if (b <= 0xa5) {
                sla = seg_lin(c, sseg, si);
                u64 r2 = (0x1000 - (sla & 0xfff)) / (u64)sz;
                if (r2 < room) room = r2;
            }
            if (room > cnt) room = cnt;
            if (room > (u64)budget) room = (u64)budget;
            if (room >= 2 && (d->asz == 8 || (si + room * sz <= am && di + room * sz <= am))) {
                u64 pa = 0;
                u8 *dp = lin_host(c, dla, 1, &pa);
                u8 *sp = 0;
                if (b <= 0xa5) sp = lin_host(c, sla, 0, &pa);
                if (dp && (b >= 0xaa || sp)) {
                    u32 bytes = (u32)(room * (u64)sz);
                    if (b >= 0xaa) {
                        u64 v = c->r[0];
                        if (sz == 1) e_memset(dp, (int)(v & 0xff), bytes);
                        else for (u32 i = 0; i < bytes; i += (u32)sz) store(dp + i, v, sz);
                    } else if (dp > sp && dp < sp + bytes) {
                        for (u32 i = 0; i < bytes; i++) dp[i] = sp[i];
                    } else {
                        e_memmove(dp, sp, bytes);   /* may overlap with dp < sp: forward copy == memmove */
                    }
                    if (b <= 0xa5) set_areg(c, 6, si + room * sz, d->asz);
                    set_areg(c, 7, di + room * sz, d->asz);
                    set_areg(c, 1, (c->r[1] & am) - room, d->asz);
                    budget -= (int)room;
                    continue;
                }
            }
        }
        switch (b) {
        case 0xa4: case 0xa5: {
            u64 v = vread(c, sseg, si, sz);
            vwrite(c, ES, di, v, sz);
            set_areg(c, 6, si + step, d->asz); set_areg(c, 7, di + step, d->asz);
            break;
        }
        case 0xa6: case 0xa7: {
            u64 x = vread(c, sseg, si, sz), y = vread(c, ES, di, sz);
            alu(c, 7, x, y, sz);
            set_areg(c, 6, si + step, d->asz); set_areg(c, 7, di + step, d->asz);
            break;
        }
        case 0xaa: case 0xab:
            vwrite(c, ES, di, c->r[0], sz);
            set_areg(c, 7, di + step, d->asz);
            break;
        case 0xac: case 0xad:
            set_reg(c, d, 0, sz, vread(c, sseg, si, sz));
            set_areg(c, 6, si + step, d->asz);
            break;
        case 0xae: case 0xaf:
            alu(c, 7, c->r[0], vread(c, ES, di, sz), sz);
            set_areg(c, 7, di + step, d->asz);
            break;
        case 0x6c: case 0x6d:
            if (d->rep && !down && c->m->chipset) {
                /* bulk path for disk PIO: copy straight from the drive buffer into the page */
                u64 dla = seg_lin(c, ES, di), pa;
                u64 cnt = c->r[1] & am;
                u32 room = (u32)(0x1000 - (dla & 0xfff));
                if ((u64)room > cnt * (u64)sz) room = (u32)(cnt * (u64)sz);
                room -= room % (u32)sz;
                if (room >= 2 * (u32)sz && (d->asz == 8 || di + room <= am)) {
                    u8 *hp = lin_host(c, dla, 1, &pa);
                    u32 got = hp ? ide_pio_read_bulk(c->m, (u16)c->r[2], hp, room, sz) : 0;
                    if (got) {
                        set_areg(c, 7, di + got, d->asz);
                        set_areg(c, 1, (c->r[1] & am) - got / (u32)sz, d->asz);
                        budget -= (int)(got / (u32)sz);
                        continue;
                    }
                }
            }
            vwrite(c, ES, di, io_in(c->m, (u16)c->r[2], sz), sz);
            set_areg(c, 7, di + step, d->asz);
            break;
        default:
            io_out(c->m, (u16)c->r[2], (u32)vread(c, sseg, si, sz), sz);
            set_areg(c, 6, si + step, d->asz);
            break;
        }
        if (!d->rep) return;
        set_areg(c, 1, (c->r[1] & am) - 1, d->asz);
        budget--;
        if (b == 0xa6 || b == 0xa7 || b == 0xae || b == 0xaf) {
            int zf = lf_zf(c);
            if (d->rep == 0xf3 && !zf) return;
            if (d->rep == 0xf2 && zf) return;
        }
    }
}

/* ================= x87 (minimal) ================= */


/* ================= main dispatch ================= */

static void jcc_or_jmp(Cpu *c, Dec *d, u64 rel) { (void)c; d->ip += rel; }

static void exec_0f(Cpu *c, Dec *d);

__attribute__((noreturn)) static void unimpl(Cpu *c, Dec *d, int two, u8 b) {
    emu_log(c->m, "#UD: opcode %s%02x at %llx", two ? "0f " : "", b, (unsigned long long)d->start);
    UD(c);
}

static void exec(Cpu *c, Dec *d, u8 b) {
    int osz = d->osz;
    switch (b) {
    /* ALU */
    case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05:
    case 0x08: case 0x09: case 0x0a: case 0x0b: case 0x0c: case 0x0d:
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
    case 0x18: case 0x19: case 0x1a: case 0x1b: case 0x1c: case 0x1d:
    case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25:
    case 0x28: case 0x29: case 0x2a: case 0x2b: case 0x2c: case 0x2d:
    case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35:
    case 0x38: case 0x39: case 0x3a: case 0x3b: case 0x3c: case 0x3d: {
        int op = b >> 3, form = b & 7, sz = (form & 1) ? osz : 1;
        if (form < 4) {
            decode_modrm(c, d);
            if (form < 2) {
                u64 r = alu(c, op, rd_rm(c, d, sz), get_reg(c, d, d->reg, sz), sz);
                if (op != 7) wr_rm(c, d, sz, r);
            } else {
                u64 r = alu(c, op, get_reg(c, d, d->reg, sz), rd_rm(c, d, sz), sz);
                if (op != 7) set_reg(c, d, d->reg, sz, r);
            }
        } else {
            u64 imm = sz == 1 ? fetch8(c, d) : fetch_iz(c, d);
            u64 r = alu(c, op, get_reg(c, d, 0, sz), imm, sz);
            if (op != 7) set_reg(c, d, 0, sz, r);
        }
        return;
    }
    case 0x06: case 0x0e: case 0x16: case 0x1e:
        if (c->mode64) unimpl(c, d, 0, b);
        push(c, c->seg[b >> 3].sel, osz);
        return;
    case 0x07: case 0x17: case 0x1f: {
        if (c->mode64) unimpl(c, d, 0, b);
        u16 sel = (u16)pop(c, osz);
        load_seg(c, b >> 3, sel);
        if (b == 0x17) c->shadow = 1;
        return;
    }
    case 0x0f: exec_0f(c, d); return;
    case 0x27: case 0x2f: {   /* DAA / DAS */
        if (c->mode64) unimpl(c, d, 0, b);
        u64 f = get_flags(c);
        u32 al = (u32)c->r[0] & 0xff, old_al = al, old_cf = f & F_CF, cf = 0, af = 0;
        if ((al & 15) > 9 || (f & F_AF)) {
            if (b == 0x27) { cf = old_cf || al + 6 > 0xff; al = (al + 6) & 0xff; }
            else { cf = old_cf || al < 6; al = (al - 6) & 0xff; }
            af = 1;
        }
        if (old_al > 0x99 || old_cf) { al = (b == 0x27 ? al + 0x60 : al - 0x60) & 0xff; cf = 1; }
        c->r[0] = (c->r[0] & ~0xffull) | al;
        set_flags_raw(c, szp(al, 1) | (cf ? F_CF : 0) | (af ? F_AF : 0));
        return;
    }
    case 0x37: case 0x3f: {   /* AAA / AAS */
        if (c->mode64) unimpl(c, d, 0, b);
        u64 f = get_flags(c);
        u32 ax = (u32)c->r[0] & 0xffff, adj = 0;
        if ((ax & 15) > 9 || (f & F_AF)) {
            if (b == 0x37) ax = (ax + 0x106) & 0xffff;
            else { ax = (ax - 6) & 0xffff; ax = (ax & 0xff) | (((ax >> 8) - 1) & 0xff) << 8; }
            adj = 1;
        }
        ax &= 0xff0f;
        c->r[0] = (c->r[0] & ~0xffffull) | ax;
        set_flags_raw(c, (f & (F_SF | F_ZF | F_PF | F_OF)) | (adj ? F_CF | F_AF : 0));
        return;
    }
    case 0xd4: case 0xd5: {   /* AAM / AAD imm8 */
        if (c->mode64) unimpl(c, d, 0, b);
        u32 imm = fetch8(c, d), al = (u32)c->r[0] & 0xff, ah = ((u32)c->r[0] >> 8) & 0xff;
        if (b == 0xd4) {
            if (!imm) cpu_exc(c, EXC_DE, 0, 0);
            ah = al / imm; al = al % imm;
        } else {
            al = (al + ah * imm) & 0xff; ah = 0;
        }
        c->r[0] = (c->r[0] & ~0xffffull) | al | ah << 8;
        set_flags_raw(c, szp(al, 1));
        return;
    }
    case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46: case 0x47:
    case 0x48: case 0x49: case 0x4a: case 0x4b: case 0x4c: case 0x4d: case 0x4e: case 0x4f: {
        /* only reached outside 64-bit mode */
        int n = b & 7;
        set_reg(c, d, n, osz, inc_dec(c, get_reg(c, d, n, osz), b >= 0x48, osz));
        return;
    }
    case 0x50: case 0x51: case 0x52: case 0x53: case 0x54: case 0x55: case 0x56: case 0x57: {
        int n = (b & 7) | (d->rexb << 3), sz = stack_sz(c, d);
        push(c, get_reg(c, d, n, sz), sz);
        return;
    }
    case 0x58: case 0x59: case 0x5a: case 0x5b: case 0x5c: case 0x5d: case 0x5e: case 0x5f: {
        int n = (b & 7) | (d->rexb << 3), sz = stack_sz(c, d);
        u64 v = pop(c, sz);
        set_reg(c, d, n, sz, v);
        return;
    }
    case 0x60: {   /* PUSHA */
        if (c->mode64) unimpl(c, d, 0, b);
        u64 osp = c->r[4];
        for (int i = 0; i < 8; i++) push(c, i == 4 ? osp & MASK[osz] : get_reg(c, d, i, osz), osz);
        return;
    }
    case 0x61: {   /* POPA */
        if (c->mode64) unimpl(c, d, 0, b);
        for (int i = 7; i >= 0; i--) { u64 v = pop(c, osz); if (i != 4) set_reg(c, d, i, osz, v); }
        return;
    }
    case 0x9a: {   /* CALL ptr16:16/32 */
        if (c->mode64) unimpl(c, d, 0, b);
        u32 off = osz == 2 ? fetch16(c, d) : fetch32(c, d);
        u16 sel = fetch16(c, d);
        u16 ocs = c->seg[CS].sel;
        u64 oip = d->ip;
        load_seg(c, CS, sel);
        push(c, ocs, osz);
        push(c, oip, osz);
        d->ip = off;
        return;
    }
    case 0xc4: case 0xc5: {   /* LES / LDS */
        if (c->mode64) unimpl(c, d, 0, b);
        decode_modrm(c, d);
        if (d->mod == 3) UD(c);
        u64 off = vread(c, d->ea_seg, ea_off(d), osz);
        u16 sel = (u16)vread(c, d->ea_seg, ea_off(d) + (u64)osz, 2);
        load_seg(c, b == 0xc4 ? ES : DS, sel);
        set_reg(c, d, d->reg, osz, off);
        return;
    }
    case 0x63:
        if (!c->mode64) unimpl(c, d, 0, b);
        decode_modrm(c, d);
        if (osz == 8) set_reg(c, d, d->reg, 8, sext(rd_rm(c, d, 4), 4));
        else set_reg(c, d, d->reg, osz, rd_rm(c, d, osz));
        return;
    case 0x68: { u64 v = fetch_iz(c, d); int sz = stack_sz(c, d); push(c, sz == 8 ? v : v & MASK[sz], sz); return; }
    case 0x6a: { u64 v = sext(fetch8(c, d), 1); int sz = stack_sz(c, d); push(c, v & MASK[sz], sz); return; }
    case 0x69: case 0x6b: {
        decode_modrm(c, d);
        u64 imm = b == 0x6b ? sext(fetch8(c, d), 1) : fetch_iz(c, d);
        u64 r = imul_trunc(c, rd_rm(c, d, osz), imm, osz);
        set_reg(c, d, d->reg, osz, r);
        return;
    }
    case 0x6c: case 0x6d: case 0x6e: case 0x6f:
    case 0xa4: case 0xa5: case 0xa6: case 0xa7: case 0xaa: case 0xab: case 0xac: case 0xad: case 0xae: case 0xaf:
        string_op(c, d, b);
        return;
    case 0x70: case 0x71: case 0x72: case 0x73: case 0x74: case 0x75: case 0x76: case 0x77:
    case 0x78: case 0x79: case 0x7a: case 0x7b: case 0x7c: case 0x7d: case 0x7e: case 0x7f: {
        u64 rel = sext(fetch8(c, d), 1);
        if (cond(c, b & 15)) jcc_or_jmp(c, d, rel);
        return;
    }
    case 0x80: case 0x81: case 0x82: case 0x83: {
        int sz = (b & 1) ? osz : 1;
        decode_modrm(c, d);
        u64 imm = b == 0x81 ? fetch_iz(c, d) : sext(fetch8(c, d), 1);
        int op = d->reg & 7;
        u64 r = alu(c, op, rd_rm(c, d, sz), imm, sz);
        if (op != 7) wr_rm(c, d, sz, r);
        return;
    }
    case 0x84: case 0x85: {
        int sz = (b & 1) ? osz : 1;
        decode_modrm(c, d);
        alu(c, 4, rd_rm(c, d, sz), get_reg(c, d, d->reg, sz), sz);
        return;
    }
    case 0x86: case 0x87: {
        int sz = (b & 1) ? osz : 1;
        decode_modrm(c, d);
        u64 a = rd_rm(c, d, sz), r = get_reg(c, d, d->reg, sz);
        wr_rm(c, d, sz, r);
        set_reg(c, d, d->reg, sz, a);
        return;
    }
    case 0x88: case 0x89: { int sz = (b & 1) ? osz : 1; decode_modrm(c, d); wr_rm(c, d, sz, get_reg(c, d, d->reg, sz)); return; }
    case 0x8a: case 0x8b: { int sz = (b & 1) ? osz : 1; decode_modrm(c, d); set_reg(c, d, d->reg, sz, rd_rm(c, d, sz)); return; }
    case 0x8c: {
        decode_modrm(c, d);
        int s = d->reg & 7;
        if (s > GS) UD(c);
        if (d->mod == 3) set_reg(c, d, d->rm, osz == 2 ? 2 : osz, c->seg[s].sel);
        else wr_rm(c, d, 2, c->seg[s].sel);
        return;
    }
    case 0x8d:
        decode_modrm(c, d);
        if (d->mod == 3) UD(c);
        set_reg(c, d, d->reg, osz, ea_off(d));
        return;
    case 0x8e: {
        decode_modrm(c, d);
        int s = d->reg & 7;
        if (s == CS || s > GS) UD(c);
        load_seg(c, s, (u16)rd_rm(c, d, 2));
        if (s == SS) c->shadow = 1;
        return;
    }
    case 0x8f: {
        decode_modrm(c, d);
        int sz = stack_sz(c, d);
        u64 v = pop(c, sz);
        wr_rm(c, d, sz, v);
        return;
    }
    case 0x90:
        if (d->rexb) { u64 t = c->r[8]; set_reg(c, d, 8, osz, c->r[0]); set_reg(c, d, 0, osz, t); }
        else if (d->rep == 0xf3 && c->m->ncpu > 1) c->yield_req = 1;   /* PAUSE: let the other CPUs run (spin locks) */
        return;     /* NOP / PAUSE */
    case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97: {
        int n = (b & 7) | (d->rexb << 3);
        u64 t = get_reg(c, d, n, osz);
        set_reg(c, d, n, osz, get_reg(c, d, 0, osz));
        set_reg(c, d, 0, osz, t);
        return;
    }
    case 0x98:
        if (osz == 2) set_reg(c, d, 0, 2, sext(c->r[0], 1));
        else if (osz == 4) set_reg(c, d, 0, 4, sext(c->r[0], 2));
        else c->r[0] = sext(c->r[0], 4);
        return;
    case 0x99:
        if (osz == 2) set_reg(c, d, 2, 2, (c->r[0] & 0x8000) ? 0xffff : 0);
        else if (osz == 4) set_reg(c, d, 2, 4, (c->r[0] & 0x80000000u) ? 0xffffffffu : 0);
        else c->r[2] = (c->r[0] >> 63) ? ~0ull : 0;
        return;
    case 0x9b: return;  /* FWAIT */
    case 0x9c: {
        int sz = stack_sz(c, d);
        push(c, get_flags(c) & 0x3c7fd7ull & MASK[sz], sz);
        return;
    }
    case 0x9d: { int sz = stack_sz(c, d); set_rflags(c, pop(c, sz), sz); return; }
    case 0x9e: set_flags_raw(c, (get_flags(c) & ~0xffull & F_ARITH) | ((c->r[0] >> 8) & 0xd5)); return;
    case 0x9f: c->r[0] = (c->r[0] & ~0xff00ull) | (((get_flags(c) & 0xd5) | 2) << 8); return;
    case 0xa0: case 0xa1: case 0xa2: case 0xa3: {
        int sz = (b & 1) ? osz : 1;
        u64 off = d->asz == 8 ? fetch64(c, d) : d->asz == 4 ? fetch32(c, d) : fetch16(c, d);
        int s = d->seg >= 0 ? d->seg : DS;
        if (b < 0xa2) set_reg(c, d, 0, sz, vread(c, s, off, sz));
        else vwrite(c, s, off, c->r[0], sz);
        return;
    }
    case 0xa8: alu(c, 4, c->r[0], fetch8(c, d), 1); return;
    case 0xa9: alu(c, 4, c->r[0], fetch_iz(c, d), osz); return;
    case 0xb0: case 0xb1: case 0xb2: case 0xb3: case 0xb4: case 0xb5: case 0xb6: case 0xb7:
        set_reg(c, d, (b & 7) | (d->rexb << 3), 1, fetch8(c, d));
        return;
    case 0xb8: case 0xb9: case 0xba: case 0xbb: case 0xbc: case 0xbd: case 0xbe: case 0xbf: {
        u64 v = osz == 8 ? fetch64(c, d) : osz == 4 ? fetch32(c, d) : fetch16(c, d);
        set_reg(c, d, (b & 7) | (d->rexb << 3), osz, v);
        return;
    }
    case 0xc0: case 0xc1: case 0xd0: case 0xd1: case 0xd2: case 0xd3: {
        int sz = (b & 1) ? osz : 1;
        decode_modrm(c, d);
        u32 cnt = b <= 0xc1 ? fetch8(c, d) : b <= 0xd1 ? 1 : (u32)(c->r[1] & 0xff);
        u64 r = shift(c, d->reg, rd_rm(c, d, sz), cnt, sz);
        wr_rm(c, d, sz, r);
        return;
    }
    case 0xc2: case 0xc3: {
        u32 extra = b == 0xc2 ? fetch16(c, d) : 0;
        int sz = c->mode64 ? (osz == 2 ? 2 : 8) : osz;
        u64 t = pop(c, sz);
        set_sp(c, c->r[4] + extra);
        d->ip = t;
        return;
    }
    case 0xc6: case 0xc7: {
        int sz = (b & 1) ? osz : 1;
        decode_modrm(c, d);
        if ((d->reg & 7) != 0) unimpl(c, d, 0, b);
        u64 imm = sz == 1 ? fetch8(c, d) : fetch_iz(c, d);
        wr_rm(c, d, sz, imm);
        return;
    }
    case 0xc8: {
        u16 size = fetch16(c, d);
        u8 level = fetch8(c, d) & 31;
        int sz = stack_sz(c, d);
        push(c, c->r[5], sz);
        u64 frame = c->r[4];
        for (int i = 1; i < level; i++) {
            c->r[5] -= (u64)sz;
            push(c, vread(c, SS, c->r[5], sz), sz);
        }
        if (level) push(c, frame, sz);
        set_reg(c, d, 5, sz, frame);
        set_sp(c, c->r[4] - size);
        return;
    }
    case 0xc9: {
        int sz = stack_sz(c, d);
        set_sp(c, c->r[5]);
        set_reg(c, d, 5, sz, pop(c, sz));
        return;
    }
    case 0xca: case 0xcb: {
        u32 extra = b == 0xca ? fetch16(c, d) : 0;
        int sz = osz;
        u64 ip = pop(c, sz);
        u16 sel = (u16)pop(c, sz);
        load_seg(c, CS, sel);
        set_sp(c, c->r[4] + extra);
        d->ip = ip;
        return;
    }
    case 0xcc: d->ip = d->ip; do_interrupt(c, 3, d->ip, 0, 0, 1); d->ip = c->rip; return;
    case 0xf1: do_interrupt(c, 1, d->ip, 0, 0, 0); d->ip = c->rip; return;   /* ICEBP: #DB trap, no DPL check */
    case 0xcd: { u8 v = fetch8(c, d); do_interrupt(c, v, d->ip, 0, 0, 1); d->ip = c->rip; return; }
    case 0xcf: {
        int sz = osz;
        int iret_cpl = c->cpl;
        u64 ip = pop(c, sz);
        u16 cs = (u16)pop(c, sz);
        u64 fl = pop(c, sz);
        if (c->mode64) {
            u64 sp = pop(c, sz);
            u16 ss = (u16)pop(c, sz);
            load_seg(c, CS, cs);
            if ((ss & ~3u) == 0) { c->seg[SS].sel = ss; c->seg[SS].base = 0; c->seg[SS].attr = 0; }
            else load_seg(c, SS, ss);
            c->r[4] = sp;
        } else {
            int old = c->cpl;
            load_seg(c, CS, cs);
            if (c->cpl > old) {
                u64 sp = pop(c, sz);
                u16 ss = (u16)pop(c, sz);
                load_seg(c, SS, ss);
                c->r[4] = sp;
            }
        }
        set_rflags_at(c, fl, sz == 2 ? 2 : 8, iret_cpl);
        d->ip = ip;
        return;
    }
    case 0xd7: {
        u64 a = (c->r[3] + (c->r[0] & 0xff));
        if (d->asz == 4) a = (u32)a;
        if (d->asz == 2) a &= 0xffff;
        set_reg(c, d, 0, 1, vread(c, d->seg >= 0 ? d->seg : DS, a, 1));
        return;
    }
    case 0xd8: case 0xd9: case 0xda: case 0xdb: case 0xdc: case 0xdd: case 0xde: case 0xdf:
        if (c->cr0 & (CR0_TS | 4)) cpu_exc(c, EXC_NM, 0, 0);
        x87_exec(c, d, b);
        return;
    case 0xe0: case 0xe1: case 0xe2: case 0xe3: {
        u64 rel = sext(fetch8(c, d), 1);
        u64 am = d->asz == 8 ? ~0ull : d->asz == 4 ? 0xffffffffull : 0xffffull;
        int take;
        if (b == 0xe3) take = (c->r[1] & am) == 0;
        else {
            u64 cnt = ((c->r[1] & am) - 1) & am;
            set_areg(c, 1, cnt, d->asz);
            take = cnt != 0;
            if (b == 0xe0) take = take && !lf_zf(c);
            if (b == 0xe1) take = take && lf_zf(c);
        }
        if (take) d->ip += rel;
        return;
    }
    case 0xe4: case 0xe5: {
        u8 port = fetch8(c, d);
        int sz = (b & 1) ? (osz == 8 ? 4 : osz) : 1;
        set_reg(c, d, 0, sz, io_in(c->m, port, sz));
        return;
    }
    case 0xe6: case 0xe7: {
        u8 port = fetch8(c, d);
        int sz = (b & 1) ? (osz == 8 ? 4 : osz) : 1;
        io_out(c->m, port, (u32)(c->r[0] & MASK[sz]), sz);
        return;
    }
    case 0xec: case 0xed: {
        int sz = (b & 1) ? (osz == 8 ? 4 : osz) : 1;
        set_reg(c, d, 0, sz, io_in(c->m, (u16)c->r[2], sz));
        return;
    }
    case 0xee: case 0xef: {
        int sz = (b & 1) ? (osz == 8 ? 4 : osz) : 1;
        io_out(c->m, (u16)c->r[2], (u32)(c->r[0] & MASK[sz]), sz);
        return;
    }
    case 0xe8: {
        u64 rel = c->mode64 ? sext(fetch32(c, d), 4) : (osz == 2 ? sext(fetch16(c, d), 2) : sext(fetch32(c, d), 4));
        push(c, d->ip, c->mode64 ? 8 : osz);
        d->ip += rel;
        return;
    }
    case 0xe9: {
        u64 rel = (!c->mode64 && osz == 2) ? sext(fetch16(c, d), 2) : sext(fetch32(c, d), 4);
        d->ip += rel;
        return;
    }
    case 0xea: {
        if (c->mode64) unimpl(c, d, 0, b);
        u32 off = osz == 2 ? fetch16(c, d) : fetch32(c, d);
        u16 sel = fetch16(c, d);
        load_seg(c, CS, sel);
        d->ip = off;
        return;
    }
    case 0xeb: { u64 rel = sext(fetch8(c, d), 1); d->ip += rel; return; }
    case 0xf4:
        if (c->cpl) GP0(c);
        c->halted = 1;
        return;
    case 0xf5: put_flags_cf_of(c, !(get_flags(c) & F_CF), lf_of(c)); return;
    case 0xf6: decode_modrm(c, d); group3(c, d, 1); return;
    case 0xf7: decode_modrm(c, d); group3(c, d, osz); return;
    case 0xf8: put_flags_cf_of(c, 0, lf_of(c)); return;
    case 0xf9: put_flags_cf_of(c, 1, lf_of(c)); return;
    case 0xfa: if (c->cpl) GP0(c); c->rflags &= ~(u64)F_IF; return;
    case 0xfb:
        if (c->cpl) GP0(c);
        if (!(c->rflags & F_IF)) c->shadow = 1;
        c->rflags |= F_IF;
        return;
    case 0xfc: c->rflags &= ~(u64)F_DF; return;
    case 0xfd: c->rflags |= F_DF; return;
    case 0xfe: {
        decode_modrm(c, d);
        int r = d->reg & 7;
        if (r > 1) unimpl(c, d, 0, b);
        wr_rm(c, d, 1, inc_dec(c, rd_rm(c, d, 1), r, 1));
        return;
    }
    case 0xff: {
        decode_modrm(c, d);
        switch (d->reg & 7) {
        case 0: case 1: wr_rm(c, d, osz, inc_dec(c, rd_rm(c, d, osz), d->reg & 1, osz)); return;
        case 2: {
            int sz = c->mode64 ? 8 : osz;
            u64 t = rd_rm(c, d, sz);
            push(c, d->ip, sz);
            d->ip = t;
            return;
        }
        case 4: d->ip = rd_rm(c, d, c->mode64 ? 8 : osz); return;
        case 3: case 5: {
            if (d->mod == 3) UD(c);
            u64 off = ea_off(d);
            u64 ip = vread(c, d->ea_seg, off, osz);
            u16 sel = (u16)vread(c, d->ea_seg, off + (u64)osz, 2);
            if ((d->reg & 7) == 3) {
                push(c, c->seg[CS].sel, osz);
                push(c, d->ip, osz);
            }
            load_seg(c, CS, sel);
            d->ip = ip;
            return;
        }
        case 6: { int sz = stack_sz(c, d); push(c, rd_rm(c, d, sz), sz); return; }
        default: unimpl(c, d, 0, b);
        }
    }
    default:
        unimpl(c, d, 0, b);
    }
}

static u64 bit_base(Cpu *c, Dec *d, i64 bitoff, int sz, u64 *addr) {
    /* register bit offsets address a bit string in memory */
    i64 bytes = (bitoff >> (sz == 8 ? 6 : sz == 4 ? 5 : 4)) * sz;
    *addr = ea_off(d) + (u64)bytes;
    if (d->asz == 4) *addr = (u32)*addr;
    return vread(c, d->ea_seg, *addr, sz);
}

static void exec_0f(Cpu *c, Dec *d) {
    u8 b = fetch8(c, d);
    d->opc = 0x100 | b;
    int osz = d->osz;
    switch (b) {
    case 0x00: {
        decode_modrm(c, d);
        switch (d->reg & 7) {
        case 0: wr_rm(c, d, d->mod == 3 ? osz : 2, c->ldtr.sel); return;
        case 1: wr_rm(c, d, d->mod == 3 ? osz : 2, c->tr.sel); return;
        case 2: if (c->cpl) GP0(c); load_system_seg(c, &c->ldtr, (u16)rd_rm(c, d, 2)); return;
        case 3: if (c->cpl) GP0(c); load_system_seg(c, &c->tr, (u16)rd_rm(c, d, 2)); return;
        default: unimpl(c, d, 1, b);
        }
    }
    case 0x01: {
        decode_modrm(c, d);
        int r = d->reg & 7;
        if (d->mod == 3) {
            int low = d->rm & 7;
            if (r == 7 && low == 0 && c->mode64) {   /* SWAPGS */
                if (c->cpl) GP0(c);
                u64 t = c->seg[GS].base; c->seg[GS].base = c->gs_kernel; c->gs_kernel = t;
                return;
            }
            if (r == 7 && low == 1) {                 /* RDTSCP */
                u64 t = tsc(c);
                c->r[0] = (u32)t; c->r[2] = t >> 32; c->r[1] = (u32)c->tsc_aux;
                return;
            }
            if (r == 4) { set_reg(c, d, d->rm, osz, c->cr0 & 0xffff); return; }
            unimpl(c, d, 1, b);
        }
        u64 off = ea_off(d);
        int s = d->ea_seg;
        switch (r) {
        case 0: case 1: {
            u16 lim = r ? c->idtr.limit : c->gdtr.limit;
            u64 base = r ? c->idtr.base : c->gdtr.base;
            vwrite(c, s, off, lim, 2);
            vwrite(c, s, off + 2, base, c->mode64 ? 8 : 4);
            return;
        }
        case 2: case 3: {
            if (c->cpl) GP0(c);
            u16 lim = (u16)vread(c, s, off, 2);
            u64 base = c->mode64 ? vread(c, s, off + 2, 8) : vread(c, s, off + 2, 4);
            if (!c->mode64 && osz == 2) base &= 0xffffff;
            if (r == 2) { c->gdtr.limit = lim; c->gdtr.base = base; }
            else { c->idtr.limit = lim; c->idtr.base = base; }
            return;
        }
        case 4: wr_rm(c, d, 2, c->cr0 & 0xffff); return;
        case 6: if (c->cpl) GP0(c); write_cr0(c, (c->cr0 & ~0xeull) | (rd_rm(c, d, 2) & 0xf) | (c->cr0 & 1)); return;
        case 7: if (c->cpl) GP0(c); tlb_flush_page(c, seg_lin(c, s, off)); return;
        default: unimpl(c, d, 1, b);
        }
    }
    case 0x06:
        if (c->cpl) GP0(c);
        if (c->m->jit && (c->cr0 & CR0_TS)) jit_flush_code(c->m);
        c->cr0 &= ~CR0_TS;
        return;
    case 0x08: case 0x09: if (c->cpl) GP0(c); return;
    case 0x0b: UD(c);
    case 0x0d: case 0x18: case 0x19: case 0x1a: case 0x1b: case 0x1c: case 0x1d: case 0x1e: case 0x1f:
        decode_modrm(c, d);
        return;
    case 0x20: case 0x22: {
        decode_modrm(c, d);
        if (c->cpl) GP0(c);
        int sz = c->mode64 ? 8 : 4;
        int crn = d->reg;
        if (b == 0x20) {
            u64 v;
            switch (crn) {
            case 0: v = c->cr0; break;
            case 2: v = c->cr2; break;
            case 3: v = c->cr3; break;
            case 4: v = c->cr4; break;
            case 8: v = c->cr8; break;
            default: UD(c);
            }
            set_reg(c, d, d->rm, sz, v);
        } else {
            u64 v = get_reg(c, d, d->rm, sz);
            switch (crn) {
            case 0: write_cr0(c, v); break;
            case 2: c->cr2 = v; break;
            case 3: stat_cr3++; c->cr3 = v; tlb_new_epoch(c, 1); break;
            case 4:
                if ((c->efer & EFER_LMA) && !(v & CR4_PAE)) GP0(c);
                if (c->m->jit && ((c->cr4 ^ v) & 0x200)) jit_flush_code(c->m);
                c->cr4 = v; tlb_flush(c); break;
            case 8: c->cr8 = v & 15; if (c->m->chipset) apic_set_tpr_from_cr8(c->m, v); break;
            default: UD(c);
            }
        }
        return;
    }
    case 0x21: decode_modrm(c, d); if (c->cpl) GP0(c);
        set_reg(c, d, d->rm, c->mode64 ? 8 : 4, d->reg == 6 ? 0xffff0ff0u : d->reg == 7 ? 0x400 : 0);
        return;
    case 0x23: decode_modrm(c, d); if (c->cpl) GP0(c); return;
    case 0x30: {
        if (c->cpl) GP0(c);
        u64 v = ((c->r[2] & 0xffffffffull) << 32) | (u32)c->r[0];
        if (!wrmsr(c, (u32)c->r[1], v)) {
            emu_log(c->m, "wrmsr %x <- %llx: unknown MSR, #GP", (u32)c->r[1], (unsigned long long)v);
            GP0(c);
        }
        return;
    }
    case 0x31: { u64 t = tsc(c); c->r[0] = (u32)t; c->r[2] = t >> 32; return; }
    case 0x32: {
        if (c->cpl) GP0(c);
        u64 v;
        if (!rdmsr(c, (u32)c->r[1], &v)) {
            emu_log(c->m, "rdmsr %x: unknown MSR, #GP", (u32)c->r[1]);
            GP0(c);
        }
        c->r[0] = (u32)v; c->r[2] = v >> 32;
        return;
    }
    case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46: case 0x47:
    case 0x48: case 0x49: case 0x4a: case 0x4b: case 0x4c: case 0x4d: case 0x4e: case 0x4f: {
        decode_modrm(c, d);
        u64 v = rd_rm(c, d, osz);
        if (cond(c, b & 15)) set_reg(c, d, d->reg, osz, v);
        else if (osz == 4) set_reg(c, d, d->reg, 4, get_reg(c, d, d->reg, 4));
        return;
    }
    case 0x80: case 0x81: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
    case 0x88: case 0x89: case 0x8a: case 0x8b: case 0x8c: case 0x8d: case 0x8e: case 0x8f: {
        u64 rel = (!c->mode64 && osz == 2) ? sext(fetch16(c, d), 2) : sext(fetch32(c, d), 4);
        if (cond(c, b & 15)) d->ip += rel;
        return;
    }
    case 0x90: case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97:
    case 0x98: case 0x99: case 0x9a: case 0x9b: case 0x9c: case 0x9d: case 0x9e: case 0x9f:
        decode_modrm(c, d);
        wr_rm(c, d, 1, (u64)cond(c, b & 15));
        return;
    case 0xa0: case 0xa8: { int sz = stack_sz(c, d); push(c, c->seg[b == 0xa0 ? FS : GS].sel, sz); return; }
    case 0xa1: case 0xa9: {
        int sz = stack_sz(c, d);
        int s = b == 0xa1 ? FS : GS;
        u16 sel = (u16)pop(c, sz);
        u64 base = c->seg[s].base;
        load_seg(c, s, sel);
        if (c->mode64 && (sel & ~3u) == 0) c->seg[s].base = base;
        return;
    }
    case 0xa2: cpuid(c); return;
    case 0x05: {   /* SYSCALL */
        if (!(c->efer & EFER_SCE) || !(c->efer & EFER_LMA)) UD(c);
        u16 ksel = (u16)(c->star >> 32) & 0xfffc;
        c->r[1] = d->ip;
        c->r[11] = get_flags(c) & ~0x10000ull;
        c->rflags = (c->r[11] & ~c->sfmask) | 2;
        c->seg[CS].sel = ksel; c->seg[CS].base = 0; c->seg[CS].limit = 0xffffffffu; c->seg[CS].attr = 0xa09b;
        c->seg[SS].sel = (u16)(ksel + 8); c->seg[SS].base = 0; c->seg[SS].limit = 0xffffffffu; c->seg[SS].attr = 0xc093;
        c->cpl = 0;
        update_mode(c);
        d->ip = c->mode64 ? c->lstar : c->cstar;
        return;
    }
    case 0x07: {   /* SYSRET */
        if (!(c->efer & EFER_SCE) || !(c->efer & EFER_LMA) || c->cpl) UD(c);
        u16 usel = (u16)(c->star >> 48);
        if (d->rexw) {
            c->seg[CS].sel = (u16)((usel + 16) | 3); c->seg[CS].attr = 0xa0fb;
            d->ip = c->r[1];
        } else {
            c->seg[CS].sel = (u16)(usel | 3); c->seg[CS].attr = 0xc0fb;
            d->ip = (u32)c->r[1];
        }
        c->seg[CS].base = 0; c->seg[CS].limit = 0xffffffffu;
        c->seg[SS].sel = (u16)((usel + 8) | 3); c->seg[SS].base = 0; c->seg[SS].limit = 0xffffffffu; c->seg[SS].attr = 0xc0f3;
        c->rflags = (c->r[11] & 0x3c7fd7ull) | 2;
        c->lf_op = LF_NONE;
        c->cpl = 3;
        update_mode(c);
        return;
    }
    case 0x38: case 0x3a: sse4_exec(c, d, b); return;
    case 0xa3: case 0xab: case 0xb3: case 0xbb: case 0xba: {
        decode_modrm(c, d);
        int op;
        i64 bit;
        if (b == 0xba) {
            op = d->reg & 7;
            if (op < 4) unimpl(c, d, 1, b);
            bit = fetch8(c, d) & (osz * 8 - 1);
            op -= 4;
        } else {
            op = b == 0xa3 ? 0 : b == 0xab ? 1 : b == 0xb3 ? 2 : 3;
            bit = (i64)sext(get_reg(c, d, d->reg, osz), osz);
        }
        u64 v, addr = 0;
        int mem = d->mod != 3;
        if (mem && b != 0xba) v = bit_base(c, d, bit, osz, &addr);
        else { v = rd_rm(c, d, osz); addr = ea_off(d); }
        u32 n = (u32)bit & (u32)(osz * 8 - 1);
        u32 cf = (v >> n) & 1;
        if (op == 1) v |= 1ull << n;
        else if (op == 2) v &= ~(1ull << n);
        else if (op == 3) v ^= 1ull << n;
        put_flags_cf_of(c, cf, lf_of(c));
        if (op) {
            if (mem) vwrite(c, d->ea_seg, addr, v, osz);
            else set_reg(c, d, d->rm, osz, v);
        }
        return;
    }
    case 0xa4: case 0xa5: case 0xac: case 0xad: {
        decode_modrm(c, d);
        u32 cnt = (b & 1) ? (u32)(c->r[1] & 0xff) : fetch8(c, d);
        u64 r = shld_shrd(c, b >= 0xac, rd_rm(c, d, osz), get_reg(c, d, d->reg, osz), cnt, osz);
        wr_rm(c, d, osz, r);
        return;
    }
    case 0xaf: {
        decode_modrm(c, d);
        u64 r = imul_trunc(c, get_reg(c, d, d->reg, osz), rd_rm(c, d, osz), osz);
        set_reg(c, d, d->reg, osz, r);
        return;
    }
    case 0xb0: case 0xb1: {
        int sz = b == 0xb0 ? 1 : osz;
        decode_modrm(c, d);
        u64 dst = rd_rm(c, d, sz), acc = c->r[0] & MASK[sz];
        alu(c, 7, acc, dst, sz);
        if (acc == dst) wr_rm(c, d, sz, get_reg(c, d, d->reg, sz));
        else {
            if (d->mod != 3) wr_rm(c, d, sz, dst);
            set_reg(c, d, 0, sz, dst);
        }
        return;
    }
    case 0xb6: case 0xb7: {
        decode_modrm(c, d);
        set_reg(c, d, d->reg, osz, rd_rm(c, d, b == 0xb6 ? 1 : 2));
        return;
    }
    case 0xb2: case 0xb4: case 0xb5: {   /* LSS / LFS / LGS */
        decode_modrm(c, d);
        if (d->mod == 3) UD(c);
        int sz = osz == 8 ? 8 : osz;
        u64 off = vread(c, d->ea_seg, ea_off(d), sz);
        u16 sel = (u16)vread(c, d->ea_seg, ea_off(d) + (u64)sz, 2);
        load_seg(c, b == 0xb2 ? SS : b == 0xb4 ? FS : GS, sel);
        set_reg(c, d, d->reg, osz, off);
        if (b == 0xb2) c->shadow = 1;
        return;
    }
    case 0xbe: case 0xbf: {
        decode_modrm(c, d);
        int ssz = b == 0xbe ? 1 : 2;
        set_reg(c, d, d->reg, osz, sext(rd_rm(c, d, ssz), ssz) & MASK[osz]);
        return;
    }
    case 0xb8: {
        if (d->rep != 0xf3) unimpl(c, d, 1, b);
        decode_modrm(c, d);
        u64 v = rd_rm(c, d, osz);
        u32 n = 0;
        while (v) { n += (u32)(v & 1); v >>= 1; }
        set_reg(c, d, d->reg, osz, n);
        set_flags_raw(c, n == 0 ? F_ZF : 0);
        return;
    }
    case 0xbc: case 0xbd: {
        /* TZCNT/LZCNT decode as BSF/BSR on CPUs without BMI1/ABM, which we advertise */
        decode_modrm(c, d);
        u64 v = rd_rm(c, d, osz);
        if (!v) { set_flags_raw(c, F_ZF); return; }
        u32 n;
        if (b == 0xbc) { n = 0; while (!((v >> n) & 1)) n++; }
        else { n = (u32)osz * 8 - 1; while (!((v >> n) & 1)) n--; }
        set_reg(c, d, d->reg, osz, n);
        set_flags_raw(c, 0);
        return;
    }
    case 0xc0: case 0xc1: {
        int sz = b == 0xc0 ? 1 : osz;
        decode_modrm(c, d);
        u64 dst = rd_rm(c, d, sz), src = get_reg(c, d, d->reg, sz);
        u64 r = alu(c, 0, dst, src, sz);
        if (d->mod == 3) {           /* SRC <- DEST, then DEST <- sum (same register: sum wins) */
            set_reg(c, d, d->reg, sz, dst);
            set_reg(c, d, d->rm, sz, r);
        } else {
            wr_rm(c, d, sz, r);
            set_reg(c, d, d->reg, sz, dst);
        }
        return;
    }
    case 0xc3:
        decode_modrm(c, d);
        if (d->mod == 3) UD(c);
        wr_rm(c, d, osz, get_reg(c, d, d->reg, osz));
        return;
    case 0xc7: {
        decode_modrm(c, d);
        if ((d->reg & 7) == 1 && d->mod != 3) {
            u64 off = ea_off(d);
            if (d->rexw) {
                if (seg_lin(c, d->ea_seg, off) & 15) GP0(c);
                u64 lo = vread(c, d->ea_seg, off, 8), hi = vread(c, d->ea_seg, off + 8, 8);
                if (lo == c->r[0] && hi == c->r[2]) {
                    vwrite(c, d->ea_seg, off, c->r[3], 8);
                    vwrite(c, d->ea_seg, off + 8, c->r[1], 8);
                    set_flags_raw(c, (get_flags(c) & F_ARITH) | F_ZF);
                } else {
                    c->r[0] = lo; c->r[2] = hi;
                    set_flags_raw(c, get_flags(c) & F_ARITH & ~(u64)F_ZF);
                }
            } else {
                u64 v = vread(c, d->ea_seg, off, 8);
                u64 cmp = ((c->r[2] & 0xffffffffull) << 32) | (u32)c->r[0];
                if (v == cmp) {
                    vwrite(c, d->ea_seg, off, ((c->r[1] & 0xffffffffull) << 32) | (u32)c->r[3], 8);
                    set_flags_raw(c, (get_flags(c) & F_ARITH) | F_ZF);
                } else {
                    c->r[0] = (u32)v; c->r[2] = v >> 32;
                    set_flags_raw(c, get_flags(c) & F_ARITH & ~(u64)F_ZF);
                }
            }
            return;
        }
        unimpl(c, d, 1, b);
    }
    case 0xc8: case 0xc9: case 0xca: case 0xcb: case 0xcc: case 0xcd: case 0xce: case 0xcf: {
        int n = (b & 7) | (d->rexb << 3);
        if (osz == 8) c->r[n] = __builtin_bswap64(c->r[n]);
        else if (osz == 4) c->r[n] = __builtin_bswap32((u32)c->r[n]);
        else set_reg(c, d, n, 2, 0);
        return;
    }
    default:
        if (sse_exec(c, d, b)) return;
        unimpl(c, d, 1, b);
    }
}

/* ================= step ================= */

static void break_hit(Cpu *c) {
    static int hits;
    if (++hits > c->m->break_max) return;
    if (c->m->break_max > 3) {   /* compact mode */
        emu_log(c->m, "BREAK #%d icount %llu rip %llx rcx %llx rdx %llx r8 %llx r14 %llx r15 %llx", hits,
                (unsigned long long)c->icount, (unsigned long long)c->rip, (unsigned long long)c->r[1],
                (unsigned long long)c->r[2], (unsigned long long)c->r[8], (unsigned long long)c->r[14],
                (unsigned long long)c->r[15]);
        return;
    }
    u64 ra = 0, pa;
    {
        static const char *rn[16] = { "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15" };
        for (int i = 0; i < 16; i += 4)
            emu_log(c->m, "  %s=%llx %s=%llx %s=%llx %s=%llx", rn[i], (unsigned long long)c->r[i], rn[i + 1],
                    (unsigned long long)c->r[i + 1], rn[i + 2], (unsigned long long)c->r[i + 2], rn[i + 3],
                    (unsigned long long)c->r[i + 3]);
        {   /* code at the function pointer [r13+0x70] and the qword at [rcx+8] */
            u64 fp = 0, q = 0; u8 *pp = lin_host(c, c->r[13] + 0x70, 0, &pa); if (pp) e_memcpy(&fp, pp, 8);
            char line[200]; u32 n = 0; const char *hx = "0123456789abcdef";
            for (int k = 0; k < 40 && fp; k++) { u8 *cp = lin_host(c, fp + (u64)k, 0, &pa); u8 bb = cp ? *cp : 0; line[n++] = hx[bb >> 4]; line[n++] = hx[bb & 15]; line[n++] = ' '; }
            line[n] = 0;
            emu_log(c->m, "  code at %llx: %s", (unsigned long long)fp, line);
            pp = lin_host(c, c->r[1] + 8, 0, &pa); if (pp) e_memcpy(&q, pp, 8);
            emu_log(c->m, "  [rcx+8] = %llx", (unsigned long long)q);
        }
        for (int k = 0; k < 0x100; k += 32) {
            u64 q[4] = { 0, 0, 0, 0 };
            for (int j = 0; j < 4; j++) { u8 *pp = lin_host(c, c->r[13] + (u64)(k + 8 * j), 0, &pa); if (pp) e_memcpy(&q[j], pp, 8); }
            emu_log(c->m, "  [r13+%02x] %016llx %016llx %016llx %016llx", k, (unsigned long long)q[0],
                    (unsigned long long)q[1], (unsigned long long)q[2], (unsigned long long)q[3]);
        }
    }
    u8 *p = lin_host(c, c->r[4], 0, &pa);
    if (p) e_memcpy(&ra, p, 8);
    emu_log(c->m, "BREAK at %llx: rcx=%llx rdx=%llx r8=%llx r9=%llx return=%llx", (unsigned long long)c->rip,
            (unsigned long long)c->r[1], (unsigned long long)c->r[2], (unsigned long long)c->r[8],
            (unsigned long long)c->r[9], (unsigned long long)ra);
    u64 a5 = 0;
    p = lin_host(c, c->r[4] + 0x28, 0, &pa);
    if (p) e_memcpy(&a5, p, 8);
    emu_log(c->m, "      5th argument [rsp+28]=%llx", (unsigned long long)a5);
    for (int i = 0; i < 48; i++) {
        u64 v = 0;
        p = lin_host(c, c->r[4] + 8 * (u64)i, 0, &pa);
        if (p) e_memcpy(&v, p, 8);
        if ((v >> 40) == 0xfffff8ull) emu_log(c->m, "      [rsp+%x] %llx", 8 * i, (unsigned long long)v);
    }
}

static inline void step_insn(Cpu *c) {
    Dec d;
    if (__builtin_expect(c->rip == c->m->break_rip, 0) && c->m->break_rip) break_hit(c);
    d.ip = d.start = c->rip;
    d.seg = -1;
    d.rep = 0;
    d.p66 = 0;
    d.rex = d.rexw = d.rexr = d.rexx = d.rexb = 0;
    d.mod = 3; d.rip_rel = 0;
    int a67 = 0;
    u8 b;
    for (;;) {
        b = fetch8(c, &d);
        switch (b) {
        case 0x66: d.p66 = 1; continue;
        case 0x67: a67 = 1; continue;
        case 0xf2: case 0xf3: d.rep = b; continue;
        case 0xf0: continue;
        case 0x26: if (!c->mode64) d.seg = ES; continue;
        case 0x2e: if (!c->mode64) d.seg = CS; continue;
        case 0x36: if (!c->mode64) d.seg = SS; continue;
        case 0x3e: if (!c->mode64) d.seg = DS; continue;
        case 0x64: d.seg = FS; continue;
        case 0x65: d.seg = GS; continue;
        }
        break;
    }
    if (c->mode64) {
        if ((b & 0xf0) == 0x40) {
            d.rex = b;
            d.rexw = (b >> 3) & 1; d.rexr = (b >> 2) & 1; d.rexx = (b >> 1) & 1; d.rexb = b & 1;
            b = fetch8(c, &d);
        }
        d.osz = d.rexw ? 8 : d.p66 ? 2 : 4;
        d.asz = a67 ? 4 : 8;
    } else {
        d.osz = (c->def32 ^ d.p66) ? 4 : 2;
        d.asz = (c->def32 ^ a67) ? 4 : 2;
    }
    d.opc = b;
    exec(c, &d, b);
    c->rip = c->mode64 ? d.ip : c->def32 ? (u32)d.ip : (d.ip & 0xffff);
    c->icount++;
}

static inline int irq_ready(Cpu *c) {
    Machine *m = c->m;
    if (!(c->rflags & F_IF) || c->shadow) return 0;
    if (m->chipset) return (c->apic_pending || (c->cpu_index == 0 && (m->pic[0].irr | m->pic[1].irr))) && apic_has_interrupt(m);
    return (m->pic[0].irr | m->pic[1].irr) && pic_has_pending(m);
}

static inline int irq_ack(Machine *m) { return m->chipset ? apic_ack(m) : pic_pending(m); }

/* One step: deliver a pending interrupt, or execute one instruction. */
static inline void step(Cpu *c) {
    if (__builtin_expect(irq_ready(c), 0)) {
        int v = irq_ack(c->m);
        if (v >= 0) { do_interrupt(c, v, c->rip, 0, 0, 0); return; }
    }
    c->shadow = 0;
    if (c->halted) return;
    step_insn(c);
}

/* 32-bit code the JIT can translate: protected (or compatibility) mode, not
 * virtual-8086, 32-bit code and stack segments, CS/DS/ES/SS with base 0 */
static inline int jit_flat32(const Cpu *c) {
    if (!c->def32 || !(c->cr0 & CR0_PE) || (c->rflags & (1u << 17))) return 0;
    if (!(c->seg[SS].attr & 0x4000)) return 0;
    return !(c->seg[CS].base | c->seg[DS].base | c->seg[ES].base | c->seg[SS].base);
}

/* Runs up to n steps (instructions or interrupt deliveries), stopping early
 * when the CPU halts. A single setjmp covers the whole batch; an exception
 * longjmps back here, is delivered, and counts as one step. */
u64 cpu_run(Cpu *c, u64 n) {
    c->run_i = 0;
    c->run_n = n;
    if (__builtin_setjmp(c->jmp)) {
        if (c->in_block) {
            /* instructions the block completed before the fault */
            c->run_i += c->icount - c->blk_icount0;
            c->in_block = 0;
        }
        deliver_exception(c);
        c->icount++;
        c->run_i++;
        c->jit_skip = 0;
    }
    while (c->run_i < c->run_n && !c->halted && !c->fatal) {
        if (__builtin_expect(irq_ready(c), 0)) {
            int v = irq_ack(c->m);
            if (v >= 0) { do_interrupt(c, v, c->rip, 0, 0, 0); c->run_i++; c->jit_skip = 0; continue; }
        }
        int shadowed = c->shadow;
        c->shadow = 0;
        /* the instruction after STI / MOV SS runs alone: an interrupt may be due right after it */
        if (c->jit_skip) c->jit_skip--;
        else if (!shadowed && c->jit_on && (c->mode64 || (!(jit_off & 0x100000) && jit_flat32(c))) && jit_run_block(c, c->run_n - c->run_i)) {
            c->run_i += c->icount - c->blk_icount0;
            if (c->yield_req) { c->yield_req = 0; break; }
            continue;
        }
        if (__builtin_expect(jit_prof_on, 0) && !c->mode64) {
            c->icount_legacy++;
            c->icount_legacy32 += c->def32;
            if (!(c->icount_legacy & 1023)) {
                extern u32 legacy_hist_pc[4096], legacy_hist_n[4096];
                u32 pc = (u32)(c->seg[CS].base + c->rip), h = (pc * 2654435761u) >> 20;
                for (int k = 0; k < 16; k++, h = (h + 1) & 4095) {
                    if (legacy_hist_pc[h] == pc) { legacy_hist_n[h]++; break; }
                    if (!legacy_hist_n[h]) { legacy_hist_pc[h] = pc; legacy_hist_n[h] = 1; break; }
                }
            }
        }
        step_insn(c);
        c->run_i++;
        if (c->yield_req) { c->yield_req = 0; break; }
    }
    return c->run_i;
}

/* entry points for generated code */
void cpu_jit_set_tpr(Cpu *c);
void cpu_jit_set_tpr(Cpu *c) { if (c->m->chipset) apic_set_tpr_from_cr8(c->m, c->cr8); }
void cpu_jit_set_zf(Cpu *c, u32 zf);
void cpu_jit_set_zf(Cpu *c, u32 zf) { set_flags_raw(c, (get_flags(c) & F_ARITH & ~(u64)F_ZF) | (zf ? F_ZF : 0)); }
void cpu_jit_rdtscp(Cpu *c);
void cpu_jit_rdtscp(Cpu *c) { u64 t = tsc(c); c->r[0] = (u32)t; c->r[2] = t >> 32; c->r[1] = (u32)c->tsc_aux; }

void cpu_step_insn(Cpu *c);
void cpu_step_insn(Cpu *c) { step_insn(c); }
int cpu_cond(Cpu *c, int cc);
int cpu_cond(Cpu *c, int cc) { return cond(c, cc); }
u32 cpu_lf_cf(Cpu *c);
u32 lfcf_hist[16];
u32 cpu_lf_cf(Cpu *c) { if (jit_prof_on) lfcf_hist[c->lf_op & 15]++; return lf_cf(c); }

/* TLB refill for generated code's slow path: host pointer for an access of
 * wsz & 0xff bytes at la (write if wsz >> 8), filling the TLB; 0 when it would
 * fault, is MMIO or crosses a page (the instruction is then interpreted, which
 * raises the fault itself). */
u8 *cpu_tlb_fill(Cpu *c, u32 lo, u32 hi, u32 wsz);
u8 *cpu_tlb_fill(Cpu *c, u32 lo, u32 hi, u32 wsz) {
    u64 la = lo | (u64)hi << 32;
    u32 sz = wsz & 0xff;
    if ((la & 0xfff) + sz > 0x1000) return 0;
    void *saved[5];
    e_memcpy(saved, c->jmp, sizeof saved);
    u8 *p = 0;
    u64 pa = 0;
    if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, la, (wsz >> 8) ? 1 : 0, &pa);
    e_memcpy(c->jmp, saved, sizeof saved);
    return p;
}
/* host address of an executable linear page (fills the exec TLB), 0 if not mapped/executable; never faults */
u8 *cpu_exec_probe(Cpu *c, u64 la);
u8 *cpu_exec_probe(Cpu *c, u64 la) {
    void *saved[5];
    e_memcpy(saved, c->jmp, sizeof saved);
    u8 *p = 0;
    if (__builtin_setjmp(c->jmp) == 0) p = lin_host(c, la & ~0xfffull, 2, 0);
    e_memcpy(c->jmp, saved, sizeof saved);
    return p;
}
void cpu_bt_flags(Cpu *c, u32 cf);
void cpu_bt_flags(Cpu *c, u32 cf) { put_flags_cf_of(c, cf, lf_of(c)); }
/* ROL/ROR: bits = CF | OF << 11 (other flags kept) */
void cpu_set_cf_of(Cpu *c, u32 bits);
void cpu_set_cf_of(Cpu *c, u32 bits) { put_flags_cf_of(c, bits & 1, (bits >> 11) & 1); }
u64 cpu_imul(Cpu *c, u64 a, u64 b, u32 sz);
u64 cpu_imul(Cpu *c, u64 a, u64 b, u32 sz) { return imul_trunc(c, a, b, (int)sz); }
u64 cpu_shift(Cpu *c, u32 op, u64 a, u32 cnt, u32 sz);
u64 cpu_shift(Cpu *c, u32 op, u64 a, u32 cnt, u32 sz) { return shift(c, (int)op, a, cnt, (int)sz); }
void cpu_muldiv(Cpu *c, u32 r, u64 src, u32 sz);
void cpu_muldiv(Cpu *c, u32 r, u64 src, u32 sz) {
    Dec d;
    e_memset(&d, 0, sizeof d);
    muldiv(c, &d, (int)r, src & MASK[sz], (int)sz);
}

void cpu_step(Cpu *c) {
    if (__builtin_setjmp(c->jmp)) {
        deliver_exception(c);
        c->icount++;
        return;
    }
    step(c);
}

void cpu_reset(Cpu *c, Machine *m) {
    e_memset(c, 0, sizeof *c);
    c->m = m;
    c->tlb_mode = 1;
    c->rflags = 2;
    c->fetch_la = 1;
    c->cr0 = CR0_ET;
    c->mxcsr = 0x1f80;
    c->fcw = 0x37f;
    c->ftw = 0xffff;
    c->pat = 0x0007040600070406ull;
    c->apic_base = 0xfee00900ull;
    c->gdtr.limit = 0xffff;
    c->idtr.limit = 0xffff;
    for (int i = 0; i < 6; i++) { c->seg[i].limit = 0xffff; c->seg[i].attr = 0x93; }
    c->seg[CS].attr = 0x9b;
    update_mode(c);
}

void cpu_set_reset_vector(Cpu *c);
void cpu_set_reset_vector(Cpu *c) {
    for (int i = 0; i < 6; i++) { c->seg[i].sel = 0; c->seg[i].base = 0; c->seg[i].limit = 0xffff; c->seg[i].attr = 0x93; }
    c->seg[CS].sel = 0xf000;
    c->seg[CS].base = 0xffff0000ull;
    c->seg[CS].attr = 0x9b;
    c->rip = 0xfff0;
    c->cr0 = 0x60000010ull;
    c->r[2] = 0x000006f1;          /* DX: processor signature */
    c->idtr.base = 0; c->idtr.limit = 0x3ff;
    c->gdtr.base = 0; c->gdtr.limit = 0xffff;
    c->cpl = 0;
    update_mode(c);
    tlb_flush(c);
}

/* application processor: INIT state (waits for a startup IPI) */
void cpu_init_ap(Cpu *c, Machine *m, int index);
void cpu_init_ap(Cpu *c, Machine *m, int index) {
    int jit = c->jit_on;
    cpu_reset(c, m);
    cpu_set_reset_vector(c);
    c->cpu_index = index;
    c->jit_on = jit;
    c->apic_base = 0xfee00800ull;
    c->wait_sipi = 1;
    c->halted = 1;
}
/* startup IPI: real mode at vector << 12 */
void cpu_sipi(Cpu *c, u32 vec);
void cpu_sipi(Cpu *c, u32 vec) {
    if (!c->wait_sipi) return;
    cpu_set_reset_vector(c);
    c->seg[CS].sel = (u16)(vec << 8);
    c->seg[CS].base = (u64)vec << 12;
    c->rip = 0;
    c->wait_sipi = 0;
    c->halted = 0;
    c->fetch_la = 1;
}

void cpu_set_flat32(Cpu *c);
void cpu_set_flat32(Cpu *c) {
    for (int i = 0; i < 6; i++) {
        c->seg[i].base = 0;
        c->seg[i].limit = 0xffffffffu;
        c->seg[i].attr = 0xc093;
        c->seg[i].sel = 0x18;
    }
    c->seg[CS].attr = 0xc09b;
    c->seg[CS].sel = 0x10;
    c->cr0 = CR0_PE | CR0_ET;
    c->cpl = 0;
    update_mode(c);
    tlb_flush(c);
}
