/* Local APIC (0xFEE00000) and IOAPIC (0xFEC00000) for firmware boots.
 * One CPU. The 8259 pair stays connected through LINT0 (virtual wire / ExtINT)
 * and is used whenever the local APIC is not taking over. */
#include "emu.h"

#define LAPIC_BASE 0xfee00000ull
#define IOAPIC_BASE 0xfec00000ull
#define APIC_BUS_NS 10u              /* 100 MHz APIC bus clock */

typedef struct {
    u32 id, tpr, ldr, dfr, svr, esr, icr_lo, icr_hi;
    u32 lvt[7];                      /* timer, thermal, perf, LINT0, LINT1, error, CMCI */
    u32 irr[8], isr[8], tmr[8];
    u32 timer_init, timer_div;
    u64 timer_start, timer_next;     /* ns; timer_next ~0 = idle */
} Lapic;

typedef struct {
    u32 index, id;
    u64 redir[24];
    u32 level;                       /* current input line levels */
} Ioapic;

static Lapic Ls[MAX_CPUS];
#define L (Ls[m->cur->cpu_index])   /* the local APIC of the CPU that is executing */
static Ioapic IO;

enum { LVT_TIMER, LVT_THERMAL, LVT_PERF, LVT_LINT0, LVT_LINT1, LVT_ERROR, LVT_CMCI };

static int bit_get(const u32 *v, int n) { return (v[n >> 5] >> (n & 31)) & 1; }
static void bit_set(u32 *v, int n) { v[n >> 5] |= 1u << (n & 31); }
static void bit_clr(u32 *v, int n) { v[n >> 5] &= ~(1u << (n & 31)); }
static int highest(const u32 *v) {
    for (int i = 7; i >= 0; i--) if (v[i]) return i * 32 + 31 - __builtin_clz(v[i]);
    return -1;
}

static int lapic_on(Machine *m) { return (m->cur->apic_base & 0x800) != 0; }
static int lapic_sw_on(Machine *m) { return lapic_on(m) && (L.svr & 0x100); }

static void update_pending_i(Machine *m, int i) { m->cpus[i]->apic_pending = highest(Ls[i].irr) >= 0; }
static void update_pending(Machine *m) { update_pending_i(m, m->cur->cpu_index); }

/* a fixed interrupt arrives at local APIC i */
static void accept_on(Machine *m, int i, int vec, int level) {
    Lapic *l = &Ls[i];
    if (vec < 16) { l->esr |= 0x40; return; }   /* illegal vector */
    bit_set(l->irr, vec);
    if (level) bit_set(l->tmr, vec); else bit_clr(l->tmr, vec);
    update_pending_i(m, i);
}
static void accept(Machine *m, int vec, int level) { accept_on(m, m->cur->cpu_index, vec, level); }

/* CPUs addressed by a destination (physical APIC ID or logical flat/cluster) */
static u32 dest_mask(Machine *m, u32 dest, int logical) {
    u32 mask = 0;
    for (int i = 0; i < m->ncpu; i++) {
        Lapic *l = &Ls[i];
        if (!logical) { if (dest == 0xff || l->id == dest) mask |= 1u << i; continue; }
        u32 ld = l->ldr >> 24;
        if ((l->dfr >> 28) == 0xf) { if (ld & dest) mask |= 1u << i; }                       /* flat */
        else if ((ld >> 4) == (dest >> 4) && (ld & dest & 0xf)) mask |= 1u << i;             /* cluster */
    }
    return mask;
}

/* fixed or lowest-priority delivery of vec to the CPUs in mask */
static void deliver_mask(Machine *m, u32 mask, int mode, int vec, int level) {
    if (mode == 1 && mask) {                                   /* lowest priority: one CPU */
        int best = -1;
        for (int i = 0; i < m->ncpu; i++) if (mask >> i & 1) if (best < 0 || (Ls[i].tpr & 0xf0) < (Ls[best].tpr & 0xf0)) best = i;
        mask = 1u << best;
    }
    for (int i = 0; i < m->ncpu; i++) if (mask >> i & 1) accept_on(m, i, vec, level);
}

void cpu_init_ap(Cpu *c, Machine *m, int index);
void cpu_sipi(Cpu *c, u32 vec);
static void lapic_reset(Machine *m, int i);

/* interprocessor interrupt from the executing CPU's ICR */
static void send_ipi(Machine *m, u32 lo, u32 hi) {
    int mode = (lo >> 8) & 7, shorthand = (lo >> 18) & 3, me = m->cur->cpu_index;
    u32 mask;
    if (shorthand == 1) mask = 1u << me;
    else if (shorthand == 2) mask = (1u << m->ncpu) - 1;
    else if (shorthand == 3) mask = ((1u << m->ncpu) - 1) & ~(1u << me);
    else mask = dest_mask(m, hi >> 24, (lo >> 11) & 1);
    switch (mode) {
    case 0: case 1: deliver_mask(m, mask, mode, (int)(lo & 0xff), 0); break;
    case 5:                                                    /* INIT (the level de-assert form does nothing) */
        if (!((lo >> 14) & 1) && ((lo >> 15) & 1)) break;
        for (int i = 1; i < m->ncpu; i++) if (mask >> i & 1) { cpu_init_ap(m->cpus[i], m, i); lapic_reset(m, i); }
        break;
    case 6:                                                    /* STARTUP */
        for (int i = 1; i < m->ncpu; i++) if (mask >> i & 1) cpu_sipi(m->cpus[i], lo & 0xff);
        break;
    default: break;                                            /* SMI, NMI: not modelled */
    }
}

static void lapic_reset(Machine *m, int i) {
    Lapic *l = &Ls[i];
    e_memset(l, 0, sizeof *l);
    for (int k = 0; k < 7; k++) l->lvt[k] = 0x10000;
    l->id = (u32)i;
    l->dfr = 0xffffffffu;
    l->svr = 0xff;
    l->timer_next = ~0ull;
    m->cpus[i]->apic_base = 0xfee00800ull | (i == 0 ? 0x100 : 0);   /* enabled; BSP flag on CPU 0 */
    m->cpus[i]->apic_pending = 0;
}

void apic_reset(Machine *m) {
    e_memset(&IO, 0, sizeof IO);
    for (int i = 0; i < 24; i++) IO.redir[i] = 0x10000;
    for (int i = 0; i < m->ncpu; i++) lapic_reset(m, i);
}

/* ---- priority ---- */

static u32 ppr(Machine *m) {
    int isr = highest(L.isr);
    u32 isrv = isr < 0 ? 0 : (u32)isr & 0xf0;
    return (L.tpr & 0xf0) >= isrv ? L.tpr & 0xff : isrv;
}

/* PIC interrupts reach the CPU: APIC disabled, or LINT0 unmasked in ExtINT mode */
static int pic_routed(Machine *m) {
    if (m->cur->cpu_index) return 0;                          /* the 8259 is wired to the BSP's LINT0 only */
    if (!lapic_sw_on(m)) return 1;
    u32 l = L.lvt[LVT_LINT0];
    return !(l & 0x10000) && ((l >> 8) & 7) == 7;
}

int apic_has_interrupt(Machine *m) {
    if (lapic_sw_on(m)) {
        int v = highest(L.irr);
        if (v >= 0 && ((u32)v & 0xf0) > (ppr(m) & 0xf0)) return 1;
    }
    return pic_routed(m) && pic_has_pending(m);
}

/* acknowledge: returns the vector to deliver, or -1 */
int apic_ack(Machine *m) {
    if (lapic_sw_on(m)) {
        int v = highest(L.irr);
        if (v >= 0 && ((u32)v & 0xf0) > (ppr(m) & 0xf0)) {
            bit_clr(L.irr, v);
            bit_set(L.isr, v);
            update_pending(m);
            return v;
        }
    }
    if (pic_routed(m)) return pic_pending(m);
    return -1;
}

/* ---- IOAPIC ---- */

static void ioapic_deliver(Machine *m, int pin) {
    u64 e = IO.redir[pin];
    if (e & 0x10000) return;                       /* masked */
    int level = (e >> 15) & 1;
    if (level && (e & 0x4000)) return;             /* remote IRR still set */
    int mode = (e >> 8) & 7;
    if (mode == 7) return;                         /* ExtINT through the IOAPIC: not used */
    if (level) IO.redir[pin] |= 0x4000;
    if (m->ncpu > 1) {
        u32 mask = dest_mask(m, (u32)(e >> 56), (int)((e >> 11) & 1));
        if (!mask) mask = 1;                                   /* misprogrammed: the BSP */
        deliver_mask(m, mask, mode == 1 ? 1 : 0, (int)(e & 0xff), level);
        return;
    }
    accept(m, (int)(e & 0xff), level);
}

void ioapic_set_irq(Machine *m, int gsi, int high) {
    if (gsi < 0 || gsi >= 24) return;
    u32 bit = 1u << gsi;
    int was = (IO.level & bit) != 0;
    int pol_low = (IO.redir[gsi] >> 13) & 1;
    int asserted = pol_low ? !high : high;
    if (asserted) IO.level |= bit; else IO.level &= ~bit;
    int level_trig = (IO.redir[gsi] >> 15) & 1;
    if (asserted && (level_trig || !was)) ioapic_deliver(m, gsi);
}

/* ISA IRQ edge (as the devices raise them): IRQ0 is wired to GSI2 (MADT override) */
void apic_isa_irq(Machine *m, int irq) {
    if (!m->chipset) return;
    int gsi = irq == 0 ? 2 : irq;
    ioapic_set_irq(m, gsi, 1);
    ioapic_set_irq(m, gsi, 0);
}

static void eoi(Machine *m) {
    int v = highest(L.isr);
    if (v < 0) return;
    bit_clr(L.isr, v);
    if (bit_get(L.tmr, v)) {
        bit_clr(L.tmr, v);
        for (int i = 0; i < 24; i++)
            if ((IO.redir[i] & 0xff) == (u64)v && (IO.redir[i] & 0x4000)) {
                IO.redir[i] &= ~0x4000ull;
                if (IO.level & (1u << i)) ioapic_deliver(m, i);
            }
    }
    update_pending(m);
}

/* ---- timer ---- */

static u32 timer_divisor_l(const Lapic *l) {
    u32 d = (l->timer_div & 3) | ((l->timer_div >> 1) & 4);
    return d == 7 ? 1 : 2u << d;
}
static u64 timer_period_l(const Lapic *l) { return (u64)l->timer_init * timer_divisor_l(l) * APIC_BUS_NS; }
#define timer_divisor() timer_divisor_l(&L)
#define timer_period_ns() timer_period_l(&L)

static void timer_arm(Machine *m) {
    L.timer_start = machine_now(m);
    L.timer_next = L.timer_init ? L.timer_start + timer_period_ns() : ~0ull;
    machine_event_changed(m);
}

static u32 timer_current(Machine *m) {
    if (!L.timer_init || L.timer_next == ~0ull) return 0;
    u64 el = (machine_now(m) - L.timer_start) / ((u64)timer_divisor() * APIC_BUS_NS);
    if (((L.lvt[LVT_TIMER] >> 17) & 3) == 1) return L.timer_init - (u32)(el % L.timer_init);
    return el >= L.timer_init ? 0 : L.timer_init - (u32)el;
}

u64 apic_next_event(Machine *m) {
    u64 t = ~0ull;
    for (int i = 0; i < m->ncpu; i++) if (Ls[i].timer_next < t) t = Ls[i].timer_next;
    return t;
}

void apic_tick(Machine *m) {
    for (int i = 0; i < m->ncpu; i++) {
        Lapic *l = &Ls[i];
        while (l->timer_next != ~0ull && m->ns >= l->timer_next) {
            if (!(l->lvt[LVT_TIMER] & 0x10000) && (m->cpus[i]->apic_base & 0x800) && (l->svr & 0x100))
                accept_on(m, i, (int)(l->lvt[LVT_TIMER] & 0xff), 0);
            if (((l->lvt[LVT_TIMER] >> 17) & 3) == 1 && l->timer_init) {
                l->timer_start = l->timer_next;
                l->timer_next += timer_period_l(l);
                if (l->timer_next <= m->ns) l->timer_next = m->ns + timer_period_l(l);
            } else {
                l->timer_next = ~0ull;
            }
        }
    }
}

/* ---- MMIO ---- */

static u32 lapic_read(Machine *m, u32 off) {
    switch (off) {
    case 0x20: return L.id << 24;
    case 0x30: return 0x00050014;
    case 0x80: return L.tpr;
    case 0x90: return 0;
    case 0xa0: return ppr(m);
    case 0xd0: return L.ldr;
    case 0xe0: return L.dfr;
    case 0xf0: return L.svr;
    case 0x280: return L.esr;
    case 0x2f0: return L.lvt[LVT_CMCI];
    case 0x300: return L.icr_lo & ~0x1000u;
    case 0x310: return L.icr_hi;
    case 0x320: return L.lvt[LVT_TIMER];
    case 0x330: return L.lvt[LVT_THERMAL];
    case 0x340: return L.lvt[LVT_PERF];
    case 0x350: return L.lvt[LVT_LINT0];
    case 0x360: return L.lvt[LVT_LINT1];
    case 0x370: return L.lvt[LVT_ERROR];
    case 0x380: return L.timer_init;
    case 0x390: return timer_current(m);
    case 0x3e0: return L.timer_div;
    }
    if (off >= 0x100 && off < 0x180) return L.isr[(off - 0x100) >> 4];
    if (off >= 0x180 && off < 0x200) return L.tmr[(off - 0x180) >> 4];
    if (off >= 0x200 && off < 0x280) return L.irr[(off - 0x200) >> 4];
    return 0;
}

static void lapic_write(Machine *m, u32 off, u32 v) {
    switch (off) {
    case 0x20: L.id = v >> 24; return;
    case 0x80: L.tpr = v & 0xff; m->cur->cr8 = (v >> 4) & 15; return;
    case 0xb0: eoi(m); return;
    case 0xd0: L.ldr = v & 0xff000000u; return;
    case 0xe0: L.dfr = v | 0x0fffffffu; return;
    case 0xf0:
        L.svr = v & 0x13ff;
        if (!(v & 0x100)) for (int i = 0; i < 7; i++) L.lvt[i] |= 0x10000;
        return;
    case 0x280: L.esr = 0; return;
    case 0x300: {
        L.icr_lo = v & ~0x1000u;
        if (m->ncpu > 1) { send_ipi(m, v, L.icr_hi); return; }
        int mode = (v >> 8) & 7, shorthand = (v >> 18) & 3;
        int self = shorthand == 1 || shorthand == 2 || (shorthand == 0 && (L.icr_hi >> 24) == L.id);
        if (mode == 0 && self) accept(m, (int)(v & 0xff), 0);   /* fixed IPI to ourselves */
        return;                                                 /* INIT/SIPI/others: one CPU */
    }
    case 0x310: L.icr_hi = v & 0xff000000u; return;
    case 0x320: L.lvt[LVT_TIMER] = v & 0x700ff; if (!(L.svr & 0x100)) L.lvt[LVT_TIMER] |= 0x10000; return;
    case 0x330: L.lvt[LVT_THERMAL] = v & 0x107ff; return;
    case 0x340: L.lvt[LVT_PERF] = v & 0x107ff; return;
    case 0x350: L.lvt[LVT_LINT0] = v & 0x1a7ff; machine_event_changed(m); return;
    case 0x360: L.lvt[LVT_LINT1] = v & 0x1a7ff; return;
    case 0x370: L.lvt[LVT_ERROR] = v & 0x100ff; return;
    case 0x2f0: L.lvt[LVT_CMCI] = v & 0x107ff; return;
    case 0x380: L.timer_init = v; timer_arm(m); return;
    case 0x3e0: L.timer_div = v & 0xb; return;
    }
}

static u32 ioapic_read(u32 off) {
    if (off == 0x00) return IO.index;
    if (off != 0x10) return 0;
    u32 r = IO.index & 0xff;
    if (r == 0) return IO.id << 24;
    if (r == 1) return 0x00170011;                    /* 24 entries, version 0x11 */
    if (r == 2) return IO.id << 24;
    if (r >= 0x10 && r < 0x10 + 48) {
        u64 e = IO.redir[(r - 0x10) / 2];
        return (r & 1) ? (u32)(e >> 32) : (u32)e;
    }
    return 0;
}

static void ioapic_write(Machine *m, u32 off, u32 v) {
    if (off == 0x00) { IO.index = v & 0xff; return; }
    if (off == 0x40) {                                /* EOI register (version 0x20+): not advertised */
        return;
    }
    if (off != 0x10) return;
    u32 r = IO.index & 0xff;
    if (r == 0) { IO.id = (v >> 24) & 15; return; }
    if (r >= 0x10 && r < 0x10 + 48) {
        int pin = (int)(r - 0x10) / 2;
        u64 e = IO.redir[pin];
        if (r & 1) e = (e & 0xffffffffull) | ((u64)v << 32);
        else e = (e & 0xffffffff00004000ull) | (v & ~0x5000u);   /* remote IRR and delivery status read-only */
        IO.redir[pin] = e;
        /* a level-triggered line that is still asserted fires when unmasked */
        if (!(e & 0x10000) && ((e >> 15) & 1) && (IO.level & (1u << pin))) ioapic_deliver(m, pin);
    }
}

int apic_mmio(Machine *m, u64 pa, int write, u64 *v, int sz) {
    (void)sz;
    if (pa >= LAPIC_BASE && pa < LAPIC_BASE + 0x1000 && lapic_on(m)) {
        u32 off = (u32)(pa - LAPIC_BASE) & 0xff0;
        if (write) lapic_write(m, off, (u32)*v); else *v = lapic_read(m, off);
        return 1;
    }
    if (pa >= IOAPIC_BASE && pa < IOAPIC_BASE + 0x100) {
        u32 off = (u32)(pa - IOAPIC_BASE) & 0xf0;
        if (write) ioapic_write(m, off, (u32)*v); else *v = ioapic_read(off);
        return 1;
    }
    return 0;
}

void apic_set_tpr_from_cr8(Machine *m, u64 cr8) { (void)m; L.tpr = (u32)(cr8 & 15) << 4; }

void apic_dump(Machine *m) {
    emu_log(m, "LAPIC: svr %x tpr %x timer lvt %x init %x div %x lint0 %x lint1 %x isr-top %d irr-top %d",
            L.svr, L.tpr, L.lvt[LVT_TIMER], L.timer_init, L.timer_div, L.lvt[LVT_LINT0], L.lvt[LVT_LINT1],
            highest(L.isr), highest(L.irr));
    for (int i = 0; i < 24; i++)
        if (!(IO.redir[i] & 0x10000)) emu_log(m, "IOAPIC pin %d: %llx (unmasked)", i, (unsigned long long)IO.redir[i]);
    emu_log(m, "PIC masks %02x %02x, CMOS A %02x B %02x", m->pic[0].imr, m->pic[1].imr, m->cmos[0x0a], m->cmos[0x0b]);
}
