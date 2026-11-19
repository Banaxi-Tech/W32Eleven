/* PC platform devices: 8259 PIC pair, 8254 PIT, CMOS RTC, 16550 UART (COM1),
 * 8042 keyboard controller with PS/2 keyboard and mouse, PCI config space
 * (empty bus), and the odd system ports. Time is virtual (Machine.ns). */
#include "emu.h"

/* ================= 8259 PIC ================= */

static int pic_best(const Pic *p, u8 irr) {
    u8 pend = irr & (u8)~p->imr;
    if (!pend) return -1;
    for (int i = 0; i < 8; i++) {
        if (p->isr & (1u << i)) return -1;      /* a higher-priority IRQ is in service */
        if (pend & (1u << i)) return i;
    }
    return -1;
}

void pic_raise(Machine *m, int irq) {
    if (m->chipset) apic_isa_irq(m, irq);
    if (irq < 8) m->pic[0].irr |= (u8)(1u << irq);
    else m->pic[1].irr |= (u8)(1u << (irq - 8));
}

void pic_lower(Machine *m, int irq) {
    if (irq < 8) m->pic[0].irr &= (u8)~(1u << irq);
    else m->pic[1].irr &= (u8)~(1u << (irq - 8));
}

int pic_has_pending(Machine *m) {
    if (!((m->pic[0].irr | m->pic[1].irr))) return 0;
    int s = pic_best(&m->pic[1], m->pic[1].irr);
    return pic_best(&m->pic[0], (u8)(m->pic[0].irr | (s >= 0 ? 4 : 0))) >= 0;
}

int pic_pending(Machine *m) {
    Pic *ma = &m->pic[0], *sl = &m->pic[1];
    int s = pic_best(sl, sl->irr);
    int i = pic_best(ma, (u8)(ma->irr | (s >= 0 ? 4 : 0)));
    if (i < 0) return -1;
    if (i == 2 && s >= 0) {
        if (!ma->auto_eoi) ma->isr |= 4;
        ma->irr &= (u8)~4u;
        sl->irr &= (u8)~(1u << s);
        if (!sl->auto_eoi) sl->isr |= (u8)(1u << s);
        return sl->vector + s;
    }
    ma->irr &= (u8)~(1u << i);
    if (!ma->auto_eoi) ma->isr |= (u8)(1u << i);
    return ma->vector + i;
}

static void pic_write(Pic *p, int port, u8 v) {
    if (port == 0) {
        if (v & 0x10) {             /* ICW1 */
            p->icw_step = 1;
            p->icw4 = v & 1;
            p->imr = 0;
            p->isr = 0;
            p->read_isr = 0;
        } else if (v & 0x08) {      /* OCW3 */
            if (v & 2) p->read_isr = v & 1;
        } else {                    /* OCW2 */
            if (v & 0x20) {
                if (v & 0x40) p->isr &= (u8)~(1u << (v & 7));
                else for (int i = 0; i < 8; i++) if (p->isr & (1u << i)) { p->isr &= (u8)~(1u << i); break; }
            }
        }
        return;
    }
    switch (p->icw_step) {
    case 1: p->vector = v & 0xf8; p->icw_step = 2; break;
    case 2: p->icw_step = p->icw4 ? 3 : 0; break;
    case 3: p->auto_eoi = (v >> 1) & 1; p->icw_step = 0; break;
    default: p->imr = v; break;
    }
}

static u8 pic_read(Pic *p, int port) {
    if (port == 0) return p->read_isr ? p->isr : p->irr;
    return p->imr;
}

/* ================= 8254 PIT ================= */

#define PIT_HZ 1193182ull

static u64 pit_ticks(Machine *m, PitChannel *ch) {
    return (machine_now(m) - ch->start_ns) * PIT_HZ / 1000000000ull;
}

static u32 pit_reload(PitChannel *ch) { return ch->reload ? ch->reload : 0x10000; }

static u16 pit_count(Machine *m, PitChannel *ch) {
    u64 t = pit_ticks(m, ch);
    u32 r = pit_reload(ch);
    if (ch->mode == 2 || ch->mode == 3) return (u16)(r - (u32)(t % r));
    if (t >= r) return (u16)(0x10000 - ((t - r) & 0xffff));
    return (u16)(r - t);
}

static int pit_out(Machine *m, PitChannel *ch) {
    u64 t = pit_ticks(m, ch);
    u32 r = pit_reload(ch);
    switch (ch->mode) {
    case 0: case 1: return t >= r;
    case 3: return (t % r) < (r + 1) / 2;
    default: return (t % r) != r - 1;
    }
}

static u64 pit_period_ns(PitChannel *ch) {
    u64 p = (u64)pit_reload(ch) * 1000000000ull / PIT_HZ;
    return p ? p : 1;
}

static u64 g_pit0_next;     /* next IRQ0 time; ~0 = none */

static void pit_start(Machine *m, int i) {
    PitChannel *ch = &m->pit[i];
    ch->start_ns = machine_now(m);
    ch->out_pending = 1;
    if (i == 0) {
        g_pit0_next = ch->start_ns + pit_period_ns(ch);
        machine_event_changed(m);
    }
}

static void pit_write(Machine *m, int port, u8 v) {
    if (port == 3) {
        int i = v >> 6;
        if (i == 3) return;         /* read-back: not implemented */
        PitChannel *ch = &m->pit[i];
        int acc = (v >> 4) & 3;
        if (acc == 0) {
            ch->latch = pit_count(m, ch);
            ch->latched = 2;
            return;
        }
        ch->access = (u8)acc;
        ch->mode = (v >> 1) & 7;
        if (ch->mode > 5) ch->mode -= 4;
        ch->w_lo = 0;
        ch->rw_lo = 0;
        if (i == 0) { g_pit0_next = ~0ull; machine_event_changed(m); }
        return;
    }
    PitChannel *ch = &m->pit[port];
    switch (ch->access) {
    case 1: ch->reload = (u16)((ch->reload & 0xff00) | v); pit_start(m, port); break;
    case 2: ch->reload = (u16)((ch->reload & 0xff) | (v << 8)); pit_start(m, port); break;
    default:
        if (!ch->w_lo) { ch->reload = (u16)((ch->reload & 0xff00) | v); ch->w_lo = 1; }
        else { ch->reload = (u16)((ch->reload & 0xff) | (v << 8)); ch->w_lo = 0; pit_start(m, port); }
        break;
    }
}

static u8 pit_read(Machine *m, int port) {
    PitChannel *ch = &m->pit[port];
    u16 v;
    if (ch->latched) {
        v = ch->latch;
        if (ch->access == 3 || ch->access == 0) {
            u8 r = ch->latched == 2 ? (u8)v : (u8)(v >> 8);
            ch->latched--;
            return r;
        }
        ch->latched = 0;
        return ch->access == 2 ? (u8)(v >> 8) : (u8)v;
    }
    v = pit_count(m, ch);
    if (ch->access == 1) return (u8)v;
    if (ch->access == 2) return (u8)(v >> 8);
    u8 r = ch->rw_lo ? (u8)(v >> 8) : (u8)v;
    ch->rw_lo ^= 1;
    return r;
}

/* ================= CMOS RTC ================= */

static u8 bcd(u32 v) { return (u8)(((v / 10) << 4) | (v % 10)); }

static void rtc_fill(Machine *m) {
    u64 t = m->boot_unix + machine_now(m) / 1000000000ull;
    u32 secs = (u32)(t % 86400);
    i64 days = (i64)(t / 86400);
    /* civil from days (Howard Hinnant) */
    i64 z = days + 719468;
    i64 era = (z >= 0 ? z : z - 146096) / 146097;
    u32 doe = (u32)(z - era * 146097);
    u32 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    i64 y = (i64)yoe + era * 400;
    u32 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    u32 mp = (5 * doy + 2) / 153;
    u32 dd = doy - (153 * mp + 2) / 5 + 1;
    u32 mm = mp < 10 ? mp + 3 : mp - 9;
    if (mm <= 2) y++;
    m->cmos[0x00] = bcd(secs % 60);
    m->cmos[0x02] = bcd((secs / 60) % 60);
    m->cmos[0x04] = bcd(secs / 3600);
    m->cmos[0x06] = bcd((u32)((days + 4) % 7) + 1);
    m->cmos[0x07] = bcd(dd);
    m->cmos[0x08] = bcd(mm);
    m->cmos[0x09] = bcd((u32)(y % 100));
    m->cmos[0x32] = bcd((u32)(y / 100));
}

/* RTC interrupts: periodic (PIE, rate from register A), update-ended (UIE, once a second) */
static u64 g_rtc_periodic_next = ~0ull, g_rtc_update_next = ~0ull;

static u64 rtc_period_ns(Machine *m) {
    u32 rs = m->cmos[0x0a] & 15;
    if (!rs) return 0;
    if (rs < 3) rs += 7;
    return 1000000000ull * (1ull << (rs - 1)) / 32768;
}

static void rtc_reprogram(Machine *m) {
    u64 now = machine_now(m), p = rtc_period_ns(m);
    g_rtc_periodic_next = ((m->cmos[0x0b] & 0x40) && p) ? now + p : ~0ull;
    g_rtc_update_next = (m->cmos[0x0b] & 0x10) ? (now / 1000000000ull + 1) * 1000000000ull : ~0ull;
    machine_event_changed(m);
}

static void rtc_tick(Machine *m) {
    while (m->ns >= g_rtc_periodic_next) {
        m->cmos[0x0c] |= 0xc0;                      /* PF | IRQF */
        pic_raise(m, 8);
        u64 p = rtc_period_ns(m);
        g_rtc_periodic_next = p ? g_rtc_periodic_next + p : ~0ull;
        if (g_rtc_periodic_next != ~0ull && g_rtc_periodic_next <= m->ns) g_rtc_periodic_next = m->ns + p;
    }
    while (m->ns >= g_rtc_update_next) {
        m->cmos[0x0c] |= 0x90;                      /* UF | IRQF */
        pic_raise(m, 8);
        g_rtc_update_next += 1000000000ull;
    }
}

/* ================= 8042 + PS/2 ================= */

static void kbc_irq(Machine *m) {
    if (!m->kbc_q.count) return;
    int aux = m->kbc_q_aux[m->kbc_q.head];
    if (aux) { if (m->kbc_cfg & 2) pic_raise(m, 12); }
    else if (m->kbc_cfg & 1) pic_raise(m, 1);
}

static void kbc_push(Machine *m, u8 v, int aux) {
    ByteQueue *q = &m->kbc_q;
    if (q->count >= sizeof q->data) return;
    u32 i = (q->head + q->count) % sizeof q->data;
    q->data[i] = v;
    m->kbc_q_aux[i] = (u8)aux;
    q->count++;
    if (q->count == 1) kbc_irq(m);
}

static u8 kbc_pop(Machine *m) {
    ByteQueue *q = &m->kbc_q;
    if (!q->count) return 0;
    u8 v = q->data[q->head];
    q->head = (u8)((q->head + 1) % sizeof q->data);
    q->count--;
    kbc_irq(m);
    return v;
}

static void kbd_cmd(Machine *m, u8 v) {
    if (m->kbd_expect) {
        u8 what = m->kbd_expect;
        m->kbd_expect = 0;
        if (what == 0xf0 && v == 0) { kbc_push(m, 0xfa, 0); kbc_push(m, m->kbd_scanset, 0); return; }
        if (what == 0xf0) m->kbd_scanset = v;
        kbc_push(m, 0xfa, 0);
        return;
    }
    switch (v) {
    case 0xff: kbc_push(m, 0xfa, 0); kbc_push(m, 0xaa, 0); break;
    case 0xee: kbc_push(m, 0xee, 0); break;
    case 0xf2: kbc_push(m, 0xfa, 0); kbc_push(m, 0xab, 0); kbc_push(m, 0x83, 0); break;
    case 0xed: case 0xf0: case 0xf3: m->kbd_expect = v; kbc_push(m, 0xfa, 0); break;
    default: kbc_push(m, 0xfa, 0); break;
    }
}

static void mouse_cmd(Machine *m, u8 v) {
    if (m->mouse_expect) {
        u8 what = m->mouse_expect;
        m->mouse_expect = 0;
        if (what == 0xf3) {
            m->mouse_rate_seq[0] = m->mouse_rate_seq[1];
            m->mouse_rate_seq[1] = m->mouse_rate_seq[2];
            m->mouse_rate_seq[2] = v;
            if (m->mouse_rate_seq[0] == 200 && m->mouse_rate_seq[1] == 100 && m->mouse_rate_seq[2] == 80)
                m->mouse_id = 3;
        }
        kbc_push(m, 0xfa, 1);
        return;
    }
    switch (v) {
    case 0xff: m->mouse_stream = 0; m->mouse_id = 0;
               kbc_push(m, 0xfa, 1); kbc_push(m, 0xaa, 1); kbc_push(m, 0x00, 1); break;
    case 0xf2: kbc_push(m, 0xfa, 1); kbc_push(m, m->mouse_id, 1); break;
    case 0xf3: case 0xe8: m->mouse_expect = v; kbc_push(m, 0xfa, 1); break;
    case 0xf4: m->mouse_stream = 1; kbc_push(m, 0xfa, 1); break;
    case 0xf5: m->mouse_stream = 0; kbc_push(m, 0xfa, 1); break;
    case 0xf6: m->mouse_stream = 0; kbc_push(m, 0xfa, 1); break;
    case 0xe9: kbc_push(m, 0xfa, 1); kbc_push(m, 0, 1); kbc_push(m, 2, 1); kbc_push(m, 100, 1); break;
    default: kbc_push(m, 0xfa, 1); break;
    }
}

void machine_key(Machine *m, u8 sc) {
    if (m->kbc_cfg & 0x10) return;      /* keyboard clock disabled */
    kbc_push(m, sc, 0);
}

void machine_mouse(Machine *m, int dx, int dy, int buttons) {
    if (!m->mouse_stream || (m->kbc_cfg & 0x20)) return;
    if (dx > 255) dx = 255;
    if (dx < -256) dx = -256;
    if (dy > 255) dy = 255;
    if (dy < -256) dy = -256;
    u8 b0 = (u8)(0x08 | (buttons & 7) | (dx < 0 ? 0x10 : 0) | (dy < 0 ? 0x20 : 0));
    kbc_push(m, b0, 1);
    kbc_push(m, (u8)dx, 1);
    kbc_push(m, (u8)dy, 1);
    if (m->mouse_id == 3) kbc_push(m, 0, 1);
}

/* ================= port dispatch ================= */

static u32 g_unknown_logged;

static void unknown_port(Machine *m, const char *dir, u16 port) {
    if (g_unknown_logged < 32) {
        g_unknown_logged++;
        emu_log(m, "unhandled I/O %s port %x", dir, port);
    }
}

static u8 in8(Machine *m, u16 port) {
    switch (port) {
    case 0x20: case 0x21: return pic_read(&m->pic[0], port & 1);
    case 0xa0: case 0xa1: return pic_read(&m->pic[1], port & 1);
    case 0x40: case 0x41: case 0x42: return pit_read(m, port - 0x40);
    case 0x43: return 0;
    case 0x60: return kbc_pop(m);
    case 0x64: {
        u8 st = 0x04 | (m->kbc_q.count ? 1 : 0);
        if (m->kbc_q.count && m->kbc_q_aux[m->kbc_q.head]) st |= 0x20;
        return st;
    }
    case 0x61: {
        u8 v = m->port61 & 0x0f;
        if ((machine_now(m) / 15000) & 1) v |= 0x10;
        if (pit_out(m, &m->pit[2])) v |= 0x20;
        return v;
    }
    case 0x70: return m->cmos_index;
    case 0x71: {
        u8 i = m->cmos_index & 0x7f;
        if (i <= 9 || i == 0x32) rtc_fill(m);
        if (i == 0x0c) { u8 v = m->cmos[0x0c]; m->cmos[0x0c] = 0; return v; }   /* reading C clears the flags */
        return m->cmos[i];
    }
    case 0x92: return 0x02;
    case 0x3f8: case 0x3f9: case 0x3fa: case 0x3fb: case 0x3fc: case 0x3fd: case 0x3fe: case 0x3ff: {
        int r = port - 0x3f8;
        int dlab = (m->uart[3] & 0x80) != 0;
        switch (r) {
        case 0: if (dlab) return m->uart_dll; m->uart_rx_full = 0; return m->uart_rbr;
        case 1: return dlab ? m->uart_dlm : m->uart[1];
        case 2: return (u8)(0x01 | ((m->uart[2] & 1) ? 0xc0 : 0));
        case 5: return (u8)(0x60 | (m->uart_rx_full ? 1 : 0));
        case 6:
            if (m->uart[4] & 0x10) return (u8)(((m->uart[4] & 0x0c) << 4) | ((m->uart[4] & 0x03) << 4));
            return 0xb0;
        default: return m->uart[r];
        }
    }
    case 0x4d0: return m->pic[0].elcr;
    case 0x4d1: return m->pic[1].elcr;
    case 0xcfc: case 0xcfd: case 0xcfe: case 0xcff: return 0xff;
    }
    if ((port >= 0x1f0 && port <= 0x1f7) || (port >= 0x170 && port <= 0x177) || port == 0x3f6 || port == 0x376)
        return 0xff;                                                /* no IDE drives: floating bus */
    if ((port >= 0x2f8 && port <= 0x2ff) || (port >= 0x3e8 && port <= 0x3ef) || (port >= 0x2e8 && port <= 0x2ef))
        return 0xff;                                                /* COM2-4 absent */
    if ((port >= 0x3b0 && port <= 0x3df) || port == 0x80 || port == 0xe9 || (port >= 0x1ce && port <= 0x1d0))
        return 0xff;
    if (port >= 0x5658 && port <= 0x565b) return 0xff;             /* VMware backdoor probe: absent */
    unknown_port(m, "in", port);
    return 0xff;
}

static void out8(Machine *m, u16 port, u8 v) {
    switch (port) {
    case 0x20: case 0x21: pic_write(&m->pic[0], port & 1, v); return;
    case 0xa0: case 0xa1: pic_write(&m->pic[1], port & 1, v); return;
    case 0x40: case 0x41: case 0x42: case 0x43: pit_write(m, port - 0x40, v); return;
    case 0x60: {
        u8 cmd = m->kbc_cmd;
        m->kbc_cmd = 0;
        switch (cmd) {
        case 0x60: m->kbc_cfg = v; return;
        case 0xd1: m->kbc_out_port = v; return;
        case 0xd2: kbc_push(m, v, 0); return;
        case 0xd3: kbc_push(m, v, 1); return;
        case 0xd4: mouse_cmd(m, v); return;
        default: kbd_cmd(m, v); return;
        }
    }
    case 0x64:
        switch (v) {
        case 0x20: kbc_push(m, m->kbc_cfg, 0); return;
        case 0x60: case 0xd1: case 0xd2: case 0xd3: case 0xd4: m->kbc_cmd = v; return;
        case 0xa7: m->kbc_cfg |= 0x20; return;
        case 0xa8: m->kbc_cfg &= (u8)~0x20; return;
        case 0xa9: kbc_push(m, 0x00, 0); return;
        case 0xaa: kbc_push(m, 0x55, 0); return;
        case 0xab: kbc_push(m, 0x00, 0); return;
        case 0xad: m->kbc_cfg |= 0x10; return;
        case 0xae: m->kbc_cfg &= (u8)~0x10; return;
        case 0xd0: kbc_push(m, m->kbc_out_port, 0); return;
        case 0xfe: emu_reset_request(m->cur);
        default: return;
        }
    case 0x61: {
        u8 old = m->port61;
        m->port61 = v;
        if (!(old & 1) && (v & 1)) pit_start(m, 2);   /* gate rising edge restarts channel 2 */
        return;
    }
    case 0x70: m->cmos_index = v; return;
    case 0x71: {
        u8 i = m->cmos_index & 0x7f;
        if (i == 0x0a) { m->cmos[i] = (u8)((m->cmos[i] & 0x80) | (v & 0x7f)); rtc_reprogram(m); }
        else if (i == 0x0b) { m->cmos[i] = v; rtc_reprogram(m); }
        else if (i >= 0x0e) m->cmos[i] = v;
        return;
    }
    case 0x80: case 0x92: return;
    case 0xe9: if (m->host.debug_out) m->host.debug_out(m->host.ctx, v); return;
    case 0x3f8: case 0x3f9: case 0x3fa: case 0x3fb: case 0x3fc: case 0x3fd: case 0x3fe: case 0x3ff: {
        int r = port - 0x3f8;
        int dlab = (m->uart[3] & 0x80) != 0;
        if (r == 0) {
            if (dlab) { m->uart_dll = v; return; }
            if (m->uart[4] & 0x10) { m->uart_rbr = v; m->uart_rx_full = 1; return; }
            if (m->host.serial_out) m->host.serial_out(m->host.ctx, v);
            return;
        }
        if (r == 1 && dlab) { m->uart_dlm = v; return; }
        if (r == 5 || r == 6) return;
        m->uart[r] = v;
        return;
    }
    case 0x4d0: m->pic[0].elcr = v; return;
    case 0x4d1: m->pic[1].elcr = v; return;
    case 0xcfc: case 0xcfd: case 0xcfe: case 0xcff: return;
    }
    if ((port >= 0x1f0 && port <= 0x1f7) || (port >= 0x170 && port <= 0x177) || port == 0x3f6 || port == 0x376)
        return;
    if ((port >= 0x2f8 && port <= 0x2ff) || (port >= 0x3e8 && port <= 0x3ef) || (port >= 0x2e8 && port <= 0x2ef))
        return;
    if ((port >= 0x3b0 && port <= 0x3df) || (port >= 0x1ce && port <= 0x1d0)) return;
    unknown_port(m, "out", port);
}

#define MASK_SZ(sz) ((sz) == 4 ? 0xffffffffull : (sz) == 2 ? 0xffffull : 0xffull)

u32 io_in(Machine *m, u16 port, int size) {
    if (port == 0xcf8 && size == 4) return m->pci_addr;
    if (m->chipset) {
        u32 v;
        if (ide_io(m, port, 0, &v, size) || ide_bm_io(m, port, 0, &v, size) || fwcfg_io(m, port, 0, &v, size) ||
            vbe_io(m, port, 0, &v, size)) {
            if (m->trace_bios >= 2 && m->cpu.mode64 && ((port >= 0x170 && port <= 0x177 && port != 0x170) || port == 0x376 ||
                (port >= 0x1f1 && port <= 0x1f7) || port == 0x3f6 || (port >= 0xc000 && port <= 0xc00f))) {
                static int n;
                if (n++ < 400) emu_log(m, "IO64 in %x -> %x", port, v & (u32)MASK_SZ(size));
            }
            return v & (u32)MASK_SZ(size);
        }
        if (port >= 0xafe0 && port <= 0xafe3) return 0;   /* GPE0 status/enable: nothing pending */
    }
    if (m->chipset && port >= 0xcfc && port <= 0xcff) {
        if (!(m->pci_addr & 0x80000000u)) return 0xffffffffu & (u32)MASK_SZ(size);
        u32 r = pci_cfg_read(m, (m->pci_addr & 0xfffffffcu) | (port & 3u), size);
        if (m->trace_bios >= 2 && m->cpu.mode64 && ((m->pci_addr >> 8) & 0xffff) == 0x0009) {
            static int n;
            if (n++ < 3000) emu_log(m, "PCI64 read %02x:%02x.%x off %02x size %d -> %x", (m->pci_addr >> 16) & 0xff,
                                   (m->pci_addr >> 11) & 31, (m->pci_addr >> 8) & 7, (m->pci_addr & 0xfc) | (port & 3), size, r);
        }
        return r;
    }
    if (m->chipset) {
        u32 v;
        if (port == 0x402) return 0xe9;                   /* QEMU debugcon readback: SeaBIOS logs here */
        if (chipset_pm_io(m, port, 0, &v, size)) return v;
        if (port <= 0x1f || (port >= 0xc0 && port <= 0xdf) || (port >= 0x80 && port <= 0x8f))
            return 0;                                     /* ISA DMA controllers / page registers */
    }
    if (port == 0xcfc && size == 4) return 0xffffffffu;   /* no PCI devices */
    u32 v = 0;
    for (int i = 0; i < size; i++) v |= (u32)in8(m, (u16)(port + i)) << (8 * i);
    return v;
}

void io_out(Machine *m, u16 port, u32 v, int size) {
    if (port == 0xcf8 && size == 4) { m->pci_addr = v; return; }
    if (m->chipset && (ide_io(m, port, 1, &v, size) || ide_bm_io(m, port, 1, &v, size) || fwcfg_io(m, port, 1, &v, size) ||
                       vbe_io(m, port, 1, &v, size))) return;
    if (m->chipset && port >= 0xafe0 && port <= 0xafe3) return;
    if (m->chipset && port >= 0xcfc && port <= 0xcff) {
        if (m->trace_bios >= 2 && m->cpu.mode64 && ((m->pci_addr >> 8) & 0xffff) == 0x0009) {
            static int n;
            if (n++ < 3000) emu_log(m, "PCI64 write %02x:%02x.%x off %02x size %d <- %x", (m->pci_addr >> 16) & 0xff,
                                   (m->pci_addr >> 11) & 31, (m->pci_addr >> 8) & 7, (m->pci_addr & 0xfc) | (port & 3), size, v);
        }
        if (m->pci_addr & 0x80000000u) pci_cfg_write(m, (m->pci_addr & 0xfffffffcu) | (port & 3u), v, size);
        return;
    }
    if (m->chipset && port == 0xcf9 && (v & 4)) emu_reset_request(m->cur);
    if (m->chipset && port == 0x402) { if (m->host.debug_out) m->host.debug_out(m->host.ctx, (u8)v); return; }
    if (m->chipset) {
        if (chipset_pm_io(m, port, 1, &v, size)) return;
        if ((port <= 0x1f) || (port >= 0xc0 && port <= 0xdf) || (port >= 0x80 && port <= 0x8f)) return;
    }
    if (port == 0xcfc && size == 4) return;
    for (int i = 0; i < size; i++) out8(m, (u16)(port + i), (u8)(v >> (8 * i)));
}

/* ================= time ================= */

void devices_init(Machine *m) {
    g_pit0_next = ~0ull;
    g_rtc_periodic_next = g_rtc_update_next = ~0ull;
    g_unknown_logged = 0;
    m->pic[0].imr = m->pic[1].imr = 0xff;
    m->pic[0].vector = 0x08;
    m->pic[1].vector = 0x70;
    for (int i = 0; i < 3; i++) { m->pit[i].mode = 3; m->pit[i].access = 3; }
    m->cmos[0x0a] = 0x26;
    m->cmos[0x0b] = 0x02;
    m->cmos[0x0d] = 0x80;
    m->cmos[0x10] = 0x00;          /* no floppy */
    m->cmos[0x14] = 0x06;          /* equipment: FPU, colour display */
    u64 kb = m->ram_size >> 10;
    u64 ext = kb > 1024 ? kb - 1024 : 0;
    if (ext > 0xffff) ext = 0xffff;
    m->cmos[0x15] = 0x80; m->cmos[0x16] = 0x02;          /* 640 KiB base */
    m->cmos[0x17] = m->cmos[0x30] = (u8)ext;
    m->cmos[0x18] = m->cmos[0x31] = (u8)(ext >> 8);
    u64 hi = m->ram_size > (16ull << 20) ? (m->ram_size - (16ull << 20)) >> 16 : 0;
    if (hi > 0xffff) hi = 0xffff;
    m->cmos[0x34] = (u8)hi; m->cmos[0x35] = (u8)(hi >> 8);
    m->kbc_cfg = 0x47;            /* IRQ1+IRQ12 on, system flag, set-1 translation (BIOS default) */
    m->kbd_scanset = 2;
    m->uart[2] = 0;
    m->boot_unix = m->host.wall_unix ? m->host.wall_unix(m->host.ctx) : 1767225600ull;
}

void devices_tick(Machine *m) {
    if (m->chipset) apic_tick(m);
    rtc_tick(m);
    while (m->ns >= g_pit0_next) {
        PitChannel *ch = &m->pit[0];
        pic_raise(m, 0);
        if (ch->mode == 2 || ch->mode == 3) {
            u64 p = pit_period_ns(ch);
            g_pit0_next += p;
            if (g_pit0_next < m->ns) g_pit0_next = m->ns + p;   /* do not storm after a long stall */
        } else {
            g_pit0_next = ~0ull;
        }
    }
}

u64 devices_next_event(Machine *m) {
    u64 n = g_pit0_next;
    if (g_rtc_periodic_next < n) n = g_rtc_periodic_next;
    if (g_rtc_update_next < n) n = g_rtc_update_next;
    if (m->chipset) { u64 a = apic_next_event(m); if (a < n) n = a; }
    return n;
}
