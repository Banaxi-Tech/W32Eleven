/* PC chipset for firmware boot: i440FX host bridge (PAM shadow RAM), PIIX3
 * ISA bridge, PIIX3 IDE function, PIIX4 power management, Bochs VGA; the
 * PCI configuration mechanism #1; and the firmware ROM mapping
 * (top of 4 GiB, plus 0xE0000-0xFFFFF through PAM). */
#include "emu.h"

void tlb_flush(Cpu *c);

typedef struct {
    u8 cfg[256];
    u32 bar_size[7];     /* 0-5 BARs, 6 = expansion ROM; 0 = absent */
    u8 bar_io[6];
} PciFn;

enum { DEV_HOST, DEV_ISA, DEV_IDE, DEV_PM, DEV_VGA, NDEV };
static const u8 pci_slot[NDEV] = { 0, 1, 1, 1, 2 };
static const u8 pci_func[NDEV] = { 0, 0, 1, 3, 0 };
static PciFn g_pci[NDEV];

static void w16(u8 *p, u16 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); }
static void w32(u8 *p, u32 v) { w16(p, (u16)v); w16(p + 2, (u16)(v >> 16)); }
static u32 r32(const u8 *p) { return (u32)p[0] | (u32)p[1] << 8 | (u32)p[2] << 16 | (u32)p[3] << 24; }

static void pci_ident(PciFn *f, u16 ven, u16 dev, u32 class_rev, u8 hdr) {
    e_memset(f, 0, sizeof *f);
    w16(f->cfg, ven); w16(f->cfg + 2, dev);
    w32(f->cfg + 8, class_rev);
    f->cfg[0x0e] = hdr;
}

void chipset_init(Machine *m) {
    (void)m;
    pci_ident(&g_pci[DEV_HOST], 0x8086, 0x1237, 0x06000002, 0x00);
    pci_ident(&g_pci[DEV_ISA], 0x8086, 0x7000, 0x06010000, 0x80);
    g_pci[DEV_ISA].cfg[0x60] = g_pci[DEV_ISA].cfg[0x61] = g_pci[DEV_ISA].cfg[0x62] = g_pci[DEV_ISA].cfg[0x63] = 0x80;
    pci_ident(&g_pci[DEV_IDE], 0x8086, 0x7010, 0x01018000, 0x00);
    g_pci[DEV_IDE].bar_size[4] = 16; g_pci[DEV_IDE].bar_io[4] = 1;
    w32(g_pci[DEV_IDE].cfg + 0x20, 1);
    pci_ident(&g_pci[DEV_PM], 0x8086, 0x7113, 0x06800003, 0x00);
    g_pci[DEV_PM].cfg[0x3d] = 1;                     /* INTA */
    pci_ident(&g_pci[DEV_VGA], 0x1234, 0x1111, 0x03000002, 0x00);
    g_pci[DEV_VGA].bar_size[0] = 16u << 20;          /* linear framebuffer, prefetchable */
    w32(g_pci[DEV_VGA].cfg + 0x10, 0x8);
    g_pci[DEV_VGA].bar_size[2] = 4096;               /* bochs-display MMIO registers */
    if (m->vga_rom) g_pci[DEV_VGA].bar_size[6] = 64u << 10;
    for (int i = 0; i < NDEV; i++) w16(g_pci[i].cfg + 6, 0x0280);   /* status: fast back-to-back, medium */
}

static PciFn *pci_find_fn(u32 addr) {
    u32 bus = (addr >> 16) & 0xff, slot = (addr >> 11) & 31, fn = (addr >> 8) & 7;
    if (bus) return 0;
    for (int i = 0; i < NDEV; i++) if (pci_slot[i] == slot && pci_func[i] == fn) return &g_pci[i];
    return 0;
}

/* framebuffer BAR moved: follow it */
static void vga_bar_update(Machine *m) {
    u32 bar0 = r32(g_pci[DEV_VGA].cfg + 0x10) & ~0xfu;
    if (bar0 && bar0 != 0xff000000u && m->fb) {
        m->fb_phys = bar0;
        for (int ci = 0; ci < m->ncpu; ci++) tlb_flush(m->cpus[ci]);
    }
}

u32 pci_cfg_read(Machine *m, u32 addr, int size) {
    (void)m;
    PciFn *f = pci_find_fn(addr);
    u32 off = (addr & 0xfc) | (addr & 3);
    if (!f) return size == 4 ? 0xffffffffu : size == 2 ? 0xffff : 0xff;
    u32 v = 0;
    for (int i = 0; i < size; i++) v |= (u32)f->cfg[(off + (u32)i) & 0xff] << (8 * i);
    return v;
}

void pci_cfg_write(Machine *m, u32 addr, u32 v, int size) {
    PciFn *f = pci_find_fn(addr);
    if (!f) return;
    u32 off = (addr & 0xfc) | (addr & 3);
    /* BARs and the expansion ROM BAR: size probing */
    if (size == 4 && ((off >= 0x10 && off <= 0x24) || off == 0x30)) {
        int bi = off == 0x30 ? 6 : (int)(off - 0x10) / 4;
        u32 sz = f->bar_size[bi];
        if (!sz) { w32(f->cfg + off, 0); return; }
        u32 flags = bi == 6 ? 0 : (f->bar_io[bi] ? 1 : (r32(f->cfg + off) & 0xf));
        u32 mask = ~(sz - 1);
        if (bi == 6) w32(f->cfg + off, (v & mask) | (v & 1));
        else w32(f->cfg + off, (v & mask) | flags);
        if (f == &g_pci[DEV_VGA]) vga_bar_update(m);
        return;
    }
    if (off < 4 || off == 8 || (off >= 0x0a && off < 0x10 && off != 0x0c && off != 0x0d)) return;   /* read-only ids */
    int pam_changed = 0;
    for (int i = 0; i < size; i++) {
        u32 o = (off + (u32)i) & 0xff;
        if (f == &g_pci[DEV_HOST] && o >= 0x59 && o <= 0x5f) pam_changed = 1;
        f->cfg[o] = (u8)(v >> (8 * i));
    }
    if (pam_changed) for (int ci = 0; ci < m->ncpu; ci++) tlb_flush(m->cpus[ci]);
    if (f == &g_pci[DEV_VGA] && off <= 0x10) vga_bar_update(m);
}

/* PAM attribute for an address in 0xC0000-0xFFFFF: bit0 read RAM, bit1 write RAM */
static u32 pam_attr(u64 pa) {
    const u8 *c = g_pci[DEV_HOST].cfg;
    if (pa >= 0xf0000) return c[0x59] >> 4;
    u32 idx = (u32)(pa - 0xc0000) >> 14;     /* 16 KiB chunks: C0000.. EFFFF -> 0..11 */
    u8 r = c[0x5a + idx / 2];
    return idx & 1 ? r >> 4 : r & 15;
}

/* Firmware-area mapping. Returns 1 if pa is handled here (*out = page or NULL). */
int chipset_map(Machine *m, u64 pa, int write, u8 **out) {
    if (!m->bios) return 0;
    if (pa >= 0x100000000ull - m->bios_size && pa < 0x100000000ull) {
        *out = write ? 0 : m->bios + (uintptr_t)((pa - (0x100000000ull - m->bios_size)) & ~0xfffull);
        return 1;
    }
    if (pa >= 0xc0000 && pa < 0x100000) {
        u32 a = pam_attr(pa);
        if (write) { *out = (a & 2) ? m->ram + (uintptr_t)(pa & ~0xfffull) : 0; return 1; }
        if (a & 1) { *out = m->ram + (uintptr_t)(pa & ~0xfffull); return 1; }
        /* reads not routed to RAM go to the ROM: the BIOS image's last 128 KiB at E0000 */
        u64 lo = 0x100000 - (m->bios_size < 0x20000 ? m->bios_size : 0x20000);
        if (pa >= lo) { *out = m->bios + (uintptr_t)((m->bios_size - (0x100000 - (pa & ~0xfffull)))); return 1; }
        *out = 0;
        return 1;
    }
    return 0;
}

/* PIIX4 power management I/O block (base in PM config 0x40, enabled by 0x80 bit 0):
 * PM1a status/enable (+0, +2), PM1a control (+4), PM timer (+8, 24-bit at 3.579545 MHz). */
static u16 pm1_sts, pm1_en, pm1_cnt = 1;      /* SCI_EN: ACPI mode is always on (FADT SMI_CMD = 0) */

int chipset_pm_io(Machine *m, u16 port, int write, u32 *v, int size) {
    const u8 *c = g_pci[DEV_PM].cfg;
    if (!(c[0x80] & 1)) return 0;
    u32 base = (r32(c + 0x40) & 0xffc0u);
    if (!base || port < base || port >= base + 64) return 0;
    u32 off = port - base;
    if (!write) {
        if (off >= 8 && off < 12) {
            u64 ticks = machine_now(m) * 3579545ull / 1000000000ull;
            u32 t = (u32)ticks & 0xffffff;
            *v = (t >> (8 * (off - 8))) & (size == 4 ? 0xffffffffu : size == 2 ? 0xffffu : 0xffu);
        } else if (off == 0) *v = pm1_sts;
        else if (off == 2) *v = pm1_en;
        else if (off == 4) *v = pm1_cnt;
        else *v = 0;
    } else {
        if (off == 0) pm1_sts &= (u16)~*v;
        else if (off == 2) pm1_en = (u16)*v;
        else if (off == 4) {
            pm1_cnt = (u16)((*v & ~0x2000u) | 1);
            if (*v & 0x2000) emu_fatal(&m->cpu, "guest entered ACPI sleep/soft-off (PM1_CNT SLP_EN)");
        }
    }
    (void)size;
    return 1;
}

const u8 *chipset_vga_rom(Machine *m, u64 pa) {
    u32 bar = r32(g_pci[DEV_VGA].cfg + 0x30);
    if (!m->vga_rom || !(bar & 1)) return 0;
    u32 base = bar & ~0x7ffu;
    if (pa >= base && pa < base + m->vga_rom_size) return m->vga_rom + (pa - base);
    return 0;
}

/* ---- Bochs VBE "DISPI" interface (index 0x1CE, data 0x1CF / 0x1D0) ---- */
static u16 dispi[16] = { 0xb0c5 };
static u16 dispi_index;

/* Recompute the visible framebuffer geometry from the DISPI registers. */
static int vbe_apply(Machine *m, int log) {
    u32 bpp = dispi[3] ? dispi[3] : 8, w = dispi[1], h = dispi[2];
    u32 bytes = (bpp + 7) / 8;
    u32 pitch = (u32)dispi[6] * bytes;
    u32 start = (u32)dispi[9] * pitch + (u32)dispi[8] * bytes;
    if (!w || !h || !m->fb || pitch < w * bytes || start + pitch * h > m->fb_size) return 0;
    m->fb_width = w; m->fb_height = h; m->fb_pitch = pitch; m->fb_bpp = bpp; m->fb_start = start;
    if (log) emu_log(m, "VBE: %ux%ux%u pitch %u offset %u,%u", w, h, bpp, pitch, (u32)dispi[8], (u32)dispi[9]);
    return 1;
}

int vbe_io(Machine *m, u16 port, int write, u32 *v, int size) {
    (void)size;
    if (port != 0x1ce && port != 0x1cf && port != 0x1d0) return 0;
    if (port == 0x1ce) { if (write) dispi_index = (u16)*v; else *v = dispi_index; return 1; }
    u32 i = dispi_index;
    if (!write) {
        if (i == 0xa) *v = m->fb_size >> 16;                  /* VRAM in 64 KiB units */
        else if ((dispi[4] & 0x02) && i >= 1 && i <= 3)      /* GETCAPS: maximum resolution/depth */
            *v = i == 1 ? 2560 : i == 2 ? 1600 : 32;
        else *v = i < 16 ? dispi[i] : 0;
        return 1;
    }
    if (i >= 16) return 1;
    u16 val = (u16)*v;
    if (i == 0) { if (val >= 0xb0c0 && val <= 0xb0c5) dispi[0] = val; return 1; }
    if (i == 4) {
        dispi[4] = val;
        if (val & 1) {
            u32 w = dispi[1];
            if (!dispi[6] || dispi[6] < w) dispi[6] = (u16)w;
            if (vbe_apply(m, 1) && !(val & 0x80)) e_memset(m->fb, 0, m->fb_pitch * m->fb_height);   /* NOCLEAR not set */
        }
        return 1;
    }
    /* resolution and depth are latched at enable; virtual width and panning offsets apply live */
    if ((dispi[4] & 1) && i >= 1 && i <= 3) return 1;
    dispi[i] = val;
    if (i == 1) dispi[6] = val;                                  /* virtual width follows X resolution */
    if ((dispi[4] & 1) && (i == 6 || i == 8 || i == 9)) vbe_apply(m, 0);
    return 1;
}

void chipset_dump(Machine *m) {
    const u8 *c = g_pci[DEV_PM].cfg;
    emu_log(m, "PIIX4 PM: cfg40 %08x cfg80 %02x cmd %04x", r32(c + 0x40), c[0x80], (u32)c[4] | (u32)c[5] << 8);
    const u8 *ide = g_pci[DEV_IDE].cfg;
    emu_log(m, "PIIX IDE: cmd %04x cfg08 %08x cfg20 %08x cfg3c %08x cfg40 %08x cfg44 %08x", (u32)ide[4] | (u32)ide[5] << 8,
            r32(ide + 8), r32(ide + 0x20), r32(ide + 0x3c), r32(ide + 0x40), r32(ide + 0x44));
}

/* I/O base of the PIIX IDE bus-master registers (BAR4), 0 if not enabled */
u32 chipset_ide_bm_base(Machine *m) {
    (void)m;
    const u8 *c = g_pci[DEV_IDE].cfg;
    if (!(c[4] & 1)) return 0;
    u32 bar = r32(c + 0x20) & ~3u;
    return bar & 0xffff;
}
