/* W32Eleven bare-metal runtime (i386 host).
 *
 * Booted by a Multiboot (v1) loader. Runs the JIT self-tests, then, if a boot
 * image was supplied as a module, emulates an x86-64 PC and boots it:
 *   - module 0 is an ISO 9660 image with a GRUB menu (/boot/grub/grub.cfg)
 *   - kernel command line: entry=N (menu entry, default: the menu default),
 *     ram=MIB (guest RAM, default 512), args="..." (appended to the guest's
 *     kernel command line, up to the end of the line)
 * The guest framebuffer is the host's Bochs/QEMU VBE linear framebuffer when
 * present (zero copy), otherwise host RAM. Guest COM1 and emulator messages
 * go to the debug console (port 0xE9). Host PS/2 keyboard and mouse input is
 * polled and forwarded to the guest's 8042.
 */
#include "../emu/emu.h"
#include "ata.h"

int jit_selftest(void);
int iso_boot(Machine *m, const u8 *iso, u64 size, int entry, const char *extra_args);

static inline void outb(u16 p, u8 v) { __asm__ volatile("outb %0,%1" ::"a"(v), "Nd"(p)); }
static inline u8 inb(u16 p) { u8 v; __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"(p)); return v; }
static inline void outw(u16 p, u16 v) { __asm__ volatile("outw %0,%1" ::"a"(v), "Nd"(p)); }
static inline u16 inw(u16 p) { u16 v; __asm__ volatile("inw %1,%0" : "=a"(v) : "Nd"(p)); return v; }
static inline void outl(u16 p, u32 v) { __asm__ volatile("outl %0,%1" ::"a"(v), "Nd"(p)); }
static inline u32 inl(u16 p) { u32 v; __asm__ volatile("inl %1,%0" : "=a"(v) : "Nd"(p)); return v; }

void print(const char *s) { while (*s) outb(0xe9, (u8)*s++); }

static void printf_(const char *fmt, ...) {
    char buf[256];
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    emu_vsnprintf(buf, sizeof buf, fmt, ap);
    __builtin_va_end(ap);
    print(buf);
}

__attribute__((noreturn)) static void stop(int ok) {
    print(ok ? "PASS\n" : "FAIL\n");
    outb(0xf4, ok ? 0x10 : 0x11);
    for (;;) __asm__ volatile("cli; hlt");
}

/* ---------- host callbacks ---------- */

static void h_serial(void *ctx, u8 c) { (void)ctx; outb(0xe9, c); }
static void h_debug(void *ctx, u8 c) { (void)ctx; outb(0xe9, c); }
static void h_log(void *ctx, const char *s) { (void)ctx; print("[w32] "); print(s); print("\n"); }

static u8 cmos(u8 r) { outb(0x70, r); return inb(0x71); }

static u64 h_wall(void *ctx) {
    (void)ctx;
    while (cmos(0x0a) & 0x80) {}
    u32 s = cmos(0), mi = cmos(2), h = cmos(4), d = cmos(7), mo = cmos(8), y = cmos(9), b = cmos(0x0b);
    if (!(b & 4)) {
#define BCD(v) (((v) & 15) + ((v) >> 4) * 10)
        s = BCD(s); mi = BCD(mi); h = BCD(h & 0x7f) | (h & 0x80); d = BCD(d); mo = BCD(mo); y = BCD(y);
    }
    if (!(b & 2) && (h & 0x80)) h = ((h & 0x7f) + 12) % 24;
    u32 year = 2000 + y;
    /* days from civil (Howard Hinnant) */
    u32 yy = year - (mo <= 2);
    u32 era = yy / 400, yoe = yy - era * 400;
    u32 doy = (153 * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    u32 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    u64 days = (u64)era * 146097 + doe - 719468;
    return days * 86400 + h * 3600 + mi * 60 + s;
}

/* ---------- PCI / VBE ---------- */

static u32 pci_read(u32 bus, u32 dev, u32 fn, u32 off) {
    outl(0xcf8, 0x80000000u | bus << 16 | dev << 11 | fn << 8 | (off & 0xfc));
    return inl(0xcfc);
}

static u8 *vbe_setup(u32 w, u32 h, u64 *phys) {
    for (u32 dev = 0; dev < 32; dev++) {
        u32 id = pci_read(0, dev, 0, 0);
        if (id != 0x11111234u) continue;
        u32 bar = pci_read(0, dev, 0, 0x10) & ~0xfu;
        outw(0x1ce, 0); u16 ver = inw(0x1cf);
        if (ver < 0xb0c0 || ver > 0xb0cf) return 0;
        outw(0x1ce, 4); outw(0x1cf, 0);
        outw(0x1ce, 1); outw(0x1cf, (u16)w);
        outw(0x1ce, 2); outw(0x1cf, (u16)h);
        outw(0x1ce, 3); outw(0x1cf, 32);
        outw(0x1ce, 4); outw(0x1cf, 0x41);
        *phys = bar;
        return (u8 *)(uintptr_t)bar;
    }
    return 0;
}

/* ---------- host PS/2 input ---------- */

static u8 mouse_pkt[3];
static int mouse_n;

static void ps2_wait_in(void) { for (int i = 0; i < 100000 && (inb(0x64) & 2); i++) {} }

static void input_init(void) {
    while (inb(0x64) & 1) inb(0x60);
    ps2_wait_in(); outb(0x64, 0xa8);              /* enable aux port */
    ps2_wait_in(); outb(0x64, 0xd4);
    ps2_wait_in(); outb(0x60, 0xf4);              /* mouse: enable streaming */
    for (int i = 0; i < 100000; i++) if (inb(0x64) & 1) { inb(0x60); break; }
}

static void input_poll(Machine *m) {
    for (int guard = 0; guard < 64; guard++) {
        u8 st = inb(0x64);
        if (!(st & 1)) return;
        u8 v = inb(0x60);
        if (st & 0x20) {
            if (mouse_n == 0 && !(v & 0x08)) continue;   /* resync */
            mouse_pkt[mouse_n++] = v;
            if (mouse_n == 3) {
                mouse_n = 0;
                int dx = mouse_pkt[1] - ((mouse_pkt[0] << 4) & 0x100);
                int dy = mouse_pkt[2] - ((mouse_pkt[0] << 3) & 0x100);
                machine_mouse(m, dx, dy, mouse_pkt[0] & 7);
            }
        } else {
            machine_key(m, v);
        }
    }
}

/* ---------- command line ---------- */

static const char *arg(const char *cmd, const char *key) {
    u32 kl = (u32)e_strlen(key);
    for (const char *p = cmd; p && *p; p++) {
        if ((p == cmd || p[-1] == ' ')) {
            u32 i = 0;
            while (i < kl && p[i] == key[i]) i++;
            if (i == kl && p[kl] == '=') return p + kl + 1;
        }
    }
    return 0;
}

static u32 num(const char *s, u32 def) {
    if (!s || *s < '0' || *s > '9') return def;
    u32 v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (u32)(*s++ - '0');
    return v;
}

extern char __image_end[];
static Machine machine;

/* ---------- firmware boots: boot CD and display ---------- */

static AtaDev boot_cd;
static u8 cd_buf[16 * 2048];
static int hdd_on;
static const u8 *unattend_img;
static u32 unattend_size;
static int hd_rw(u64 off, u32 len, u8 *rbuf, const u8 *wbuf);
/* the disc is patched (tools/w32-patch-iso.py): the guest must see the original
 * BIOS boot entry, or its BIOS would boot this emulator again */
static u32 cat_lba;             /* 0: not patched */
static u8 orig_entry[32];

static int cd_find_patch(void) {
    static u8 sec[2048];
    if (atapi_read(&boot_cd, 17, 1, sec)) return -1;
    if (sec[0] != 0 || sec[1] != 'C' || sec[2] != 'D') return 0;
    u32 cl = (u32)sec[0x47] | (u32)sec[0x48] << 8 | (u32)sec[0x49] << 16 | (u32)sec[0x4a] << 24;
    if (atapi_read(&boot_cd, cl, 1, sec)) return -1;
    u32 img = (u32)sec[40] | (u32)sec[41] << 8 | (u32)sec[42] << 16 | (u32)sec[43] << 24;
    static u8 h[2048];
    for (u32 back = 1; back < 4096 && back <= img; back++) {
        if (atapi_read(&boot_cd, img - back, 1, h)) return -1;
        static const char magic[] = "W32ELEVEN-PATCH1";
        int ok = 1;
        for (int i = 0; i < 16; i++) if (h[i] != (u8)magic[i]) ok = 0;
        u32 hl = (u32)h[20] | (u32)h[21] << 8 | (u32)h[22] << 16 | (u32)h[23] << 24;
        if (ok && hl == img) {
            cat_lba = cl;
            e_memcpy(orig_entry, h + 32, 32);
            return 1;
        }
    }
    return 0;
}

/* IDE unit 2 (secondary master, ATAPI) reads the real boot CD */
static int h_disk_read(void *ctx, int unit, u64 off, u32 len, u8 *buf) {
    (void)ctx;
    if (unit == 0 && hdd_on) return hd_rw(off, len, buf, 0);
    if (unit == 1 && unattend_img) {   /* second CD: autounattend.xml (in host RAM) */
        if (off + len > unattend_size) return -1;
        e_memcpy(buf, unattend_img + off, len);
        return 0;
    }
    if (unit != 2) return -1;
    while (len) {
        u32 lba = (u32)(off >> 11), skip = (u32)off & 2047;
        u32 n = (skip + len + 2047) >> 11;
        if (n > 16) n = 16;
        if (atapi_read(&boot_cd, lba, n, cd_buf)) return -1;
        u32 take = n * 2048 - skip;
        if (take > len) take = len;
        if (cat_lba && cat_lba >= lba && cat_lba < lba + n) e_memcpy(cd_buf + (cat_lba - lba) * 2048 + 32, orig_entry, 32);
        e_memcpy(buf, cd_buf + skip, take);
        buf += take; off += take; len -= take;
    }
    return 0;
}
/* IDE unit 0 (primary master): the real hard disk, only with hdd=ata */
static AtaDev hdd;
static u8 hd_buf[128 * 512];

/* disk boot stub (installed by install=1): the first MiB holds GRUB and this
 * emulator; the guest sees the original MBR boot code (saved in the header at
 * LBA 2047) and zeros in the gap, so its BIOS boots Windows, not us */
#define STUB_HDR_LBA 2047u
static int stub_on;
static u8 stub_hdr[512];            /* "W32ELEVEN-DISK01", +64: the guest's MBR boot code (440 bytes) */
static u8 stub_code[440];           /* the real MBR boot code (GRUB) */
static u8 tmp_sec[512];

static int mem_differs(const u8 *a, const u8 *b, u32 n) { for (u32 i = 0; i < n; i++) if (a[i] != b[i]) return 1; return 0; }

static void stub_fix_read(u64 lba, u32 n, u8 *buf) {
    for (u32 i = 0; i < n; i++) {
        u64 l = lba + i;
        if (l == 0) e_memcpy(buf, stub_hdr + 64, 440);
        else if (l <= STUB_HDR_LBA) e_memset(buf + i * 512, 0, 512);
    }
}
/* before writing: keep the real stub sectors, remember the guest's boot code */
static int stub_fix_write(u64 lba, u32 n, u8 *buf) {
    for (u32 i = 0; i < n; i++) {
        u64 l = lba + i;
        if (l == 0) {
            if (mem_differs(stub_hdr + 64, buf, 440)) {
                e_memcpy(stub_hdr + 64, buf, 440);
                if (ata_write(&hdd, STUB_HDR_LBA, 1, stub_hdr)) return -1;
            }
            e_memcpy(buf, stub_code, 440);
        } else if (l <= STUB_HDR_LBA) {
            if (ata_read(&hdd, l, 1, buf + i * 512)) return -1;
        }
    }
    return 0;
}

static int hd_rw(u64 off, u32 len, u8 *rbuf, const u8 *wbuf) {
    while (len) {
        u64 lba = off >> 9;
        u32 skip = (u32)off & 511;
        u32 n = (skip + len + 511) >> 9;
        if (n > 128) n = 128;
        u32 take = n * 512 - skip;
        if (take > len) take = len;
        int prot = stub_on && lba <= STUB_HDR_LBA;
        if (rbuf) {
            if (ata_read(&hdd, lba, n, hd_buf)) return -1;
            if (prot) stub_fix_read(lba, n, hd_buf);
            e_memcpy(rbuf, hd_buf + skip, take);
            rbuf += take;
        } else {
            if (skip || (take & 511)) {   /* partial sectors: read-modify-write */
                if (ata_read(&hdd, lba, n, hd_buf)) return -1;
            }
            if (prot && (skip || (take & 511))) stub_fix_read(lba, n, hd_buf);   /* the guest's view, then its bytes */
            e_memcpy(hd_buf + skip, wbuf, take);
            if (prot && stub_fix_write(lba, n, hd_buf)) return -1;
            if (ata_write(&hdd, lba, n, hd_buf)) return -1;
            wbuf += take;
        }
        off += take; len -= take;
    }
    return 0;
}

static int h_disk_write(void *ctx, int unit, u64 off, u32 len, const u8 *buf) {
    (void)ctx;
    if (unit != 0 || !hdd_on) return -1;
    return hd_rw(off, len, 0, buf);
}

/* the host's display: Bochs VBE (QEMU) or the framebuffer the boot loader set up */
typedef struct { u8 *p; u32 w, h, pitch, bpp; } HostFb;

/* copy the guest's VBE picture (24/32 bpp) to the host framebuffer (16/24/32 bpp) */
static void blit(Machine *m, const u8 *vram, const HostFb *hf) {
    u32 bpp = m->fb_bpp;
    if (bpp != 32 && bpp != 24) return;
    u32 w = m->fb_width < hf->w ? m->fb_width : hf->w, h = m->fb_height < hf->h ? m->fb_height : hf->h;
    for (u32 y = 0; y < h; y++) {
        const u8 *src = vram + m->fb_start + y * m->fb_pitch;
        u8 *dst = hf->p + y * hf->pitch;
        if (bpp == 32 && hf->bpp == 32) { e_memcpy(dst, src, w * 4); continue; }
        for (u32 x = 0; x < w; x++) {
            const u8 *px = src + x * (bpp / 8);
            u32 b = px[0], g = px[1], r = px[2];
            if (hf->bpp == 32) ((u32 *)(void *)dst)[x] = b | g << 8 | r << 16;
            else if (hf->bpp == 24) { dst[3 * x] = (u8)b; dst[3 * x + 1] = (u8)g; dst[3 * x + 2] = (u8)r; }
            else ((u16 *)(void *)dst)[x] = (u16)((r >> 3) << 11 | (g >> 2) << 5 | b >> 3);   /* RGB565 */
        }
    }
}

static const u32 *find_module(const u32 *mod, u32 nmods, const char *name) {
    for (u32 i = 0; i < nmods; i++) {
        const char *s = (const char *)(uintptr_t)mod[4 * i + 2];
        if (!s) continue;
        u32 k = 0;
        while (name[k] && s[k] == name[k]) k++;
        if (!name[k] && (s[k] == 0 || s[k] == ' ')) return &mod[4 * i];
    }
    return 0;
}

static int stub_load(void) {
    static const char magic[] = "W32ELEVEN-DISK01";
    if (ata_read(&hdd, STUB_HDR_LBA, 1, stub_hdr)) return -1;
    for (int i = 0; i < 16; i++) if (stub_hdr[i] != (u8)magic[i]) return 0;
    if (ata_read(&hdd, 0, 1, tmp_sec)) return -1;
    e_memcpy(stub_code, tmp_sec, 440);
    stub_on = 1;
    return 1;
}

/* install=1: put the boot stub (module 'diskstub') on the hard disk */
static void stub_install(const u32 *m) {
    const u8 *blob = (const u8 *)(uintptr_t)m[0];
    static const char smagic[] = "W32ELEVEN-STUB01";
    for (int i = 0; i < 16; i++) if (blob[i] != (u8)smagic[i]) { print("[w32] install: bad diskstub module\n"); return; }
    u32 gap = (u32)blob[16] | (u32)blob[17] << 8 | (u32)blob[18] << 16 | (u32)blob[19] << 24;
    if (gap + 1 != STUB_HDR_LBA || m[1] - m[0] < 1024 + gap * 512) { print("[w32] install: diskstub size mismatch\n"); return; }
    static u8 mbr[512];
    if (ata_read(&hdd, 0, 1, mbr)) { print("[w32] install: cannot read the MBR\n"); return; }
    if (mbr[510] != 0x55 || mbr[511] != 0xaa) { print("[w32] install: the disk has no MBR (install Windows first)\n"); return; }
    for (int p = 0; p < 4; p++) {
        const u8 *e = mbr + 446 + 16 * p;
        u32 start = (u32)e[8] | (u32)e[9] << 8 | (u32)e[10] << 16 | (u32)e[11] << 24;
        if (e[4] == 0xee) { print("[w32] install: GPT disk, not supported (the emulated PC boots in BIOS/MBR mode)\n"); return; }
        if (e[4] && start <= STUB_HDR_LBA) { printf_("[w32] install: partition %d starts at sector %u, the first MiB is not free\n", p + 1, start); return; }
    }
    int had = stub_load();
    if (had < 0) { print("[w32] install: disk read error\n"); return; }
    if (!had) {
        /* the gap must be empty: nothing else lives there */
        for (u32 l = 1; l <= STUB_HDR_LBA; l++) {
            if (ata_read(&hdd, l, 1, tmp_sec)) { print("[w32] install: disk read error\n"); return; }
            for (int i = 0; i < 512; i++) if (tmp_sec[i]) { printf_("[w32] install: sector %u is in use (another boot loader?), not touching the disk\n", l); return; }
        }
        static const char magic[] = "W32ELEVEN-DISK01";
        e_memset(stub_hdr, 0, 512);
        for (int i = 0; i < 16; i++) stub_hdr[i] = (u8)magic[i];
        e_memcpy(stub_hdr + 64, mbr, 440);              /* Windows' boot code, for the guest */
    }
    stub_on = 0;
    if (ata_write(&hdd, STUB_HDR_LBA, 1, stub_hdr) || ata_write(&hdd, 1, gap, blob + 1024)) { print("[w32] install: write error\n"); return; }
    e_memcpy(mbr, blob + 512, 440);                     /* GRUB boot.img code; the partition table stays */
    if (ata_write(&hdd, 0, 1, mbr) || ata_flush(&hdd)) { print("[w32] install: write error on the MBR\n"); return; }
    print(had ? "[w32] install: boot stub updated\n" : "[w32] install: boot stub installed; this disk now starts Windows through W32Eleven\n");
    stub_load();
}

static void firmware_main(const char *cmd, const u32 *mod, u32 nmods, u64 used, u64 top, const u32 *mbi) {
    const u32 *bm = find_module(mod, nmods, "bios"), *vm = find_module(mod, nmods, "vgabios");
    if (!bm) { print("firmware boot: no 'bios' module\n"); stop(0); }
    print("[w32] firmware boot: looking for the boot CD...\n");
    static u8 scratch[2048];
    const char *cdopt = arg(cmd, "cdrom");
    int force_bios = cdopt && cdopt[0] == 'b' && cdopt[1] == 'i' && cdopt[2] == 'o' && cdopt[3] == 's';   /* cdrom=bios */
    int have_cd = !force_bios && atapi_find_cd(&boot_cd, scratch) == 0;
    if (!have_cd) {
        /* booted from a USB stick (or any disk the BIOS knows): read the image through the BIOS;
         * the boot loader's drive first, then the BIOS hard disks */
        int first = (mbi[0] & 2) ? (int)(mbi[3] >> 24) : 0x80;
        for (int k = -1; k < 8 && !have_cd; k++) {
            int drv = k < 0 ? first : 0x80 + k;
            if (k >= 0 && drv == first) continue;
            if (bios_open(&boot_cd, drv)) continue;
            printf_("[w32] BIOS drive %x: %u-byte sectors, %u MiB\n", (u32)drv, boot_cd.secsz, (u32)((boot_cd.sectors * boot_cd.secsz) >> 20));
            if (atapi_read(&boot_cd, 16, 1, scratch) == 0 && scratch[1] == 'C' && scratch[2] == 'D' && scratch[3] == '0' &&
                scratch[4] == '0' && scratch[5] == '1') {
                have_cd = 1;
                printf_("[w32] boot medium: BIOS drive %x, read through INT 13h\n", (u32)drv);
            }
        }
        if (!have_cd) e_memset(&boot_cd, 0, sizeof boot_cd);
    }
    if (!have_cd) {
        boot_cd.ch = boot_cd.dev = -1;
        if (!(cdopt && cdopt[0] == 'o')) { print("[w32] no ATAPI CD/DVD drive with an ISO 9660 disc found\n"); stop(0); }
        print("[w32] no CD/DVD (booting from the hard disk)\n");
    } else if (boot_cd.bios) { /* reported above */ }
    else if (boot_cd.ahci) printf_("[w32] boot CD: AHCI port %d, %u MiB\n", boot_cd.ch, (u32)((u64)boot_cd.sectors * 2048 >> 20));
    else printf_("[w32] boot CD: IDE %s %s, %u MiB\n", boot_cd.ch ? "secondary" : "primary", boot_cd.dev ? "slave" : "master",
            (u32)((u64)boot_cd.sectors * 2048 >> 20));
    if (have_cd && cd_find_patch() > 0) printf_("[w32] patched disc: the guest boots the original BIOS entry (catalog LBA %u)\n", cat_lba);

    /* host memory: JIT, VRAM, guest RAM */
    u32 jit_size = num(arg(cmd, "jitmb"), 256) << 20;
    u8 *jit_mem = 0;
    if (num(arg(cmd, "jit"), 1) && top > used + jit_size + (512u << 20)) { jit_mem = (u8 *)(uintptr_t)used; used += jit_size; }
    const u32 vram_size = 16u << 20;
    u8 *vram = (u8 *)(uintptr_t)used; used += vram_size;
    u64 avail = top > used + (1u << 20) ? (top - used - (1u << 20)) & ~0xfffffull : 0;
    u64 want = arg(cmd, "ram") ? (u64)num(arg(cmd, "ram"), 0) << 20 : avail;
    if (want > avail) want = avail;
    if (want > (3584ull << 20)) want = 3584ull << 20;     /* below the guest's PCI hole */
    if (want < (256u << 20)) { printf_("not enough host RAM for a guest (%u MiB free)\n", (u32)(avail >> 20)); stop(0); }

    Host host = { h_serial, h_debug, h_log, h_wall, h_disk_read, h_disk_write, 0 };
    machine_init(&machine, (u8 *)(uintptr_t)used, want, host);
    printf_("[w32] host RAM top %u MiB, guest RAM %u MiB, JIT %u MiB\n", (u32)(top >> 20), (u32)(want >> 20), jit_mem ? jit_size >> 20 : 0);
    e_memset(vram, 0, vram_size);
    machine_set_framebuffer(&machine, vram, 0xfd000000ull, 1024, 768, 4096);
    machine.smbios_cores = num(arg(cmd, "cores"), 2);
    machine_set_cpus(&machine, (int)num(arg(cmd, "cpus"), 2));   /* emulated CPUs (time-sliced on this one) */   /* Windows 11 Setup wants >= 2 (SMBIOS); the guest still runs one CPU */
    machine_bios_reset(&machine, (u8 *)(uintptr_t)bm[0], bm[1] - bm[0], vm ? (const u8 *)(uintptr_t)vm[0] : 0, vm ? vm[1] - vm[0] : 0);
    machine_set_vram_size(&machine, vram_size);
    if (have_cd) ide_attach(&machine, 2, 1, boot_cd.bios ? (boot_cd.sectors * boot_cd.secsz) & ~2047ull : (u64)boot_cd.sectors * 2048);
    {
        const u32 *um = find_module(mod, nmods, "unattend");
        if (um && num(arg(cmd, "unattend"), 1)) {
            unattend_img = (const u8 *)(uintptr_t)um[0];
            unattend_size = (um[1] - um[0]) & ~2047u;
            ide_attach(&machine, 1, 1, unattend_size);
            printf_("[w32] setup answer file: %u KiB CD on the guest's primary slave\n", unattend_size >> 10);
        }
    }
    {
        const char *h = arg(cmd, "hdd");
        if (h && h[0] == 'a') {
            int found = ata_find_disk(&hdd, &boot_cd) == 0;
            if (!found) {
                /* no IDE/AHCI disk: a hard disk the BIOS knows (never the boot medium) */
                int bootdrv = (mbi[0] & 2) ? (int)(mbi[3] >> 24) : -1;
                for (int drv = 0x80; drv < 0x84 && !found; drv++)
                    if (drv != bootdrv && bios_open(&hdd, drv) == 0 && hdd.secsz == 512) found = 1;
            }
            if (found) {
                hdd_on = 1;
                if (num(arg(cmd, "install"), 0)) {
                    const u32 *ds = find_module(mod, nmods, "diskstub");
                    if (ds) stub_install(ds); else print("[w32] install: no diskstub module\n");
                }
                if (stub_load() > 0) print("[w32] disk boot stub present: hidden from the guest\n");
                ide_attach(&machine, 0, 0, hdd.sectors * 512);
                if (hdd.bios) printf_("[w32] hard disk: BIOS drive %x, %u MiB, through INT 13h (the guest's primary master)\n", (u32)hdd.dev, (u32)(hdd.sectors >> 11));
                else if (hdd.ahci) printf_("[w32] hard disk: AHCI port %d, %u MiB (the guest's primary master)\n", hdd.ch, (u32)(hdd.sectors >> 11));
                else printf_("[w32] hard disk: IDE %s %s, %u MiB%s (the guest's primary master)\n", hdd.ch ? "secondary" : "primary",
                        hdd.dev ? "slave" : "master", (u32)(hdd.sectors >> 11), hdd.lba48 ? ", LBA48" : "");
            } else {
                extern u32 ata_log[4];
                printf_("[w32] hdd=ata: no ATA hard disk found (probes %x %x %x %x)\n", ata_log[0], ata_log[1], ata_log[2], ata_log[3]);
            }
        }
    }
    if (jit_mem && jit_init(&machine, jit_mem, jit_size) == 0) print("[w32] JIT: x86-64 -> i386 translation enabled\n");

    HostFb hf = { 0, 1024, 768, 4096, 32 };
    {
        u64 lfb_phys = 0;
        const u8 *mb = (const u8 *)mbi;
        if ((mbi[0] & (1u << 12)) && mb[109] == 1 && (mb[108] == 32 || mb[108] == 24 || mb[108] == 16) && !mbi[23]) {
            /* the boot loader's linear framebuffer (VESA on real PCs) */
            hf.p = (u8 *)(uintptr_t)mbi[22]; hf.pitch = mbi[24]; hf.w = mbi[25]; hf.h = mbi[26]; hf.bpp = mb[108];
            printf_("[w32] display: boot loader framebuffer %ux%ux%u at %x\n", hf.w, hf.h, hf.bpp, mbi[22]);
        } else if ((hf.p = vbe_setup(1024, 768, &lfb_phys)) != 0) {
            printf_("[w32] display: host VBE 1024x768 at %x\n", (u32)lfb_phys);
        } else print("[w32] display: none found, running headless\n");
    }

    input_init();
    u64 report = 1000000000ull, next_blit = 0;
    for (;;) {
        if (machine.cpu.fatal == 3) { print("[w32] guest reset: restarting the emulated PC\n"); machine_reboot(&machine); }
        if (machine.cpu.fatal) break;
        machine_run(&machine, 100000);
        input_poll(&machine);
        if (hf.p && machine.cpu.icount >= next_blit) { blit(&machine, vram, &hf); next_blit = machine.cpu.icount + 20000000ull; }
        if (machine.cpu.icount >= report) {
            printf_("[w32] %u M instructions, guest time %u ms\n", (u32)(machine.cpu.icount / 1000000), (u32)(machine.ns / 1000000));
            report += 1000000000ull;
        }
    }
    printf_("[w32] guest stopped at rip %x after %u M instructions\n", (u32)machine.cpu.rip, (u32)(machine.cpu.icount / 1000000));
    stop(0);
}

void kernel_main(u32 magic, u32 mbi_addr) {
    print("W32Eleven: protected-mode i386 runtime\n");
    if (!jit_selftest()) stop(0);
    if (magic != 0x2badb002u) { print("not booted by a Multiboot loader\n"); stop(0); }
    /* keep copies of the boot loader's information: the BIOS disk thunk uses low memory */
    static u32 mbi_copy[32];
    static char cmd_copy[1024];
    static u32 mod_copy[16 * 4];
    static char mod_names[16][48];
    e_memcpy(mbi_copy, (const void *)(uintptr_t)mbi_addr, sizeof mbi_copy);
    const u32 *mbi = mbi_copy;
    if (mbi[0] & 4) {
        const char *c0 = (const char *)(uintptr_t)mbi[4];
        u32 i = 0;
        for (; c0[i] && i < sizeof cmd_copy - 1; i++) cmd_copy[i] = c0[i];
        cmd_copy[i] = 0;
        mbi_copy[4] = (u32)(uintptr_t)cmd_copy;
    }
    if ((mbi[0] & 8) && mbi[5]) {
        u32 nm = mbi[5] > 16 ? 16 : mbi[5];
        e_memcpy(mod_copy, (const void *)(uintptr_t)mbi[6], nm * 16);
        for (u32 i = 0; i < nm; i++) {
            const char *nmp = (const char *)(uintptr_t)mod_copy[4 * i + 2];
            u32 k = 0;
            if (nmp) for (; nmp[k] && k < 47; k++) mod_names[i][k] = nmp[k];
            mod_names[i][k] = 0;
            mod_copy[4 * i + 2] = (u32)(uintptr_t)mod_names[i];
        }
        mbi_copy[5] = nm;
        mbi_copy[6] = (u32)(uintptr_t)mod_copy;
    }
    u32 flags = mbi[0];
    const char *cmd = (flags & 4) ? (const char *)(uintptr_t)mbi[4] : "";
    if (!(flags & 8) || mbi[5] == 0) {
        print("no boot image module: self-tests only\n");
        stop(1);
    }
    const u32 *mod = (const u32 *)(uintptr_t)mbi[6];
    u32 nmods = mbi[5];
    u64 top = 0x100000ull + (u64)mbi[2] * 1024;            /* end of RAM above 1 MiB */
    u64 used = (u64)(uintptr_t)__image_end;
    for (u32 i = 0; i < nmods; i++) if (mod[4 * i + 1] > used) used = mod[4 * i + 1];
    used = (used + 0xfffff) & ~0xfffffull;
    {
        const char *mode = arg(cmd, "mode");
        if (mode && mode[0] == 'f') firmware_main(cmd, mod, nmods, used, top, mbi);
    }
    u64 want = (u64)num(arg(cmd, "ram"), 512) << 20;
    int use_jit = num(arg(cmd, "jit"), 1) != 0;
    const u32 jit_size = 64u << 20;
    u8 *jit_mem = 0;
    if (use_jit && top > used + jit_size + (128u << 20)) {   /* JIT code and tables right after the modules */
        jit_mem = (u8 *)(uintptr_t)used;
        used += jit_size;
    }
    u64 avail = top > used + (1u << 20) ? (top - used - (1u << 20)) & ~0xfffffull : 0;
    if (avail < want) want = avail;
    if (want < (64u << 20)) { printf_("not enough host RAM for a guest (%u MiB free)\n", (u32)(avail >> 20)); stop(0); }

    Host host = { h_serial, h_debug, h_log, h_wall, 0, 0, 0 };
    machine_init(&machine, (u8 *)(uintptr_t)used, want, host);
    printf_("[w32] host RAM top %u MiB, guest RAM %u MiB at %x\n", (u32)(top >> 20), (u32)(want >> 20), (u32)used);

    const u32 fw = 1280, fh = 800;
    u64 fb_phys = 0;
    u8 *fb = vbe_setup(fw, fh, &fb_phys);
    if (fb) {
        printf_("[w32] display: VBE linear framebuffer at %x\n", (u32)fb_phys);
    } else {
        /* headless: give the guest a framebuffer in host RAM at the top of the guest area */
        u32 sz = (fw * 4 * fh + 0xfff) & ~0xfffu;
        machine.ram_size -= sz + (1u << 20);
        fb = machine.ram + machine.ram_size + (1u << 20);
        fb_phys = 0xfd000000ull;
        print("[w32] display: none found, rendering headless\n");
    }
    machine_set_framebuffer(&machine, fb, fb_phys, fw, fh, fw * 4);

    if (jit_mem && jit_init(&machine, jit_mem, jit_size) == 0) print("[w32] JIT: x86-64 -> i386 translation enabled\n");
    else print("[w32] JIT off: interpreting\n");

    const u8 *iso = (const u8 *)(uintptr_t)mod[0];
    u32 iso_size = mod[1] - mod[0];
    const char *entry_s = arg(cmd, "entry");
    int entry = entry_s ? (int)num(entry_s, 0) : -1;
    const char *extra = arg(cmd, "args");
    if (iso_boot(&machine, iso, iso_size, entry, extra ? extra : "")) stop(0);

    input_init();
    u64 report = 100000000ull;
    while (!machine.cpu.fatal) {
        machine_run(&machine, 100000);
        input_poll(&machine);
        if (machine.cpu.icount >= report) {
            printf_("[w32] %u M instructions, guest time %u ms\n", (u32)(machine.cpu.icount / 1000000),
                    (u32)(machine.ns / 1000000));
            report += 100000000ull;
        }
    }
    printf_("[w32] guest stopped at rip %x after %u M instructions\n", (u32)machine.cpu.rip,
            (u32)(machine.cpu.icount / 1000000));
    stop(0);
}
