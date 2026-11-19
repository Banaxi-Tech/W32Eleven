/* QEMU-compatible fw_cfg device (ports 0x510 selector, 0x511 data) and the
 * ACPI tables W32Eleven publishes through it, in the format SeaBIOS installs:
 * "etc/acpi/rsdp" + "etc/acpi/tables" placed and patched by "etc/table-loader".
 *
 * PM block at 0x600 (where SeaBIOS places the PIIX4 PM I/O space), GPE0 at 0xAFE0.
 * Tables: RSDP(rev 0) -> RSDT -> { FADT(rev 3) -> FACS, DSDT ; MADT }.
 * The DSDT is assembled here from a tiny AML encoder: \_SB.PCI0 (PNP0A03) with
 * resources and an APIC-mode _PRT, ISA devices (keyboard, mouse, RTC, PIT,
 * speaker... ) and \_S5. Hardware matches chipset.c/devices.c. */
#include "emu.h"

/* ================= fw_cfg ================= */

#define FW_MAX_FILES 8
typedef struct { char name[56]; u8 *data; u32 size; u16 key; } FwFile;

static FwFile g_files[FW_MAX_FILES];
static int g_nfiles;
static u8 g_dir[4 + 64 * FW_MAX_FILES];
static u16 g_sel;
static u32 g_off;

static u8 g_store[64 * 1024];      /* backing store for generated files */
static u32 g_store_used;

static u8 *store_alloc(u32 n) {
    u8 *p = g_store + g_store_used;
    g_store_used += (n + 15) & ~15u;
    e_memset(p, 0, n);
    return p;
}

static void be32(u8 *p, u32 v) { p[0] = (u8)(v >> 24); p[1] = (u8)(v >> 16); p[2] = (u8)(v >> 8); p[3] = (u8)v; }
static void le16(u8 *p, u32 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); }
static void le32(u8 *p, u32 v) { le16(p, v); le16(p + 2, v >> 16); }
static void le64(u8 *p, u64 v) { le32(p, (u32)v); le32(p + 4, (u32)(v >> 32)); }

static void fw_add_file(const char *name, u8 *data, u32 size) {
    FwFile *f = &g_files[g_nfiles];
    e_memset(f->name, 0, sizeof f->name);
    for (int i = 0; name[i] && i < 55; i++) f->name[i] = name[i];
    f->data = data; f->size = size;
    f->key = (u16)(0x20 + g_nfiles);
    g_nfiles++;
}

static void fw_build_dir(void) {
    be32(g_dir, (u32)g_nfiles);
    for (int i = 0; i < g_nfiles; i++) {
        u8 *e = g_dir + 4 + 64 * i;
        be32(e, g_files[i].size);
        e[4] = (u8)(g_files[i].key >> 8); e[5] = (u8)g_files[i].key;
        e_memcpy(e + 8, g_files[i].name, 56);
    }
}

/* value of the selected item at offset; returns 0 past the end */
static int g_ncpu = 1;
static u8 fw_byte(u16 key, u32 off) {
    static const u8 sig[4] = { 'Q', 'E', 'M', 'U' };
    switch (key) {
    case 0x0000: return off < 4 ? sig[off] : 0;                    /* signature */
    case 0x0001: return off == 0 ? 1 : 0;                          /* ID: traditional interface only */
    case 0x0005: case 0x000f: return off == 0 ? (u8)g_ncpu : 0;   /* NB_CPUS, MAX_CPUS (16-bit LE) */
    case 0x0019: return off < 4 + 64u * (u32)g_nfiles ? g_dir[off] : 0;
    }
    for (int i = 0; i < g_nfiles; i++)
        if (g_files[i].key == key) return off < g_files[i].size ? g_files[i].data[off] : 0;
    return 0;
}

int fwcfg_io(Machine *m, u16 port, int write, u32 *v, int size) {
    (void)m;
    if (port == 0x510 || (port == 0x511 - 1 && size == 2)) {
        if (write) { g_sel = (u16)*v; g_off = 0; } else *v = g_sel;
        return 1;
    }
    if (port == 0x511) {
        if (write) return 1;
        u32 r = 0;
        for (int i = 0; i < size; i++) r |= (u32)fw_byte(g_sel, g_off++) << (8 * i);
        *v = r;
        return 1;
    }
    if (port >= 0x514 && port <= 0x51b) { if (!write) *v = 0; return 1; }   /* no DMA interface */
    return 0;
}

/* ================= AML encoder ================= */

typedef struct { u8 *p; u32 n, cap; } Aml;

static void a8(Aml *a, u32 v) { if (a->n < a->cap) a->p[a->n] = (u8)v; a->n++; }
static void abytes(Aml *a, const void *s, u32 n) { for (u32 i = 0; i < n; i++) a8(a, ((const u8 *)s)[i]); }
static void aname(Aml *a, const char *seg) { for (int i = 0; i < 4; i++) a8(a, (u8)(seg[i] ? seg[i] : '_')); }

/* package length covers its own bytes plus `body` bytes */
static void apkglen(Aml *a, u32 body) {
    if (body + 1 < 0x40) { a8(a, body + 1); return; }
    u32 len = body + 2;
    if (len < 0x1000) { a8(a, 0x40 | (len & 15)); a8(a, len >> 4); return; }
    len = body + 3;
    a8(a, 0x80 | (len & 15)); a8(a, (len >> 4) & 0xff); a8(a, len >> 12);
}

static void aint(Aml *a, u64 v) {
    if (v == 0) a8(a, 0x00);
    else if (v == 1) a8(a, 0x01);
    else if (v <= 0xff) { a8(a, 0x0a); a8(a, (u32)v); }
    else if (v <= 0xffff) { a8(a, 0x0b); a8(a, (u32)v); a8(a, (u32)v >> 8); }
    else if (v <= 0xffffffffull) { a8(a, 0x0c); for (int i = 0; i < 4; i++) a8(a, (u32)(v >> (8 * i))); }
    else { a8(a, 0x0e); for (int i = 0; i < 8; i++) a8(a, (u32)(v >> (8 * i))); }
}


static u32 eisaid(const char *id) {
    u32 v = ((u32)(id[0] - 0x40) << 26) | ((u32)(id[1] - 0x40) << 21) | ((u32)(id[2] - 0x40) << 16);
    for (int i = 3; i < 7; i++) {
        char c = id[i];
        u32 h = (u32)(c >= 'A' ? c - 'A' + 10 : c - '0');
        v |= h << (4 * (6 - i));
    }
    return __builtin_bswap32(v);
}

/* nested constructs: open writes opcode + placeholder; close fixes the length */
static u8 g_tmp[8192];

typedef struct { Aml *parent; Aml body; u8 op[2]; int oplen; char name[4]; int named; } Open;

static void open_(Open *o, Aml *parent, u8 op1, u8 op2, int oplen, const char *name, u8 *buf, u32 cap) {
    o->parent = parent; o->op[0] = op1; o->op[1] = op2; o->oplen = oplen;
    o->named = name != 0;
    if (name) for (int i = 0; i < 4; i++) o->name[i] = name[i];
    o->body.p = buf; o->body.n = 0; o->body.cap = cap;
}

static void close_(Open *o) {
    Aml *a = o->parent;
    for (int i = 0; i < o->oplen; i++) a8(a, o->op[i]);
    apkglen(a, o->body.n + (o->named ? 4 : 0));
    if (o->named) aname(a, o->name);
    abytes(a, o->body.p, o->body.n);
}

/* Name(seg, Integer) */
static void name_int(Aml *a, const char *n, u64 v) { a8(a, 0x08); aname(a, n); aint(a, v); }

/* Name(seg, Buffer(resource template bytes + end tag)) */
static void name_res(Aml *a, const char *n, const u8 *res, u32 len) {
    u8 tmp[512];
    e_memcpy(tmp, res, len);
    tmp[len] = 0x79; tmp[len + 1] = 0;          /* EndTag with checksum 0 */
    u32 blen = len + 2;
    a8(a, 0x08); aname(a, n);
    a8(a, 0x11);                                 /* BufferOp */
    Aml sz = { g_tmp, 0, sizeof g_tmp };
    aint(&sz, blen);
    apkglen(a, sz.n + blen);
    abytes(a, g_tmp, sz.n);
    abytes(a, tmp, blen);
}

/* small resource descriptors */
static u32 r_io(u8 *p, u16 base, u8 len) { p[0] = 0x47; p[1] = 1; le16(p + 2, base); le16(p + 4, base); p[6] = 1; p[7] = len; return 8; }
static u32 r_irq(u8 *p, int irq) { p[0] = 0x22; le16(p + 1, 1u << irq); return 3; }

/* a legacy ISA device: Device(n) { _HID EISAID, _CRS } */
static void isa_dev(Aml *parent, const char *n, const char *hid, const u8 *res, u32 rlen, u8 *buf) {
    Open d;
    open_(&d, parent, 0x5b, 0x82, 2, n, buf, 1024);
    name_int(&d.body, "_HID", eisaid(hid));
    name_res(&d.body, "_CRS", res, rlen);
    close_(&d);
}

static u32 g_pci_mem_base;

static u32 build_dsdt_body(u8 *out, u32 cap) {
    static u8 b_sb[6000], b_pci[5000], b_isa[3000], b_dev[1024], b_prt[2048];
    Aml root = { out, 0, cap };

    Open sb;
    open_(&sb, &root, 0x10, 0, 1, "_SB_", b_sb, sizeof b_sb);          /* Scope(\_SB) */
    Open pci;
    open_(&pci, &sb.body, 0x5b, 0x82, 2, "PCI0", b_pci, sizeof b_pci);  /* Device(PCI0) */
    name_int(&pci.body, "_HID", eisaid("PNP0A03"));
    name_int(&pci.body, "_ADR", 0);
    name_int(&pci.body, "_UID", 0);
    {
        u8 r[256]; u32 n = 0;
        /* bus 0-255 (word address space, bus range) */
        u8 bus[16] = { 0x88, 13, 0, 2, 0x0c, 0, 0, 0, 0, 0, 0xff, 0, 0, 0, 0x00, 0x01 };   /* min 0, max 255, len 256 */
        e_memcpy(r + n, bus, 16); n += 16;
        n += r_io(r + n, 0xcf8, 8);
        /* I/O 0x0000-0x0CF7 and 0x0D00-0xFFFF (word I/O ranges) */
        u8 io1[16] = { 0x88, 13, 0, 1, 0x0c, 3, 0, 0, 0, 0, 0xf7, 0x0c, 0, 0, 0xf8, 0x0c };
        u8 io2[16] = { 0x88, 13, 0, 1, 0x0c, 3, 0, 0, 0, 0x0d, 0xff, 0xff, 0, 0, 0x00, 0xf3 };
        e_memcpy(r + n, io1, 16); n += 16;
        e_memcpy(r + n, io2, 16); n += 16;
        /* memory: legacy VGA A0000-BFFFF, PCI hole 0xE0000000-0xFEBFFFFF (dword memory) */
        u8 m1[26] = { 0x87, 23, 0, 0, 0x0c, 3, 0, 0, 0, 0, 0, 0, 0x0a, 0, 0xff, 0xff, 0x0b, 0, 0, 0, 0, 0, 0, 0, 0x02, 0 };
        /* PCI memory window: from the top of RAM to 0xFEBFFFFF */
        u32 lo = g_pci_mem_base, len = 0xfec00000u - lo;
        u8 m2[26] = { 0x87, 23, 0, 0, 0x0c, 1, 0, 0, 0, 0, (u8)lo, (u8)(lo >> 8), (u8)(lo >> 16), (u8)(lo >> 24),
                      0xff, 0xff, 0xbf, 0xfe, 0, 0, 0, 0, (u8)len, (u8)(len >> 8), (u8)(len >> 16), (u8)(len >> 24) };
        e_memcpy(r + n, m1, 26); n += 26;
        e_memcpy(r + n, m2, 26); n += 26;
        name_res(&pci.body, "_CRS", r, n);
    }
    /* _PRT (APIC mode): slot s, pin p -> GSI 16 + ((s + p) & 3) */
    {
        Aml *a = &pci.body;
        Aml pk = { b_prt, 0, sizeof b_prt };
        int count = 0;
        for (int slot = 1; slot < 32; slot++)
            for (int pin = 0; pin < 4; pin++) {
                Aml e = { g_tmp, 0, sizeof g_tmp };
                u8 inner[32]; Aml in = { inner, 0, sizeof inner };
                aint(&in, ((u64)slot << 16) | 0xffff);
                aint(&in, (u64)pin);
                aint(&in, 0);
                aint(&in, (u64)(16 + ((slot + pin) & 3)));
                a8(&e, 0x12); apkglen(&e, in.n + 1); a8(&e, 4); abytes(&e, inner, in.n);
                abytes(&pk, g_tmp, e.n);
                count++;
            }
        a8(a, 0x08); aname(a, "_PRT");
        a8(a, 0x12);
        u8 cnt[4]; u32 cn;
        if (count < 256) { cnt[0] = (u8)count; cn = 1; } else { cn = 0; }
        apkglen(a, pk.n + cn);
        abytes(a, cnt, cn);
        abytes(a, b_prt, pk.n);
    }
    /* ISA bridge with legacy devices */
    Open isa;
    open_(&isa, &pci.body, 0x5b, 0x82, 2, "ISA_", b_isa, sizeof b_isa);
    name_int(&isa.body, "_ADR", 0x00010000);
    {
        u8 r[32]; u32 n;
        n = r_io(r, 0x60, 1); n += r_io(r + n, 0x64, 1); n += r_irq(r + n, 1);
        isa_dev(&isa.body, "KBD_", "PNP0303", r, n, b_dev);
        n = r_irq(r, 12);
        isa_dev(&isa.body, "MOU_", "PNP0F13", r, n, b_dev);
        n = r_io(r, 0x70, 2); n += r_irq(r + n, 8);
        isa_dev(&isa.body, "RTC_", "PNP0B00", r, n, b_dev);
        n = r_io(r, 0x40, 4); n += r_irq(r + n, 0);
        isa_dev(&isa.body, "TMR_", "PNP0100", r, n, b_dev);
        n = r_io(r, 0x61, 1);
        isa_dev(&isa.body, "SPKR", "PNP0800", r, n, b_dev);
        n = r_io(r, 0x3f8, 8); n += r_irq(r + n, 4);
        isa_dev(&isa.body, "COM1", "PNP0501", r, n, b_dev);
    }
    close_(&isa);
    close_(&pci);
    close_(&sb);
    /* Name(\_S5, Package(4) { 0, 0, 0, 0 }) */
    a8(&root, 0x08); a8(&root, '\\'); aname(&root, "_S5_");
    a8(&root, 0x12); apkglen(&root, 5); a8(&root, 4); a8(&root, 0); a8(&root, 0); a8(&root, 0); a8(&root, 0);
    return root.n;
}

/* ================= ACPI tables ================= */

static void sdt_header(u8 *t, const char *sig, u32 len, u8 rev) {
    e_memcpy(t, sig, 4);
    le32(t + 4, len);
    t[8] = rev;
    e_memcpy(t + 10, "W32ELV", 6);
    e_memcpy(t + 16, "W32ELEVN", 8);
    le32(t + 24, 1);
    e_memcpy(t + 28, "W32E", 4);
    le32(t + 32, 1);
}

static u8 *g_loader;
static u32 g_loader_n;

static void ld_cmd(u32 cmd) { u8 *e = g_loader + 128 * g_loader_n++; e_memset(e, 0, 128); le32(e, cmd); }
static void ld_name(u8 *dst, const char *s) { for (int i = 0; s[i] && i < 55; i++) dst[i] = (u8)s[i]; }

static void ld_alloc(const char *file, u32 align, u8 zone) {
    ld_cmd(1);
    u8 *e = g_loader + 128 * (g_loader_n - 1) + 4;
    ld_name(e, file); le32(e + 56, align); e[60] = zone;
}
static void ld_pointer(const char *dst, const char *src, u32 off, u8 size) {
    ld_cmd(2);
    u8 *e = g_loader + 128 * (g_loader_n - 1) + 4;
    ld_name(e, dst); ld_name(e + 56, src); le32(e + 112, off); e[116] = size;
}
static void ld_checksum(const char *file, u32 off, u32 start, u32 len) {
    ld_cmd(3);
    u8 *e = g_loader + 128 * (g_loader_n - 1) + 4;
    ld_name(e, file); le32(e + 56, off); le32(e + 60, start); le32(e + 64, len);
}

static void gas(u8 *p, u8 space, u8 bits, u64 addr, u8 access) { p[0] = space; p[1] = bits; p[2] = 0; p[3] = access; le64(p + 4, addr); }

/* SMBIOS 2.8 tables for SeaBIOS (it adds its own type 0): system, processor
 * (core count: Windows Setup reads it from type 4), boot info, end */
static u8 *sm_struct(u8 *p, const u8 *fmt, u32 len, const char *const *strs, int nstr) {
    e_memcpy(p, fmt, len);
    p += len;
    if (!nstr) { *p++ = 0; *p++ = 0; return p; }
    for (int i = 0; i < nstr; i++) { u32 n = (u32)e_strlen(strs[i]); e_memcpy(p, strs[i], n + 1); p += n + 1; }
    *p++ = 0;
    return p;
}
static void smbios_init(Machine *m) {
    u8 *t = store_alloc(1024), *p = t;
    u32 maxs = 0, n = 0;
    u8 s1[27] = { 1, 27, 0x00, 0x01, 1, 2, 3, 0 };
    for (int i = 0; i < 16; i++) s1[8 + i] = (u8)(0x57 + 13 * i);       /* fixed UUID */
    s1[24] = 6;                                                          /* wake-up: power switch */
    static const char *const st1[] = { "W32Eleven", "x86-64 PC (emulated)", "1.0" };
    u8 *q = sm_struct(p, s1, 27, st1, 3); if ((u32)(q - p) > maxs) maxs = (u32)(q - p); p = q; n++;
    u8 s4[42] = { 4, 42, 0x00, 0x04, 1, 3, 0xfe, 2 };
    le32(s4 + 8, 0x000006f1);                                            /* CPUID 1 EAX */
    le32(s4 + 12, 0x078bfbff);
    s4[16] = 3;
    le16(s4 + 18, 100); le16(s4 + 20, 2000); le16(s4 + 22, 2000);
    s4[24] = 0x41;                                                       /* populated, enabled */
    s4[25] = 1;
    le16(s4 + 26, 0xffff); le16(s4 + 28, 0xffff); le16(s4 + 30, 0xffff);
    s4[35] = s4[36] = s4[37] = (u8)m->smbios_cores;
    le16(s4 + 38, 0x0004 | 0x0008);                                      /* 64-bit capable, multi-core */
    le16(s4 + 40, 0x00c6);                                               /* family 2: Intel Core i7 class */
    static const char *const st4[] = { "CPU 0", "GenuineIntel", "W32Eleven x86-64 interpreter" };
    q = sm_struct(p, s4, 42, st4, 3); if ((u32)(q - p) > maxs) maxs = (u32)(q - p); p = q; n++;
    static const u8 s32[11] = { 32, 11, 0x00, 0x20 };
    q = sm_struct(p, s32, 11, 0, 0); p = q; n++;
    static const u8 s127[4] = { 127, 4, 0x00, 0x7f };
    q = sm_struct(p, s127, 4, 0, 0); p = q; n++;
    u32 len = (u32)(p - t);
    u8 *ep = store_alloc(31);
    e_memcpy(ep, "_SM_", 4);
    ep[5] = 31; ep[6] = 2; ep[7] = 8;
    le16(ep + 8, maxs);
    e_memcpy(ep + 16, "_DMI_", 5);
    le16(ep + 22, len); le32(ep + 24, 0); le16(ep + 28, n); ep[30] = 0x28;
    u8 sum = 0;
    for (int i = 16; i < 31; i++) sum = (u8)(sum + ep[i]);
    ep[21] = (u8)-sum;
    sum = 0;
    for (int i = 0; i < 31; i++) sum = (u8)(sum + ep[i]);
    ep[4] = (u8)-sum;
    fw_add_file("etc/smbios/smbios-tables", t, len);
    fw_add_file("etc/smbios/smbios-anchor", ep, 31);
}

void acpi_init(Machine *m) {
    g_ncpu = m->ncpu > 0 ? m->ncpu : 1;
    g_nfiles = 0;
    g_store_used = 0;
    const char *TB = "etc/acpi/tables", *RS = "etc/acpi/rsdp";
    {   /* the PCI hole starts above RAM (256 MiB aligned), as SeaBIOS assumes */
        u64 top = (m->ram_size + 0xfffffffull) & ~0xfffffffull;
        g_pci_mem_base = top < 0xe0000000ull ? (u32)top : 0xe0000000u;
    }
    u8 *tables = store_alloc(16384);
    u32 at = 0;

    /* FACS (64-byte aligned) */
    u32 facs = at;
    e_memcpy(tables + facs, "FACS", 4); le32(tables + facs + 4, 64);
    at += 64;

    /* DSDT */
    u32 dsdt = at;
    u32 aml = build_dsdt_body(tables + dsdt + 36, 8192);
    sdt_header(tables + dsdt, "DSDT", 36 + aml, 1);
    at += (36 + aml + 15) & ~15u;

    /* FADT, revision 3 */
    u32 fadt = at, flen = 244;
    u8 *f = tables + fadt;
    sdt_header(f, "FACP", flen, 3);
    /* FIRMWARE_CTRL @36 and DSDT @40 are patched by the loader (offsets inside "tables") */
    le32(f + 36, facs); le32(f + 40, dsdt);
    f[45] = 0;                           /* preferred PM profile: unspecified */
    le16(f + 46, 9);                     /* SCI_INT */
    le32(f + 48, 0);                     /* SMI_CMD: 0 = ACPI is always enabled (SCI_EN set) */
    le32(f + 56, 0x600);                /* PM1a_EVT_BLK */
    le32(f + 64, 0x604);                /* PM1a_CNT_BLK */
    le32(f + 76, 0x608);                /* PM_TMR_BLK */
    le32(f + 80, 0xafe0);                /* GPE0_BLK */
    f[88] = 4; f[89] = 2; f[91] = 4; f[92] = 4;   /* PM1_EVT_LEN, PM1_CNT_LEN, PM_TMR_LEN, GPE0_BLK_LEN */
    le16(f + 96, 0xfff); le16(f + 98, 0xfff);     /* P_LVL2/3 latency: C2/C3 unsupported */
    f[108] = 0x32;                       /* CENTURY CMOS index */
    le16(f + 109, 0x0003);               /* IAPC_BOOT_ARCH: legacy devices, 8042 */
    le32(f + 112, (1u << 0) | (1u << 2) | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 10));
    /* WBINVD, PROC_C1, PWR_BUTTON (none), SLP_BUTTON (none), FIX_RTC, RESET_REG_SUP */
    gas(f + 116, 1, 8, 0xcf9, 1);        /* RESET_REG: I/O 0xCF9 */
    f[128] = 0x06;                       /* RESET_VALUE */
    le64(f + 132, 0);                    /* X_FIRMWARE_CTRL (patched as 64-bit) */
    le64(f + 140, 0);                    /* X_DSDT */
    le64(f + 132, facs); le64(f + 140, dsdt);
    gas(f + 148, 1, 32, 0x600, 2);      /* X_PM1a_EVT_BLK */
    gas(f + 172, 1, 16, 0x604, 2);      /* X_PM1a_CNT_BLK */
    gas(f + 208, 1, 32, 0x608, 3);      /* X_PM_TMR_BLK */
    gas(f + 220, 1, 32, 0xafe0, 1);      /* X_GPE0_BLK */
    at += (flen + 15) & ~15u;

    /* MADT */
    u32 madt = at;
    u8 *a = tables + madt;
    u32 n = 44;
    le32(a + 36, 0xfee00000u); le32(a + 40, 1);           /* local APIC address, PCAT_COMPAT */
    for (int i = 0; i < (m->ncpu > 0 ? m->ncpu : 1); i++) { a[n] = 0; a[n + 1] = 8; a[n + 2] = (u8)i; a[n + 3] = (u8)i; le32(a + n + 4, 1); n += 8; }   /* CPUs */
    a[n] = 1; a[n + 1] = 12; a[n + 2] = 0; le32(a + n + 4, 0xfec00000u); le32(a + n + 8, 0); n += 12;  /* IOAPIC */
    a[n] = 2; a[n + 1] = 10; a[n + 2] = 0; a[n + 3] = 0; le32(a + n + 4, 2); le16(a + n + 8, 0); n += 10;   /* IRQ0 -> GSI2 */
    a[n] = 2; a[n + 1] = 10; a[n + 2] = 0; a[n + 3] = 9; le32(a + n + 4, 9); le16(a + n + 8, 0x000d); n += 10;  /* SCI: level, high */
    a[n] = 4; a[n + 1] = 6; a[n + 2] = 0xff; le16(a + n + 3, 0); a[n + 5] = 1; n += 6;      /* LINT1 = NMI */
    sdt_header(a, "APIC", n, 1);
    at += (n + 15) & ~15u;

    /* RSDT */
    u32 rsdt = at;
    sdt_header(tables + rsdt, "RSDT", 36 + 8, 1);
    le32(tables + rsdt + 36, fadt);
    le32(tables + rsdt + 40, madt);
    at += 48;

    /* RSDP (revision 0) */
    u8 *rsdp = store_alloc(20);
    e_memcpy(rsdp, "RSD PTR ", 8);
    e_memcpy(rsdp + 9, "W32ELV", 6);
    rsdp[15] = 0;
    le32(rsdp + 16, rsdt);

    /* loader script */
    g_loader = store_alloc(128 * 24);
    g_loader_n = 0;
    ld_alloc(RS, 16, 2);                 /* F-segment */
    ld_alloc(TB, 64, 1);                 /* high memory */
    ld_pointer(TB, TB, fadt + 36, 4);
    ld_pointer(TB, TB, fadt + 40, 4);
    ld_pointer(TB, TB, fadt + 132, 8);
    ld_pointer(TB, TB, fadt + 140, 8);
    ld_pointer(TB, TB, rsdt + 36, 4);
    ld_pointer(TB, TB, rsdt + 40, 4);
    ld_pointer(RS, TB, 16, 4);
    ld_checksum(TB, dsdt + 9, dsdt, 36 + aml);
    ld_checksum(TB, fadt + 9, fadt, flen);
    ld_checksum(TB, madt + 9, madt, n);
    ld_checksum(TB, rsdt + 9, rsdt, 44);
    ld_checksum(RS, 8, 0, 20);

    fw_add_file(RS, rsdp, 20);
    fw_add_file(TB, tables, at);
    fw_add_file("etc/table-loader", g_loader, 128 * g_loader_n);
    if (m->smbios_cores) smbios_init(m);
    fw_build_dir();
    emu_log(m, "acpi: %u bytes of tables (DSDT AML %u bytes)", at, aml);
}
