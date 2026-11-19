/* Polled AHCI (SATA) driver: ATA disks (READ/WRITE DMA EXT, FLUSH CACHE EXT) and
 * ATAPI drives (PACKET + READ(10)). One command slot per port, physical =
 * linear addresses (no paging on the host), 32-bit DMA only. i386 instructions only. */
#include "ata.h"

static inline void outl(u16 p, u32 v) { __asm__ volatile("outl %0,%1" ::"a"(v), "Nd"(p)); }
static inline u32 inl(u16 p) { u32 v; __asm__ volatile("inl %1,%0" : "=a"(v) : "Nd"(p)); return v; }

static u32 pci_read(u32 bus, u32 dev, u32 fn, u32 off) {
    outl(0xcf8, 0x80000000u | bus << 16 | dev << 11 | fn << 8 | (off & 0xfc));
    return inl(0xcfc);
}
static void pci_write(u32 bus, u32 dev, u32 fn, u32 off, u32 v) {
    outl(0xcf8, 0x80000000u | bus << 16 | dev << 11 | fn << 8 | (off & 0xfc));
    outl(0xcfc, v);
}

/* port registers (offsets from the port base, 0x100 + 0x80 * port) */
enum { PxCLB = 0x00, PxCLBU = 0x04, PxFB = 0x08, PxFBU = 0x0c, PxIS = 0x10, PxIE = 0x14, PxCMD = 0x18,
       PxTFD = 0x20, PxSIG = 0x24, PxSSTS = 0x28, PxSERR = 0x30, PxCI = 0x38 };

#define MAX_PORTS 8
/* per port: command list (1 KiB aligned), received FIS (256 B), command table (128 B) */
static u8 port_mem[MAX_PORTS][4096] __attribute__((aligned(4096)));

static volatile u32 *hba;               /* ABAR */
static int nports_found;
static struct { int port; int atapi; } found[MAX_PORTS];

static volatile u32 *preg(int port, u32 off) { return (volatile u32 *)(void *)((volatile u8 *)hba + 0x100 + 0x80 * (u32)port + off); }

static int spin_clear(volatile u32 *r, u32 mask, u32 spins) {
    for (u32 i = 0; i < spins; i++) if (!(*r & mask)) return 0;
    return -1;
}

static int port_start(int port) {
    volatile u32 *cmd = preg(port, PxCMD);
    *cmd &= ~(1u << 0);                                   /* ST off */
    if (spin_clear(cmd, 1u << 15, 5000000)) return -1;    /* CR */
    *cmd &= ~(1u << 4);                                   /* FRE off */
    if (spin_clear(cmd, 1u << 14, 5000000)) return -1;    /* FR */
    u8 *m = port_mem[port];
    e_memset(m, 0, 4096);
    *preg(port, PxCLB) = (u32)(uintptr_t)m;  *preg(port, PxCLBU) = 0;            /* command list at +0 */
    *preg(port, PxFB) = (u32)(uintptr_t)(m + 1024); *preg(port, PxFBU) = 0;     /* received FIS at +1024 */
    *preg(port, PxSERR) = 0xffffffffu;
    *preg(port, PxIS) = 0xffffffffu;
    *preg(port, PxIE) = 0;                                /* polled */
    *cmd |= 1u << 4;                                      /* FRE */
    if (spin_clear(preg(port, PxTFD), 0x88, 50000000)) return -1;   /* BSY/DRQ clear */
    *cmd |= 1u << 0;                                      /* ST */
    return 0;
}

/* AHCI controller: ABAR from BAR5, AHCI mode on, ports with a device recorded */
int ahci_init(void) {
    if (hba) return nports_found;
    for (u32 bus = 0; bus < 8; bus++)
        for (u32 dev = 0; dev < 32; dev++)
            for (u32 fn = 0; fn < 8; fn++) {
                u32 id = pci_read(bus, dev, fn, 0);
                if (id == 0xffffffffu) { if (!fn) break; continue; }
                u32 cls = pci_read(bus, dev, fn, 8) >> 8;
                if (cls != 0x010601) continue;            /* mass storage, SATA, AHCI 1.0 */
                u32 bar = pci_read(bus, dev, fn, 0x24) & ~0xfu;
                if (!bar) continue;
                pci_write(bus, dev, fn, 4, pci_read(bus, dev, fn, 4) | 6);   /* memory space + bus master */
                hba = (volatile u32 *)(uintptr_t)bar;
                hba[1] |= 1u << 31;                       /* GHC.AE */
                u32 pi = hba[3];
                for (int p = 0; p < 32 && nports_found < MAX_PORTS; p++) {
                    if (!(pi >> p & 1) || p >= MAX_PORTS) continue;
                    if ((*preg(p, PxSSTS) & 0xf) != 3) continue;   /* no device / no link */
                    if (port_start(p)) continue;
                    u32 sig = *preg(p, PxSIG);
                    found[nports_found].port = p;
                    found[nports_found].atapi = sig == 0xeb140101u;
                    nports_found++;
                }
                return nports_found;
            }
    return 0;
}

/* one command in slot 0: cfis command/LBA/count, optional ATAPI packet, one buffer */
static int ahci_cmd(int port, u8 command, u64 lba, u32 count, const u8 *pkt, void *buf, u32 bytes, int write) {
    u8 *m = port_mem[port];
    u32 *hdr = (u32 *)(void *)m;                           /* command header 0 */
    u8 *tbl = m + 2048;                                    /* command table (128-byte aligned) */
    e_memset(tbl, 0, 256);
    u8 *fis = tbl;
    fis[0] = 0x27; fis[1] = 0x80;                          /* H2D register FIS, command */
    fis[2] = command;
    fis[3] = pkt ? 1 : 0;                                  /* features: DMA for PACKET */
    fis[4] = (u8)lba; fis[5] = (u8)(lba >> 8); fis[6] = (u8)(lba >> 16);
    fis[7] = 0x40;                                         /* LBA mode */
    fis[8] = (u8)(lba >> 24); fis[9] = (u8)(lba >> 32); fis[10] = (u8)(lba >> 40);
    fis[12] = (u8)count; fis[13] = (u8)(count >> 8);
    if (pkt) {
        for (int i = 0; i < 12; i++) tbl[0x40 + i] = pkt[i];
        fis[5] = (u8)bytes; fis[6] = (u8)(bytes >> 8);     /* byte count limit */
    }
    u32 *prd = (u32 *)(void *)(tbl + 0x80);
    prd[0] = (u32)(uintptr_t)buf; prd[1] = 0; prd[2] = 0; prd[3] = (bytes - 1) & 0x3fffff;
    hdr[0] = 5u | (pkt ? 1u << 5 : 0) | (write ? 1u << 6 : 0) | (bytes ? 1u << 16 : 0);   /* CFL 5, A, W, PRDTL */
    hdr[1] = 0;
    hdr[2] = (u32)(uintptr_t)tbl; hdr[3] = 0;
    *preg(port, PxIS) = 0xffffffffu;
    if (spin_clear(preg(port, PxTFD), 0x88, 50000000)) return -1;
    *preg(port, PxCI) = 1;
    for (u32 i = 0; i < 200000000u; i++) {
        if (*preg(port, PxIS) & (1u << 30)) return -1;    /* task file error */
        if (!(*preg(port, PxCI) & 1)) return (*preg(port, PxTFD) & 1) ? -1 : 0;
    }
    return -1;
}

/* next device of the kind (ATA disk or ATAPI) at or after index *from; *from moves past it */
int ahci_find(AtaDev *d, int atapi, const AtaDev *skip, int *from) {
    if (!ahci_init()) return -1;
    for (int i = from ? *from : 0; i < nports_found; i++) {
        if (from) *from = i + 1;
        if (found[i].atapi != atapi) continue;
        if (skip && skip->ahci && skip->ch == found[i].port) continue;
        d->ahci = 1; d->ch = found[i].port; d->dev = 0;
        if (atapi) {
            static u8 cap[8];
            u8 pkt[12] = { 0x25 };
            int ok = -1;
            for (int t = 0; t < 4 && ok; t++) ok = ahci_cmd(d->ch, 0xa0, 0, 0, pkt, cap, 8, 0);
            if (ok) continue;
            d->sectors = ((u32)cap[0] << 24 | (u32)cap[1] << 16 | (u32)cap[2] << 8 | cap[3]) + 1u;
        } else {
            static u16 id[256];
            if (ahci_cmd(d->ch, 0xec, 0, 0, 0, id, 512, 0)) continue;
            d->lba48 = 1;
            d->sectors = (u64)id[103] << 48 | (u64)id[102] << 32 | (u64)id[101] << 16 | id[100];
            if (!d->sectors) d->sectors = (u32)id[61] << 16 | id[60];
        }
        return 0;
    }
    return -1;
}

int ahci_atapi_read(AtaDev *d, u32 lba, u32 count, u8 *buf) {
    while (count) {
        u32 n = count > 16 ? 16 : count;
        u8 pkt[12] = { 0x28, 0, (u8)(lba >> 24), (u8)(lba >> 16), (u8)(lba >> 8), (u8)lba, 0, (u8)(n >> 8), (u8)n };
        int ok = -1;
        for (int t = 0; t < 3 && ok; t++) ok = ahci_cmd(d->ch, 0xa0, 0, 0, pkt, buf, n * 2048, 0);
        if (ok) return -1;
        lba += n; count -= n; buf += n * 2048;
    }
    return 0;
}

int ahci_rw(AtaDev *d, u64 lba, u32 count, u8 *buf, int write) {
    while (count) {
        u32 n = count > 128 ? 128 : count;
        if (ahci_cmd(d->ch, write ? 0x35 : 0x25, lba, n, 0, buf, n * 512, write)) return -1;   /* WRITE/READ DMA EXT */
        lba += n; count -= n; buf += n * 512;
    }
    return 0;
}

int ahci_flush(AtaDev *d) { return ahci_cmd(d->ch, 0xea, 0, 0, 0, 0, 0, 0); }
