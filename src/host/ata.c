/* Polled PIO driver for the legacy IDE ports (primary 0x1F0, secondary 0x170):
 * enough to read the boot CD/DVD (ATAPI) on QEMU and on chipsets in IDE or
 * compatible mode. No interrupts (nIEN set), no DMA. i386 instructions only. */
#include "ata.h"

static inline void outb(u16 p, u8 v) { __asm__ volatile("outb %0,%1" ::"a"(v), "Nd"(p)); }
static inline u8 inb(u16 p) { u8 v; __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"(p)); return v; }
static inline void outw(u16 p, u16 v) { __asm__ volatile("outw %0,%1" ::"a"(v), "Nd"(p)); }
static inline u16 inw(u16 p) { u16 v; __asm__ volatile("inw %1,%0" : "=a"(v) : "Nd"(p)); return v; }
static inline void insw(u16 p, void *buf, u32 n) { __asm__ volatile("cld; rep insw" : "+D"(buf), "+c"(n) : "d"(p) : "memory"); }

static const u16 io_base[2] = { 0x1f0, 0x170 }, ctl_base[2] = { 0x3f6, 0x376 };

enum { R_DATA = 0, R_ERR = 1, R_FEAT = 1, R_COUNT = 2, R_LBA0 = 3, R_LBA1 = 4, R_LBA2 = 5, R_DEV = 6, R_STAT = 7, R_CMD = 7 };
enum { ST_ERR = 1, ST_DRQ = 8, ST_DF = 0x20, ST_BSY = 0x80 };

static void delay400(int ch) { for (int i = 0; i < 4; i++) inb(ctl_base[ch]); }

/* wait until not busy; returns the status, or 0xff on timeout */
static u8 wait_idle(int ch, u32 spins) {
    for (u32 i = 0; i < spins; i++) {
        u8 st = inb(io_base[ch] + R_STAT);
        if (st == 0xff) return 0xff;          /* floating bus: no device */
        if (!(st & ST_BSY)) return st;
    }
    return 0xff;
}

static int select_dev(int ch, int dev) {
    outb(io_base[ch] + R_DEV, (u8)(0xa0 | dev << 4));
    delay400(ch);
    return wait_idle(ch, 1000000) != 0xff;
}

/* read out any data a previous command left pending (DRQ) */
static void drain(int ch) {
    for (u32 i = 0; i < 65536 && (inb(io_base[ch] + R_STAT) & (ST_DRQ | ST_BSY)) == ST_DRQ; i++) inw(io_base[ch] + R_DATA);
}

/* ATAPI device present? (signature after a device reset / failed IDENTIFY) */
static int is_atapi(int ch, int dev) {
    outb(ctl_base[ch], 2);                    /* nIEN: polled operation */
    if (!select_dev(ch, dev)) return 0;
    drain(ch);
    outb(io_base[ch] + R_CMD, 0xec);          /* IDENTIFY DEVICE: ATAPI devices abort it */
    delay400(ch);
    if (inb(io_base[ch] + R_STAT) == 0) return 0;
    u8 st = wait_idle(ch, 1000000);
    if (st == 0xff) return 0;
    if (!(st & ST_ERR)) { drain(ch); return 0; }   /* an ATA device: it answered (leave nothing pending) */
    return inb(io_base[ch] + R_LBA1) == 0x14 && inb(io_base[ch] + R_LBA2) == 0xeb;
}

/* PACKET command with a data-in phase of up to len bytes; returns bytes read or -1 */
static int packet_in(AtaDev *d, const u8 pkt[12], u8 *buf, u32 len) {
    int ch = d->ch;
    u16 io = io_base[ch];
    if (!select_dev(ch, d->dev)) return -1;
    outb(io + R_FEAT, 0);                     /* PIO */
    u32 lim = len > 0xf800 ? 0xf800 : len;
    outb(io + R_LBA1, (u8)lim);
    outb(io + R_LBA2, (u8)(lim >> 8));
    outb(io + R_CMD, 0xa0);
    delay400(ch);
    u8 st = wait_idle(ch, 5000000);
    if (st == 0xff || (st & ST_ERR) || !(st & ST_DRQ)) return -1;
    for (int i = 0; i < 6; i++) outw(io + R_DATA, (u16)(pkt[2 * i] | pkt[2 * i + 1] << 8));
    u32 got = 0;
    for (;;) {
        delay400(ch);
        st = wait_idle(ch, 50000000);
        if (st == 0xff || (st & (ST_ERR | ST_DF))) return -1;
        if (!(st & ST_DRQ)) break;            /* command complete */
        u32 n = (u32)inb(io + R_LBA1) | (u32)inb(io + R_LBA2) << 8;
        if (!(n & 1) && got + n <= len) {
            insw(io + R_DATA, buf + got, n / 2);            /* the common case: straight into the buffer */
        } else {
            for (u32 i = 0; i < n; i += 2) {
                u16 w = inw(io + R_DATA);
                if (got + i < len) buf[got + i] = (u8)w;
                if (got + i + 1 < len) buf[got + i + 1] = (u8)(w >> 8);
            }
        }
        got += n;
    }
    return (int)(got > len ? len : got);
}

static int capacity(AtaDev *d) {
    u8 pkt[12] = { 0x25 };                    /* READ CAPACITY */
    u8 r[8];
    for (int tries = 0; tries < 4; tries++) {  /* the first command after a media change may report UNIT ATTENTION */
        if (packet_in(d, pkt, r, 8) == 8) {
            u32 last = (u32)r[0] << 24 | (u32)r[1] << 16 | (u32)r[2] << 8 | r[3];
            d->sectors = last + 1;
            return 0;
        }
    }
    return -1;
}

int atapi_read(AtaDev *d, u32 lba, u32 count, u8 *buf) {
    if (d->bios) return d->secsz == 2048 ? bios_disk_rw(d->dev, 2048, lba, count, buf, 0) : bios_disk_rw(d->dev, 512, (u64)lba * 4, count * 4, buf, 0);
    if (d->ahci) return ahci_atapi_read(d, lba, count, buf);
    while (count) {
        u32 n = count > 16 ? 16 : count;       /* 32 KiB per command */
        u8 pkt[12] = { 0x28, 0, (u8)(lba >> 24), (u8)(lba >> 16), (u8)(lba >> 8), (u8)lba, 0, (u8)(n >> 8), (u8)n };
        int ok = -1;
        for (int tries = 0; tries < 3 && ok < 0; tries++) ok = packet_in(d, pkt, buf, n * 2048) == (int)(n * 2048) ? 0 : -1;
        if (ok < 0) return -1;
        lba += n; count -= n; buf += n * 2048;
    }
    return 0;
}

/* first ATAPI drive holding an ISO 9660 disc */
int atapi_find_cd(AtaDev *d, u8 *scratch2048) {
    for (int ch = 0; ch < 2; ch++)
        for (int dev = 0; dev < 2; dev++) {
            if (!is_atapi(ch, dev)) continue;
            d->ch = ch; d->dev = dev;
            if (capacity(d) < 0) continue;
            d->ahci = 0;
            if (atapi_read(d, 16, 1, scratch2048) < 0) continue;
            if (scratch2048[0] == 1 && scratch2048[1] == 'C' && scratch2048[2] == 'D' && scratch2048[3] == '0' &&
                scratch2048[4] == '0' && scratch2048[5] == '1') return 0;
        }
    /* SATA controllers in AHCI mode */
    int from = 0;
    while (ahci_find(d, 1, 0, &from) == 0) {
        if (atapi_read(d, 16, 1, scratch2048) == 0 && scratch2048[0] == 1 && scratch2048[1] == 'C' && scratch2048[2] == 'D' &&
            scratch2048[3] == '0' && scratch2048[4] == '0' && scratch2048[5] == '1') return 0;
    }
    return -1;
}

/* ---------- ATA hard disks (LBA28/LBA48 PIO) ---------- */

static inline void outsw(u16 p, const void *buf, u32 n) { __asm__ volatile("cld; rep outsw" : "+S"(buf), "+c"(n) : "d"(p) : "memory"); }

int ata_why; u8 ata_dbg[8]; u32 ata_log[4];
/* IDENTIFY an ATA (non-packet) device; fills the sector count */
static int ata_identify(AtaDev *d) {
    int ch = d->ch;
    u16 io = io_base[ch];
    outb(ctl_base[ch], 2);
    if (!select_dev(ch, d->dev)) { ata_why = 4; return -1; }
    drain(ch);
    outb(io + R_COUNT, 0); outb(io + R_LBA0, 0); outb(io + R_LBA1, 0); outb(io + R_LBA2, 0);
    outb(io + R_CMD, 0xec);
    delay400(ch);
    u8 st0 = inb(io + R_STAT);
    if (st0 == 0) { ata_why = 1; return -1; }
    u8 st = wait_idle(ch, 5000000);
    ata_dbg[0] = st0; ata_dbg[1] = st; ata_dbg[2] = inb(io + R_LBA1); ata_dbg[3] = inb(io + R_LBA2);
    if (st == 0xff || (st & ST_ERR)) { ata_why = 2; return -1; }   /* aborted: ATAPI (or no device) */
    for (u32 i = 0; i < 1000000 && !(st & (ST_DRQ | ST_ERR)); i++) st = inb(io + R_STAT);
    ata_dbg[4] = st;
    if (!(st & ST_DRQ)) { ata_why = 3; return -1; }
    static u16 id[256];
    insw(io + R_DATA, id, 256);
    if (!(id[49] & 0x200)) { ata_why = 5; return -1; }                  /* no LBA */
    d->lba48 = (id[83] & 0x400) != 0;
    d->sectors = d->lba48 ? ((u64)id[103] << 48 | (u64)id[102] << 32 | (u64)id[101] << 16 | id[100])
                          : ((u32)id[61] << 16 | id[60]);
    return d->sectors ? 0 : -1;
}

int ata_find_disk(AtaDev *d, const AtaDev *skip) {
    for (int ch = 0; ch < 2; ch++)
        for (int dev = 0; dev < 2; dev++) {
            if (skip && !skip->ahci && skip->ch == ch && skip->dev == dev) continue;
            d->ch = ch; d->dev = dev; d->ahci = 0;
            if (ata_identify(d) == 0) return 0;
            ata_log[ch * 2 + dev] = (u32)ata_why << 24 | (u32)ata_dbg[0] << 16 | (u32)ata_dbg[1] << 8 | ata_dbg[4];
            ata_why = 0; ata_dbg[0] = ata_dbg[1] = ata_dbg[4] = 0;
        }
    return ahci_find(d, 0, skip, 0);
}

/* command setup for n sectors (n <= 256, or 65536 with LBA48) at lba */
static int ata_cmd(AtaDev *d, u64 lba, u32 n, int write) {
    int ch = d->ch;
    u16 io = io_base[ch];
    int ext = d->lba48 && (lba + n > 0x0fffffffull || n > 256);
    if (!ext && (lba + n > 0x0fffffffull || n > 256)) return -1;
    outb(io + R_DEV, (u8)(0xe0 | d->dev << 4 | (ext ? 0 : (u32)(lba >> 24) & 15)));
    delay400(ch);
    if (wait_idle(ch, 5000000) == 0xff) return -1;
    if (ext) {
        outb(io + R_COUNT, (u8)(n >> 8));
        outb(io + R_LBA0, (u8)(lba >> 24)); outb(io + R_LBA1, (u8)(lba >> 32)); outb(io + R_LBA2, (u8)(lba >> 40));
    }
    outb(io + R_COUNT, (u8)n);
    outb(io + R_LBA0, (u8)lba); outb(io + R_LBA1, (u8)(lba >> 8)); outb(io + R_LBA2, (u8)(lba >> 16));
    outb(io + R_CMD, write ? (ext ? 0x34 : 0x30) : (ext ? 0x24 : 0x20));
    return 0;
}

static int wait_drq(int ch) {
    delay400(ch);
    u8 st = wait_idle(ch, 50000000);
    if (st == 0xff || (st & (ST_ERR | ST_DF)) || !(st & ST_DRQ)) return -1;
    return 0;
}

int ata_read(AtaDev *d, u64 lba, u32 count, u8 *buf) {
    if (d->bios) return bios_disk_rw(d->dev, 512, lba, count, buf, 0);
    if (d->ahci) return ahci_rw(d, lba, count, buf, 0);
    while (count) {
        u32 n = count > 128 ? 128 : count;
        if (ata_cmd(d, lba, n, 0)) return -1;
        for (u32 i = 0; i < n; i++) {
            if (wait_drq(d->ch)) return -1;
            insw(io_base[d->ch] + R_DATA, buf, 256);
            buf += 512;
        }
        lba += n; count -= n;
    }
    return 0;
}

int ata_write(AtaDev *d, u64 lba, u32 count, const u8 *buf) {
    if (d->bios) return bios_disk_rw(d->dev, 512, lba, count, (u8 *)(uintptr_t)buf, 1);
    if (d->ahci) return ahci_rw(d, lba, count, (u8 *)(uintptr_t)buf, 1);
    while (count) {
        u32 n = count > 128 ? 128 : count;
        if (ata_cmd(d, lba, n, 1)) return -1;
        for (u32 i = 0; i < n; i++) {
            if (wait_drq(d->ch)) return -1;
            outsw(io_base[d->ch] + R_DATA, buf, 256);
            buf += 512;
        }
        delay400(d->ch);
        u8 st = wait_idle(d->ch, 50000000);
        if (st == 0xff || (st & (ST_ERR | ST_DF))) return -1;
        lba += n; count -= n;
    }
    return 0;
}

int ata_flush(AtaDev *d) {
    if (d->bios) return 0;
    if (d->ahci) return ahci_flush(d);
    if (!select_dev(d->ch, d->dev)) return -1;
    outb(io_base[d->ch] + R_CMD, d->lba48 ? 0xea : 0xe7);
    delay400(d->ch);
    u8 st = wait_idle(d->ch, 100000000);
    return st == 0xff || (st & ST_ERR) ? -1 : 0;
}
