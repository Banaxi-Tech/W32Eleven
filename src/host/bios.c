/* Disk access through the PC's own BIOS (INT 13h extensions), called from the
 * 32-bit runtime through a real-mode thunk (src/host/thunk.asm). This reaches
 * every disk the BIOS can boot from — USB sticks (BIOS USB legacy support),
 * SATA in any mode, NVMe with CSM, CDs — without a driver for each controller.
 * Low memory used: 0x6000 DAP/parameter buffers, 0x6F00 thunk parameters,
 * 0x7000 thunk code, 0x20000-0x27FFF bounce buffer. */
#include "ata.h"

extern u8 thunk_blob[], thunk_blob_end[], gdt_desc[];

static inline void outb(u16 p, u8 v) { __asm__ volatile("outb %0,%1" ::"a"(v), "Nd"(p)); }
static inline u8 inb(u16 p) { u8 v; __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"(p)); return v; }

#define PARAMS ((volatile u32 *)(uintptr_t)0x6f00)
#define DAP ((volatile u8 *)(uintptr_t)0x6000)
#define BOUNCE ((u8 *)(uintptr_t)0x20000)
#define BOUNCE_SIZE 0x8000u

static int ready;

void bios_init(void) {
    if (ready) return;
    u32 n = (u32)(thunk_blob_end - thunk_blob);
    e_memcpy((void *)(uintptr_t)0x7000, thunk_blob, n);
    e_memcpy((void *)(uintptr_t)0x6f20, gdt_desc, 6);
    ready = 1;
}

/* INT n with the parameter block's registers; returns 0 if CF is clear */
static int bios_int(u8 n) {
    bios_init();
    ((volatile u8 *)(uintptr_t)0x6f00)[28] = n;
    u8 m0 = inb(0x21), m1 = inb(0xa1);
    outb(0x21, m0 | 0x02); outb(0xa1, m1 | 0x10);        /* keyboard and mouse stay ours */
    ((void (*)(void))(uintptr_t)0x7000)();
    outb(0x21, m0); outb(0xa1, m1);
    return (PARAMS[6] & 1) ? -1 : 0;
}

/* sector size and count of BIOS drive `drive` (0 on success) */
int bios_disk_params(int drive, u32 *secsz, u64 *sectors) {
    volatile u8 *b = DAP;
    for (int i = 0; i < 0x42; i++) b[i] = 0;
    b[0] = 0x1e;
    PARAMS[0] = 0x4800; PARAMS[3] = (u32)drive; PARAMS[4] = 0x6000;
    if (bios_int(0x13)) return -1;
    u32 ss = (u32)b[0x18] | (u32)b[0x19] << 8;
    u64 n = 0;
    for (int i = 7; i >= 0; i--) n = n << 8 | b[0x10 + i];
    if (ss != 512 && ss != 2048) return -1;
    *secsz = ss;
    *sectors = n;
    return 0;
}

/* read or write count sectors (of the drive's size) at lba */
int bios_disk_rw(int drive, u32 secsz, u64 lba, u32 count, u8 *buf, int write) {
    u32 per = BOUNCE_SIZE / secsz;
    while (count) {
        u32 n = count > per ? per : count;
        if (write) e_memcpy(BOUNCE, buf, n * secsz);
        int ok = -1;
        for (int t = 0; t < 3 && ok; t++) {
            volatile u8 *d = DAP;
            d[0] = 0x10; d[1] = 0;
            d[2] = (u8)n; d[3] = (u8)(n >> 8);
            d[4] = 0; d[5] = 0;                               /* buffer 2000:0000 */
            d[6] = 0x00; d[7] = 0x20;
            for (int i = 0; i < 8; i++) d[8 + i] = (u8)(lba >> (8 * i));
            PARAMS[0] = write ? 0x4300 : 0x4200;
            PARAMS[3] = (u32)drive; PARAMS[4] = 0x6000;
            ok = bios_int(0x13);
        }
        if (ok) return -1;
        if (!write) e_memcpy(buf, BOUNCE, n * secsz);
        lba += n; count -= n; buf += n * secsz;
    }
    return 0;
}

/* a BIOS drive as an AtaDev: sectors counts 512-byte units for disks, 2048 for CDs */
int bios_open(AtaDev *d, int drive) {
    u32 ss; u64 n;
    if (bios_disk_params(drive, &ss, &n)) return -1;
    e_memset(d, 0, sizeof *d);
    d->bios = 1; d->dev = drive; d->secsz = ss; d->sectors = n; d->ch = -1;
    return 0;
}
