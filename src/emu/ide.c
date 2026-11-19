/* PIIX IDE: two channels (0x1F0/0x3F6 IRQ 14, 0x170/0x376 IRQ 15), PIO only.
 * Devices: ATAPI CD-ROM (2048-byte sectors) or ATA disk (512-byte sectors),
 * both backed by the embedder's Host.disk_read / disk_write callbacks. */
#include "emu.h"
void jit_invalidate_page(Machine *m, u64 pa);

enum { ST_ERR = 0x01, ST_DRQ = 0x08, ST_DSC = 0x10, ST_DRDY = 0x40, ST_BSY = 0x80 };

typedef struct {
    int present, atapi;
    u64 sectors;             /* in device blocks (2048 for ATAPI, 512 for ATA) */
} IdeDev;

typedef struct {
    IdeDev dev[2];
    int sel;                 /* selected device */
    u8 feat, err, nsect, lba[3], lba_hob[3], nsect_hob, devhead, status, ctrl;
    u8 buf[65536 + 2048];
    u32 pos, len;            /* current PIO buffer */
    /* pending multi-block transfer */
    int xfer;                /* 0 none, 1 ATAPI read, 2 ATA read, 3 ATA write, 4 packet CDB */
    u64 next_lba;
    u32 blocks_left, per_drq, bytes_limit;
    u8 sense_key, asc;
    int unit_base;           /* unit number of dev[0] for the host callbacks */
    /* bus-master DMA (PIIX BMIDE registers) */
    u8 bm_cmd, bm_status;
    u32 bm_prdt;
    int dma_mode;            /* PACKET issued with the DMA feature bit */
    int dma_pending;         /* 1 = device->memory from disk, 2 = memory->device, 3 = device->memory from buf */
    u64 dma_lba;
    u32 dma_blocks, dma_bsize;
} IdeChan;

static IdeChan g_ch[2];

static int chan_irq(int ch) { return ch ? 15 : 14; }

static void raise_irq(Machine *m, int ch) {
    if (!(g_ch[ch].ctrl & 2)) pic_raise(m, chan_irq(ch));
}

static IdeDev *cur(IdeChan *c) { return &c->dev[c->sel]; }

/* machine reset: controller state cleared, attached devices kept */
void ide_reset(Machine *m);
void ide_reset(Machine *m) {
    (void)m;
    for (int i = 0; i < 2; i++) {
        IdeDev d0 = g_ch[i].dev[0], d1 = g_ch[i].dev[1];
        int ub = g_ch[i].unit_base;
        e_memset(&g_ch[i], 0, sizeof g_ch[i]);
        g_ch[i].dev[0] = d0; g_ch[i].dev[1] = d1; g_ch[i].unit_base = ub;
        g_ch[i].status = ST_DRDY | ST_DSC;
    }
}

void ide_attach(Machine *m, int unit, int atapi, u64 bytes) {
    (void)m;
    IdeChan *c = &g_ch[unit / 2];
    IdeDev *d = &c->dev[unit & 1];
    d->present = 1;
    d->atapi = atapi;
    d->sectors = bytes / (atapi ? 2048 : 512);
}

static void set_signature(IdeChan *c) {
    IdeDev *d = cur(c);
    c->nsect = 1; c->lba[0] = 1;
    if (d->atapi) { c->lba[1] = 0x14; c->lba[2] = 0xeb; }
    else { c->lba[1] = 0; c->lba[2] = 0; }
}

static void put_str(u8 *id, int word, int nwords, const char *s) {
    for (int i = 0; i < nwords * 2; i++) {
        char ch = *s ? *s++ : ' ';
        id[2 * word + (i ^ 1)] = (u8)ch;
    }
}

static void identify(IdeChan *c, int packet) {
    IdeDev *d = cur(c);
    u8 *id = c->buf;
    e_memset(id, 0, 512);
    if (packet) {
        id[0] = 0x80; id[1] = 0x85;            /* ATAPI, CD-ROM, removable, 12-byte packets */
    } else {
        id[0] = 0x40;
        u32 cyl = (u32)(d->sectors / (16 * 63)); if (cyl > 16383) cyl = 16383;
        id[2] = (u8)cyl; id[3] = (u8)(cyl >> 8); id[6] = 16; id[12] = 63;
        id[94] = 0x01; id[95] = 0x80;          /* word 47: READ/WRITE MULTIPLE, 1 sector per block */
        id[118] = 0x01; id[119] = 0x01;        /* word 59: multiple setting valid, 1 */
    }
    put_str(id, 10, 10, "W32E0001");
    put_str(id, 23, 4, "1.0");
    put_str(id, 27, 20, packet ? "W32Eleven DVD-ROM" : "W32Eleven HARDDISK");
    id[98] = 0x00; id[99] = 0x03;              /* capabilities: LBA, DMA (advertised; PIO used) */
    id[106] = 0x06;                            /* words 64-70, 88 valid */
    id[128] = 0x03;                            /* PIO modes 3,4 */
    id[126] = 0x07; id[127] = 0x04;            /* word 63: MWDMA 0-2 supported, mode 2 selected */
    id[176] = 0x3f; id[177] = 0x20;            /* word 88: UDMA 0-5 supported, mode 5 selected */
    if (!packet) {
        u32 s28 = d->sectors > 0x0fffffff ? 0x0fffffff : (u32)d->sectors;
        id[120] = (u8)s28; id[121] = (u8)(s28 >> 8); id[122] = (u8)(s28 >> 16); id[123] = (u8)(s28 >> 24);
        id[166] = 0x00; id[167] = 0x44;        /* word 83: valid, LBA48 supported */
        id[168] = 0x00; id[169] = 0x40;        /* word 84: valid */
        id[174] = 0x00; id[175] = 0x40;        /* word 87: valid */
        id[172] = 0x00; id[173] = 0x04;
        for (int i = 0; i < 8; i++) id[200 + i] = (u8)(d->sectors >> (8 * i));
    }
    id[160] = 0x7e; id[161] = 0x00;            /* ATA/ATAPI-1..6 */
    c->pos = 0; c->len = 512;
    c->status = ST_DRDY | ST_DSC | ST_DRQ;
    c->xfer = 0;
}

static void abort_cmd(IdeChan *c) {
    c->err = 0x04;
    c->status = ST_DRDY | ST_ERR;
    c->len = c->pos = 0;
}

/* ---- bus-master DMA ---- */

static void dma_start_check(Machine *m, int ch);

/* walk the PRD table moving `total` bytes; read_dev(): device->memory */
static int dma_run(Machine *m, int ch) {
    IdeChan *c = &g_ch[ch];
    int to_mem = c->dma_pending != 2;
    u64 prd = c->bm_prdt;
    u32 done_blk = 0, off_in_buf = 0, buf_len = 0;
    int unit = ch * 2 + c->sel;
    int ok = 1;
    if (c->dma_pending == 3) buf_len = c->len;
    u32 total = c->dma_pending == 3 ? c->len : c->dma_blocks * c->dma_bsize;
    u32 moved = 0;
    for (int guard = 0; guard < 4096 && moved < total; guard++) {
        if (prd + 8 > m->ram_size) { ok = 0; break; }
        u32 addr, cnt;
        e_memcpy(&addr, m->ram + (uintptr_t)prd, 4);
        e_memcpy(&cnt, m->ram + (uintptr_t)prd + 4, 4);
        int eot = (cnt >> 31) & 1;
        cnt &= 0xffff;
        if (!cnt) cnt = 0x10000;
        while (cnt && moved < total) {
            if (c->dma_pending != 3 && off_in_buf == buf_len) {
                /* refill the staging buffer with up to 32 blocks */
                u32 n = c->dma_blocks - done_blk;
                if (n > 32) n = 32;
                if (c->dma_pending == 1) {
                    if (!m->host.disk_read || m->host.disk_read(m->host.ctx, unit, (c->dma_lba + done_blk) * c->dma_bsize, n * c->dma_bsize, c->buf)) { ok = 0; break; }
                }
                buf_len = n * c->dma_bsize;
                off_in_buf = 0;
            }
            u32 chunk = cnt;
            if (chunk > buf_len - off_in_buf) chunk = buf_len - off_in_buf;
            if ((u64)addr + chunk > m->ram_size) { ok = 0; break; }
            if (to_mem) {
                /* device writes to RAM: translated code on those pages is stale now */
                if (m->jit && chunk) for (u32 pg = addr >> 12; pg <= (addr + chunk - 1) >> 12; pg++) jit_invalidate_page(m, (u64)pg << 12);
                e_memcpy(m->ram + (uintptr_t)addr, c->buf + off_in_buf, chunk);
            }
            else e_memcpy(c->buf + off_in_buf, m->ram + (uintptr_t)addr, chunk);
            addr += chunk; cnt -= chunk; off_in_buf += chunk; moved += chunk;
            if (c->dma_pending != 3 && off_in_buf == buf_len) {
                u32 n = buf_len / c->dma_bsize;
                if (c->dma_pending == 2 && (!m->host.disk_write ||
                    m->host.disk_write(m->host.ctx, unit, (c->dma_lba + done_blk) * c->dma_bsize, buf_len, c->buf))) { ok = 0; break; }
                done_blk += n;
            }
        }
        if (!ok || eot) break;
        prd += 8;
    }
    if (moved < total) ok = 0;
    return ok;
}

static void dma_finish(Machine *m, int ch, int ok) {
    IdeChan *c = &g_ch[ch];
    if (m->trace_bios >= 2) emu_log(m, "IDE ch%d DMA done ok %d kind %d blocks %u prdt %x", ch, ok, c->dma_pending, c->dma_blocks, c->bm_prdt);
    c->dma_pending = 0;
    c->bm_status = (u8)((c->bm_status & ~1u) | 4u | (ok ? 0 : 2u));
    c->len = c->pos = 0;
    c->xfer = 0;
    if (cur(c)->atapi) {
        c->nsect = 3;
        if (ok) { c->status = ST_DRDY | ST_DSC; c->err = 0; }
        else { c->sense_key = 3; c->asc = 0x11; c->status = ST_DRDY | ST_ERR; c->err = 0x30; }
    } else {
        c->status = ok ? (u8)(ST_DRDY | ST_DSC) : (u8)(ST_DRDY | ST_ERR);
        if (!ok) c->err = 0x40;
    }
    raise_irq(m, ch);
}

static void dma_start_check(Machine *m, int ch) {
    IdeChan *c = &g_ch[ch];
    if (c->dma_pending && (c->bm_cmd & 1)) dma_finish(m, ch, dma_run(m, ch));
}

static void dma_begin(Machine *m, int ch, int kind, u64 lba, u32 blocks, u32 bsize) {
    IdeChan *c = &g_ch[ch];
    c->dma_pending = kind;
    c->dma_lba = lba; c->dma_blocks = blocks; c->dma_bsize = bsize;
    c->status = ST_DRDY | ST_DSC | ST_DRQ;
    if (cur(c)->atapi) c->nsect = kind == 2 ? 0 : 2;
    dma_start_check(m, ch);
}

/* BMIDE registers: base+0 command, +2 status, +4 PRD table (channel 1 at +8) */
int ide_bm_io(Machine *m, u16 port, int write, u32 *v, int size) {
    u32 base = chipset_ide_bm_base(m);
    if (!base || port < base || port >= base + 16) return 0;
    int ch = (port - base) >= 8;
    u32 r = (port - base) & 7;
    IdeChan *c = &g_ch[ch];
    if (!write) {
        u32 val = 0;
        for (int i = 0; i < size; i++) {
            u32 o = r + (u32)i, b = 0;
            if (o == 0) b = c->bm_cmd;
            else if (o == 2) b = c->bm_status;
            else if (o >= 4 && o < 8) b = (c->bm_prdt >> (8 * (o - 4))) & 0xff;
            val |= b << (8 * i);
        }
        *v = val;
        return 1;
    }
    for (int i = 0; i < size; i++) {
        u32 o = r + (u32)i, b = (*v >> (8 * i)) & 0xff;
        if (o == 0) {
            int start = (b & 1) && !(c->bm_cmd & 1);
            c->bm_cmd = (u8)(b & 9);
            if (b & 1) c->bm_status |= 1; else c->bm_status &= (u8)~1u;
            if (start) dma_start_check(m, ch);
        } else if (o == 2) {
            c->bm_status = (u8)((c->bm_status & ~(b & 6) & ~0x60u) | (b & 0x60));
        } else if (o >= 4 && o < 8) {
            u32 sh = 8 * (o - 4);
            c->bm_prdt = (c->bm_prdt & ~(0xffu << sh)) | (b << sh);
            c->bm_prdt &= ~3u;
        }
    }
    return 1;
}

/* ---- ATAPI ---- */

static void atapi_done(Machine *m, int ch, int ok) {
    IdeChan *c = &g_ch[ch];
    c->len = c->pos = 0;
    c->xfer = 0;
    c->nsect = 3;                              /* I/O=1, CoD=1: command completed */
    if (ok) { c->status = ST_DRDY | ST_DSC; c->err = 0; }
    else { c->status = ST_DRDY | ST_ERR; c->err = (u8)(c->sense_key << 4); }
    raise_irq(m, ch);
}

static void atapi_data(Machine *m, int ch, u32 n) {
    IdeChan *c = &g_ch[ch];
    if (c->dma_mode) { c->len = n; c->pos = 0; dma_begin(m, ch, 3, 0, 0, 1); return; }
    c->pos = 0; c->len = n;
    c->lba[1] = (u8)n; c->lba[2] = (u8)(n >> 8);
    c->nsect = 2;                              /* I/O=1 (to host), CoD=0 */
    c->status = ST_DRDY | ST_DSC | ST_DRQ;
    raise_irq(m, ch);
}

static void atapi_read_next(Machine *m, int ch) {
    IdeChan *c = &g_ch[ch];
    if (!c->blocks_left) { atapi_done(m, ch, 1); return; }
    u32 n = c->bytes_limit / 2048;
    if (n == 0) n = 1;
    if (n > c->blocks_left) n = c->blocks_left;
    if (n > 32) n = 32;
    int unit = ch * 2 + c->sel;
    if (!m->host.disk_read || m->host.disk_read(m->host.ctx, unit, c->next_lba * 2048, n * 2048, c->buf)) {
        c->sense_key = 3; c->asc = 0x11;       /* medium error */
        atapi_done(m, ch, 0);
        return;
    }
    c->next_lba += n;
    c->blocks_left -= n;
    c->xfer = 1;
    atapi_data(m, ch, n * 2048);
}

static void put_be32(u8 *p, u32 v) { p[0] = (u8)(v >> 24); p[1] = (u8)(v >> 16); p[2] = (u8)(v >> 8); p[3] = (u8)v; }

static void atapi_packet(Machine *m, int ch) {
    IdeChan *c = &g_ch[ch];
    IdeDev *d = cur(c);
    u8 cdb[12];
    e_memcpy(cdb, c->buf, 12);
    if (m->trace_bios >= 2) emu_log(m, "ATAPI ch%d pkt %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x dma %d", ch, cdb[0], cdb[1], cdb[2],
                                     cdb[3], cdb[4], cdb[5], cdb[6], cdb[7], cdb[8], cdb[9], c->dma_mode);
    u32 alloc;
    c->xfer = 0;
    switch (cdb[0]) {
    case 0x00: case 0x1b: case 0x1e: case 0x2b: case 0x35: case 0xbb:   /* TUR, START/STOP, PREVENT, SEEK, SYNC, SET SPEED */
        c->sense_key = 0; atapi_done(m, ch, 1); return;
    case 0x03: {   /* REQUEST SENSE */
        e_memset(c->buf, 0, 18);
        c->buf[0] = 0x70; c->buf[2] = c->sense_key; c->buf[7] = 10; c->buf[12] = c->asc;
        alloc = cdb[4] < 18 ? cdb[4] : 18;
        c->sense_key = 0; c->asc = 0;
        atapi_data(m, ch, alloc);
        return;
    }
    case 0x12: {   /* INQUIRY */
        e_memset(c->buf, 0, 36);
        c->buf[0] = 0x05; c->buf[1] = 0x80; c->buf[2] = 0x05; c->buf[3] = 0x32; c->buf[4] = 31;
        e_memcpy(c->buf + 8, "W32ELEVN", 8);
        e_memcpy(c->buf + 16, "DVD-ROM         ", 16);
        e_memcpy(c->buf + 32, "1.0 ", 4);
        alloc = cdb[4] < 36 ? cdb[4] : 36;
        atapi_data(m, ch, alloc);
        return;
    }
    case 0x25:     /* READ CAPACITY */
        put_be32(c->buf, (u32)(d->sectors - 1));
        put_be32(c->buf + 4, 2048);
        atapi_data(m, ch, 8);
        return;
    case 0x28: case 0xa8: {   /* READ(10) / READ(12) */
        u32 lba = (u32)cdb[2] << 24 | (u32)cdb[3] << 16 | (u32)cdb[4] << 8 | cdb[5];
        u32 cnt = cdb[0] == 0x28 ? ((u32)cdb[7] << 8 | cdb[8])
                                 : ((u32)cdb[6] << 24 | (u32)cdb[7] << 16 | (u32)cdb[8] << 8 | cdb[9]);
        if ((u64)lba + cnt > d->sectors) { c->sense_key = 5; c->asc = 0x21; atapi_done(m, ch, 0); return; }
        if (c->dma_mode) {
            if (!cnt) { atapi_done(m, ch, 1); return; }
            dma_begin(m, ch, 1, lba, cnt, 2048);
            return;
        }
        c->next_lba = lba;
        c->blocks_left = cnt;
        atapi_read_next(m, ch);
        return;
    }
    case 0x43: {   /* READ TOC (format 0, single data track) */
        int msf = (cdb[1] >> 1) & 1;
        u8 *b = c->buf;
        e_memset(b, 0, 20);
        b[1] = 18; b[2] = 1; b[3] = 1;
        b[5] = 0x14; b[6] = 1; put_be32(b + 8, 0);
        b[13] = 0x16; b[14] = 0xaa;
        u32 end = (u32)d->sectors;
        if (msf) { u32 f = end + 150; b[16] = 0; b[17] = (u8)(f / 75 / 60); b[18] = (u8)(f / 75 % 60); b[19] = (u8)(f % 75); }
        else put_be32(b + 16, end);
        alloc = (u32)cdb[7] << 8 | cdb[8];
        atapi_data(m, ch, alloc < 20 ? alloc : 20);
        return;
    }
    case 0x46: {   /* GET CONFIGURATION: DVD-ROM current profile */
        e_memset(c->buf, 0, 8);
        put_be32(c->buf, 4); c->buf[6] = 0x00; c->buf[7] = 0x10;
        alloc = (u32)cdb[7] << 8 | cdb[8];
        atapi_data(m, ch, alloc < 8 ? alloc : 8);
        return;
    }
    case 0x4a: {   /* GET EVENT STATUS NOTIFICATION: no events */
        e_memset(c->buf, 0, 8);
        c->buf[1] = 6; c->buf[2] = 0x04; c->buf[3] = 0x10;
        alloc = (u32)cdb[7] << 8 | cdb[8];
        atapi_data(m, ch, alloc < 8 ? alloc : 8);
        return;
    }
    case 0x5a: {   /* MODE SENSE(10): empty header */
        e_memset(c->buf, 0, 8);
        c->buf[1] = 6; c->buf[2] = 0x70;
        alloc = (u32)cdb[7] << 8 | cdb[8];
        atapi_data(m, ch, alloc < 8 ? alloc : 8);
        return;
    }
    case 0x51: {   /* READ DISC INFORMATION */
        e_memset(c->buf, 0, 34);
        c->buf[1] = 32; c->buf[2] = 0x0e; c->buf[3] = 1; c->buf[4] = 1; c->buf[5] = 1; c->buf[6] = 1;
        alloc = (u32)cdb[7] << 8 | cdb[8];
        atapi_data(m, ch, alloc < 34 ? alloc : 34);
        return;
    }
    case 0xbd: {   /* MECHANISM STATUS */
        e_memset(c->buf, 0, 8);
        alloc = (u32)cdb[8] << 8 | cdb[9];
        atapi_data(m, ch, alloc < 8 ? alloc : 8);
        return;
    }
    default:
        emu_log(m, "ATAPI: unsupported packet command %02x", cdb[0]);
        c->sense_key = 5; c->asc = 0x20;       /* illegal request, invalid command */
        atapi_done(m, ch, 0);
        return;
    }
}

/* ---- ATA disk ---- */

static u64 ata_lba(IdeChan *c, int lba48) {
    if (lba48)
        return (u64)c->lba[0] | (u64)c->lba[1] << 8 | (u64)c->lba[2] << 16 |
               (u64)c->lba_hob[0] << 24 | (u64)c->lba_hob[1] << 32 | (u64)c->lba_hob[2] << 40;
    return (u64)c->lba[0] | (u64)c->lba[1] << 8 | (u64)c->lba[2] << 16 | (u64)(c->devhead & 15) << 24;
}

static void ata_read_next(Machine *m, int ch) {
    IdeChan *c = &g_ch[ch];
    if (!c->blocks_left) { c->xfer = 0; c->status = ST_DRDY | ST_DSC; return; }
    int unit = ch * 2 + c->sel;
    if (!m->host.disk_read || m->host.disk_read(m->host.ctx, unit, c->next_lba * 512, 512, c->buf)) {
        abort_cmd(c); raise_irq(m, ch); return;
    }
    c->next_lba++;
    c->blocks_left--;
    c->pos = 0; c->len = 512;
    c->xfer = 2;
    c->status = ST_DRDY | ST_DSC | ST_DRQ;
    raise_irq(m, ch);
}

/* PIO write: DRQ for the next sector, or finish */
static void ata_write_next(Machine *m, int ch, int first) {
    IdeChan *c = &g_ch[ch];
    if (!first) {
        int unit = ch * 2 + c->sel;
        if (!m->host.disk_write || m->host.disk_write(m->host.ctx, unit, c->next_lba * 512, 512, c->buf)) {
            abort_cmd(c); raise_irq(m, ch); return;
        }
        c->next_lba++;
        c->blocks_left--;
    }
    if (!c->blocks_left) { c->xfer = 0; c->len = c->pos = 0; c->status = ST_DRDY | ST_DSC; raise_irq(m, ch); return; }
    c->pos = 0; c->len = 512;
    c->xfer = 3;
    c->status = ST_DRDY | ST_DSC | ST_DRQ;
    if (!first) raise_irq(m, ch);                  /* no interrupt before the first sector */
}

static void command(Machine *m, int ch, u8 cmd) {
    IdeChan *c = &g_ch[ch];
    IdeDev *d = cur(c);
    if (m->trace_bios >= 2) emu_log(m, "IDE ch%d dev%d cmd %02x feat %02x present %d", ch, c->sel, cmd, c->feat, d->present);
    if (!d->present) { return; }
    c->err = 0;
    switch (cmd) {
    case 0xec:                                         /* IDENTIFY DEVICE */
        if (d->atapi) { set_signature(c); abort_cmd(c); raise_irq(m, ch); return; }
        identify(c, 0); raise_irq(m, ch); return;
    case 0xa1:                                         /* IDENTIFY PACKET DEVICE */
        if (!d->atapi) { abort_cmd(c); raise_irq(m, ch); return; }
        identify(c, 1); raise_irq(m, ch); return;
    case 0xa0:                                         /* PACKET */
        if (!d->atapi) { abort_cmd(c); raise_irq(m, ch); return; }
        c->dma_mode = c->feat & 1;
        c->bytes_limit = (u32)c->lba[1] | (u32)c->lba[2] << 8;
        if (!c->bytes_limit || c->bytes_limit == 0xffff) c->bytes_limit = 0xfffe;
        c->pos = 0; c->len = 12;
        c->xfer = 4;
        c->nsect = 1;                                  /* CoD=1, I/O=0 */
        c->status = ST_DRDY | ST_DSC | ST_DRQ;
        return;
    case 0x08:                                         /* DEVICE RESET */
        set_signature(c);
        c->status = 0; c->err = 1;
        return;
    case 0x90:                                         /* EXECUTE DEVICE DIAGNOSTIC */
        set_signature(c); c->err = 1;
        c->status = d->atapi ? 0 : ST_DRDY | ST_DSC;
        raise_irq(m, ch);
        return;
    case 0xef: case 0x91: case 0xe0: case 0xe1: case 0xe7: case 0xea: case 0xc6: case 0xf5:
        c->status = ST_DRDY | ST_DSC; raise_irq(m, ch); return;   /* features, idle, flush: accepted */
    case 0xc8: case 0xc9: case 0x25: case 0xca: case 0xcb: case 0x35: {   /* READ/WRITE DMA [EXT] */
        if (d->atapi) { set_signature(c); abort_cmd(c); raise_irq(m, ch); return; }
        int ext = cmd == 0x25 || cmd == 0x35;
        u32 n = ext ? ((u32)c->nsect_hob << 8 | c->nsect) : c->nsect;
        if (!n) n = ext ? 65536 : 256;
        dma_begin(m, ch, (cmd == 0xca || cmd == 0xcb || cmd == 0x35) ? 2 : 1, ata_lba(c, ext), n, 512);
        return;
    }
    case 0x20: case 0x21: case 0x24: case 0xc4: case 0x29: {   /* READ SECTOR(S) [EXT], READ MULTIPLE */
        if (d->atapi) { set_signature(c); abort_cmd(c); raise_irq(m, ch); return; }
        int ext = cmd == 0x24 || cmd == 0x29;
        u32 n = ext ? ((u32)c->nsect_hob << 8 | c->nsect) : c->nsect;
        if (!n) n = ext ? 65536 : 256;
        c->next_lba = ata_lba(c, ext);
        c->blocks_left = n;
        ata_read_next(m, ch);
        return;
    }
    case 0x30: case 0x31: case 0x34: case 0xc5: case 0x39: {   /* WRITE SECTOR(S) [EXT], WRITE MULTIPLE [EXT] */
        if (d->atapi) { set_signature(c); abort_cmd(c); raise_irq(m, ch); return; }
        int ext = cmd == 0x34 || cmd == 0x39;
        u32 n = ext ? ((u32)c->nsect_hob << 8 | c->nsect) : c->nsect;
        if (!n) n = ext ? 65536 : 256;
        c->next_lba = ata_lba(c, ext);
        c->blocks_left = n;
        ata_write_next(m, ch, 1);
        return;
    }
    case 0xe5: case 0x98:                               /* CHECK POWER MODE: active */
        c->nsect = 0xff; c->status = ST_DRDY | ST_DSC; raise_irq(m, ch); return;
    default:
        emu_log(m, "IDE: unsupported command %02x on unit %d", cmd, ch * 2 + c->sel);
        abort_cmd(c); raise_irq(m, ch);
        return;
    }
}

/* data register */
static u32 data_read(Machine *m, int ch, int size) {
    IdeChan *c = &g_ch[ch];
    if (!(c->status & ST_DRQ) || c->xfer == 4) return 0xffffffffu;
    u32 v = 0;
    for (int i = 0; i < size && c->pos < c->len; i++) v |= (u32)c->buf[c->pos++] << (8 * i);
    if (c->pos >= c->len) {
        if (c->xfer == 1) atapi_read_next(m, ch);
        else if (c->xfer == 2) ata_read_next(m, ch);
        else if (cur(c)->atapi && c->len && c->nsect == 2) atapi_done(m, ch, 1);
        else { c->status = ST_DRDY | ST_DSC; c->len = c->pos = 0; }
    }
    return v;
}

/* REP INS fast path: copy up to `bytes` (a multiple of the element size) from the
 * current PIO buffer of the channel owning data port `port`. Returns bytes copied;
 * 0 means "use the per-element path". Stops at the end of the current buffer so
 * that refills and completion happen exactly as for single reads. */
u32 ide_pio_read_bulk(Machine *m, u16 port, u8 *dst, u32 bytes, int elem) {
    (void)m;
    int ch;
    if (port == 0x1f0) ch = 0; else if (port == 0x170) ch = 1; else return 0;
    IdeChan *c = &g_ch[ch];
    if (!(c->dev[0].present || c->dev[1].present) || !cur(c)->present) return 0;
    if (!(c->status & ST_DRQ) || c->xfer == 4 || c->pos >= c->len) return 0;
    u32 n = c->len - c->pos;
    if (n > bytes) n = bytes;
    n -= n % (u32)elem;
    if (n <= (u32)elem) return 0;
    n -= (u32)elem;                      /* leave the last element to the normal path (it triggers refills) */
    e_memcpy(dst, c->buf + c->pos, n);
    c->pos += n;
    return n;
}

static void data_write(Machine *m, int ch, u32 v, int size) {
    IdeChan *c = &g_ch[ch];
    if (!(c->status & ST_DRQ)) return;
    for (int i = 0; i < size && c->pos < c->len; i++) c->buf[c->pos++] = (u8)(v >> (8 * i));
    if (c->pos >= c->len && c->xfer == 4) atapi_packet(m, ch);
    else if (c->pos >= c->len && c->xfer == 3) ata_write_next(m, ch, 0);
}

int ide_io(Machine *m, u16 port, int write, u32 *v, int size);
static int g_in_trace;
static int ide_io_trace_read(Machine *m, u16 port, int size) {
    if (g_in_trace) return 0;
    g_in_trace = 1;
    /* peek without side effects is not possible for every register; log after the real read instead */
    g_in_trace = 0;
    (void)m; (void)port; (void)size;
    return 0;
}

int ide_io(Machine *m, u16 port, int write, u32 *v, int size) {
    int ch;
    u32 r;
    if (port >= 0x1f0 && port <= 0x1f7) { ch = 0; r = port - 0x1f0u; }
    else if (port >= 0x170 && port <= 0x177) { ch = 1; r = port - 0x170u; }
    else if (port == 0x3f6) { ch = 0; r = 8; }
    else if (port == 0x376) { ch = 1; r = 8; }
    else return 0;
    IdeChan *c = &g_ch[ch];
    int any = c->dev[0].present || c->dev[1].present;
    if (m->trace_bios >= 2 && m->cpu.mode64 && r != 0) {
        static int n;
        if (n++ < 400) {
            if (write) emu_log(m, "IDE64 out ch%d reg%u <- %x (status %x)", ch, r, *v & 0xff, c->status);
            else {
                int rr = ide_io_trace_read(m, port, size);
                (void)rr;
            }
        }
    }
    if (!write) {
        if (!any) { *v = 0xffffffffu; return 1; }
        if (!cur(c)->present && r != 0 && r != 6) { *v = 0; return 1; }   /* absent device on a live channel */
        switch (r) {
        case 0: *v = data_read(m, ch, size); return 1;
        case 1: *v = c->err; return 1;
        case 2: *v = (c->ctrl & 0x80) ? c->nsect_hob : c->nsect; return 1;
        case 3: case 4: case 5: *v = (c->ctrl & 0x80) ? c->lba_hob[r - 3] : c->lba[r - 3]; return 1;
        case 6: *v = c->devhead; return 1;
        case 7: pic_lower(m, chan_irq(ch)); *v = c->status; return 1;
        default: *v = c->status; return 1;       /* alternate status */
        }
    }
    u8 b = (u8)*v;
    switch (r) {
    case 0: data_write(m, ch, *v, size); return 1;
    case 1: c->feat = b; return 1;
    case 2: c->nsect_hob = c->nsect; c->nsect = b; return 1;
    case 3: case 4: case 5: c->lba_hob[r - 3] = c->lba[r - 3]; c->lba[r - 3] = b; return 1;
    case 6: c->devhead = b; c->sel = (b >> 4) & 1; return 1;
    case 7: if (any) command(m, ch, b); return 1;
    default:
        if ((b & 4) && !(c->ctrl & 4)) {         /* software reset */
            for (int i = 0; i < 2; i++) if (c->dev[i].present) { int s = c->sel; c->sel = i; set_signature(c); c->sel = s; }
            c->status = ST_BSY;
            c->xfer = 0; c->len = c->pos = 0;
        } else if (!(b & 4) && (c->ctrl & 4)) {
            c->status = cur(c)->atapi ? 0 : ST_DRDY | ST_DSC;
            c->err = 1;
            set_signature(c);
        }
        c->ctrl = b;
        return 1;
    }
}
