#ifndef W32_ATA_H
#define W32_ATA_H
#include "../emu/emu.h"

typedef struct {
    int ch, dev;        /* channel 0 primary / 1 secondary, device 0 master / 1 slave */
    u64 sectors;        /* ATAPI: 2048-byte sectors; ATA: 512-byte sectors */
    int lba48;
    int ahci;           /* on an AHCI port (ch = port number) */
    int bios;           /* BIOS drive (INT 13h): number in dev, sector size in secsz */
    u32 secsz;
} AtaDev;

int atapi_find_cd(AtaDev *d, u8 *scratch2048);
int atapi_read(AtaDev *d, u32 lba, u32 count, u8 *buf);
int ata_find_disk(AtaDev *d, const AtaDev *skip);
int ata_read(AtaDev *d, u64 lba, u32 count, u8 *buf);
int ata_write(AtaDev *d, u64 lba, u32 count, const u8 *buf);
int ata_flush(AtaDev *d);
int ahci_init(void);
int ahci_find(AtaDev *d, int atapi, const AtaDev *skip, int *from);
int ahci_atapi_read(AtaDev *d, u32 lba, u32 count, u8 *buf);
int ahci_rw(AtaDev *d, u64 lba, u32 count, u8 *buf, int write);
int ahci_flush(AtaDev *d);
void bios_init(void);
int bios_disk_params(int drive, u32 *secsz, u64 *sectors);
int bios_disk_rw(int drive, u32 secsz, u64 lba, u32 count, u8 *buf, int write);
int bios_open(AtaDev *d, int drive);

#endif
