/* Multiboot2 direct kernel boot: replaces GRUB for a Multiboot2 ELF kernel.
 * Loads PT_LOAD segments and modules into guest RAM, builds the boot
 * information structure, and leaves the CPU in the handoff state the
 * specification defines (32-bit flat protected mode, paging off). */
#include "emu.h"

#define MB2_HEADER_MAGIC 0xe85250d6u
#define MB2_BOOT_MAGIC 0x36d76289u

static u32 rd32(const u8 *p) { u32 v; e_memcpy(&v, p, 4); return v; }
static u16 rd16(const u8 *p) { u16 v; e_memcpy(&v, p, 2); return v; }
static u64 rd64(const u8 *p) { u64 v; e_memcpy(&v, p, 8); return v; }

typedef struct { u8 *base; u32 n, cap; } Buf;

static void b_put(Buf *b, const void *p, u32 n) {
    if (b->n + n <= b->cap) e_memcpy(b->base + b->n, p, n);
    b->n += n;
}
static void b_u32(Buf *b, u32 v) { b_put(b, &v, 4); }
static void b_u64(Buf *b, u64 v) { b_put(b, &v, 8); }
static void b_u8(Buf *b, u8 v) { b_put(b, &v, 1); }
static void b_u16(Buf *b, u16 v) { b_put(b, &v, 2); }
static void b_align(Buf *b) { while (b->n & 7) b_u8(b, 0); }
static u32 b_tag(Buf *b, u32 type) { b_align(b); u32 at = b->n; b_u32(b, type); b_u32(b, 0); return at; }
static void b_end(Buf *b, u32 at) { u32 sz = b->n - at; if (at + 8 <= b->cap) e_memcpy(b->base + at + 4, &sz, 4); }

static void mmap_entry(Buf *b, u64 base, u64 len, u32 type) {
    b_u64(b, base); b_u64(b, len); b_u32(b, type); b_u32(b, 0);
}

int mb2_boot(Machine *m, const u8 *elf, u32 elf_size, const char *cmdline,
             const BootModuleSrc *mods, int nmods) {
    /* --- find and check the Multiboot2 header --- */
    u32 hdr = ~0u;
    for (u32 off = 0; off + 16 <= elf_size && off < 32768; off += 8) {
        if (rd32(elf + off) == MB2_HEADER_MAGIC) {
            u32 arch = rd32(elf + off + 4), len = rd32(elf + off + 8), sum = rd32(elf + off + 12);
            if (arch == 0 && (u32)(MB2_HEADER_MAGIC + arch + len + sum) == 0) { hdr = off; break; }
        }
    }
    if (hdr == ~0u) { emu_log(m, "mb2: no Multiboot2 header"); return -1; }
    u32 hlen = rd32(elf + hdr + 8);
    for (u32 t = hdr + 16; t + 8 <= hdr + hlen;) {
        u16 type = rd16(elf + t);
        u32 size = rd32(elf + t + 4);
        if (type == 0) break;
        if (type == 5)
            emu_log(m, "mb2: kernel asks for a %ux%ux%u framebuffer", rd32(elf + t + 8), rd32(elf + t + 12), rd32(elf + t + 16));
        t += (size + 7) & ~7u;
    }

    /* --- ELF --- */
    if (elf_size < 64 || rd32(elf) != 0x464c457fu) { emu_log(m, "mb2: not an ELF file"); return -1; }
    int is64 = elf[4] == 2;
    u64 entry, phoff;
    u32 phnum, phentsize;
    if (is64) { entry = rd64(elf + 24); phoff = rd64(elf + 32); phentsize = rd16(elf + 54); phnum = rd16(elf + 56); }
    else { entry = rd32(elf + 24); phoff = rd32(elf + 28); phentsize = rd16(elf + 42); phnum = rd16(elf + 44); }
    u64 kend = 0;
    for (u32 i = 0; i < phnum; i++) {
        const u8 *ph = elf + phoff + (u64)i * phentsize;
        if (phoff + (u64)(i + 1) * phentsize > elf_size) return -1;
        if (rd32(ph) != 1) continue;   /* PT_LOAD */
        u64 off, paddr, filesz, memsz;
        if (is64) { off = rd64(ph + 8); paddr = rd64(ph + 24); filesz = rd64(ph + 32); memsz = rd64(ph + 40); }
        else { off = rd32(ph + 4); paddr = rd32(ph + 12); filesz = rd32(ph + 16); memsz = rd32(ph + 20); }
        if (paddr + memsz > m->ram_size || off + filesz > elf_size || filesz > memsz) {
            emu_log(m, "mb2: segment %u at %llx+%llx does not fit guest RAM", i, (unsigned long long)paddr,
                    (unsigned long long)memsz);
            return -1;
        }
        e_memcpy(m->ram + (uintptr_t)paddr, elf + off, (size_t)filesz);
        e_memset(m->ram + (uintptr_t)(paddr + filesz), 0, (size_t)(memsz - filesz));
        if (paddr + memsz > kend) kend = paddr + memsz;
        emu_log(m, "mb2: loaded segment %llx-%llx", (unsigned long long)paddr, (unsigned long long)(paddr + memsz));
    }

    /* --- modules, page aligned after the kernel --- */
    u64 at = (kend + 0xfff) & ~0xfffull;
    u64 mstart[8], mend[8];
    if (nmods > 8) nmods = 8;
    for (int i = 0; i < nmods; i++) {
        if (at + mods[i].size > m->ram_size) { emu_log(m, "mb2: module %d does not fit", i); return -1; }
        e_memcpy(m->ram + (uintptr_t)at, mods[i].data, mods[i].size);
        mstart[i] = at;
        mend[i] = at + mods[i].size;
        emu_log(m, "mb2: module '%s' at %llx-%llx", mods[i].name, (unsigned long long)mstart[i], (unsigned long long)mend[i]);
        at = (mend[i] + 0xfff) & ~0xfffull;
    }

    /* --- boot information --- */
    u64 info = at;
    if (info + 65536 > m->ram_size) return -1;
    Buf b = { m->ram + (uintptr_t)info, 0, 65536 };
    b_u32(&b, 0); b_u32(&b, 0);
    u32 t = b_tag(&b, 1);
    b_put(&b, cmdline, (u32)e_strlen(cmdline) + 1);
    b_end(&b, t);
    t = b_tag(&b, 2);
    b_put(&b, "W32Eleven", 10);
    b_end(&b, t);
    for (int i = 0; i < nmods; i++) {
        t = b_tag(&b, 3);
        b_u32(&b, (u32)mstart[i]); b_u32(&b, (u32)mend[i]);
        b_put(&b, mods[i].name, (u32)e_strlen(mods[i].name) + 1);
        b_end(&b, t);
    }
    t = b_tag(&b, 4);
    b_u32(&b, 639); b_u32(&b, (u32)((m->ram_size >> 10) - 1024));
    b_end(&b, t);
    t = b_tag(&b, 6);
    b_u32(&b, 24); b_u32(&b, 0);
    mmap_entry(&b, 0, 0x9fc00, 1);
    mmap_entry(&b, 0x9fc00, 0x400, 2);
    mmap_entry(&b, 0xf0000, 0x10000, 2);
    mmap_entry(&b, 0x100000, m->ram_size - 0x100000, 1);
    if (m->fb) mmap_entry(&b, m->fb_phys, m->fb_size, 2);
    b_end(&b, t);
    if (m->fb) {
        t = b_tag(&b, 8);
        b_u64(&b, m->fb_phys);
        b_u32(&b, m->fb_pitch); b_u32(&b, m->fb_width); b_u32(&b, m->fb_height);
        b_u8(&b, 32); b_u8(&b, 1); b_u16(&b, 0);
        b_u8(&b, 16); b_u8(&b, 8); b_u8(&b, 8); b_u8(&b, 8); b_u8(&b, 0); b_u8(&b, 8);
        b_end(&b, t);
    }
    t = b_tag(&b, 0);
    b_end(&b, t);
    if (b.n > b.cap) return -1;
    e_memcpy(b.base, &b.n, 4);

    /* --- CPU handoff --- */
    Cpu *c = &m->cpu;
    cpu_set_flat32(c);
    c->r[0] = MB2_BOOT_MAGIC;
    c->r[3] = info;
    c->rip = entry;
    c->rflags = 2;
    emu_log(m, "mb2: entry %llx, info at %llx (%u bytes), cmdline '%s'", (unsigned long long)entry,
            (unsigned long long)info, b.n, cmdline);
    return 0;
}
