/* W32Eleven x86-64 system emulator core.
 *
 * Portable, freestanding C. The same sources build for the bare-metal i386
 * runtime and for the hosted Linux development harness (tools/hosted.c).
 * Nothing in here touches host hardware; the embedder supplies RAM, the
 * framebuffer and a small set of callbacks in struct Host.
 */
#ifndef W32_EMU_H
#define W32_EMU_H
#include <stdint.h>
#include <stddef.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

/* ---------- embedder interface ---------- */
typedef struct Host {
    void (*serial_out)(void *ctx, u8 c);     /* COM1 transmit */
    void (*debug_out)(void *ctx, u8 c);      /* port 0xE9 */
    void (*log)(void *ctx, const char *s);   /* emulator diagnostics */
    u64 (*wall_unix)(void *ctx);             /* seconds since 1970 for the RTC */
    /* block devices (IDE units 0-3: primary master/slave, secondary master/slave);
     * return 0 on success */
    int (*disk_read)(void *ctx, int unit, u64 offset, u32 len, u8 *buf);
    int (*disk_write)(void *ctx, int unit, u64 offset, u32 len, const u8 *buf);
    void *ctx;
} Host;

/* ---------- CPU ---------- */
typedef union {
    u8 b[16]; u16 w[8]; u32 d[4]; u64 q[2];
    i8 sb[16]; i16 sw[8]; i32 sd[4]; i64 sq[2];
    float f[4]; double fd[2];
} Xmm;

typedef struct {
    u16 sel;
    u64 base;
    u32 limit;
    u16 attr;          /* descriptor bits 40..55: type,S,DPL,P,(limit),AVL,L,D,G */
} Seg;

enum { ES, CS, SS, DS, FS, GS };

enum {
    F_CF = 1u << 0, F_PF = 1u << 2, F_AF = 1u << 4, F_ZF = 1u << 6, F_SF = 1u << 7,
    F_TF = 1u << 8, F_IF = 1u << 9, F_DF = 1u << 10, F_OF = 1u << 11,
    F_ARITH = F_CF | F_PF | F_AF | F_ZF | F_SF | F_OF
};

/* lazy flag producers */
enum { LF_NONE, LF_ADD, LF_ADC, LF_SUB, LF_SBB, LF_LOGIC, LF_INC, LF_DEC, LF_SHL, LF_SHR, LF_SAR, LF_MUL };
/* LF_MUL: lf_res = truncated product, lf_c = CF = OF; AF reads as 0 */
/* shifts: lf_a = operand, lf_b = masked nonzero count; AF reads as 0 */

#define TLB_SIZE 4096
#define TLB_CODE 8u     /* tag bit on a write entry: page holds translated code (JIT) */
typedef struct {
    u64 tag;            /* (linear & ~0xfff) | 1, 0 = invalid */
    u8 *host;           /* host pointer of the page */
#if UINTPTR_MAX == 0xffffffffu
    u32 addend;         /* host - (u32)linear page: generated code adds the address (16-byte entries) */
#endif
} TlbEntry;

struct Machine;

typedef struct Cpu {
    u64 r[16];
    u64 rip;
    u64 rflags;         /* arithmetic bits are stale while lf_op != LF_NONE */
    u64 lf_res, lf_a, lf_b;
    u16 lf_op, lf_size; /* adjacent: generated code writes both with one store */
    u32 lf_c;           /* lf_res/lf_a/lf_b above lf_size bytes may hold garbage (readers mask) */
    u32 jit_tmp;        /* scratch for generated code */
    u32 jit_x[4];       /* 16-byte scratch for generated code */
    u32 jit_esp, jit_limit;   /* host stack at chain entry, and the deepest call-frame stack allowed */
    int cpl;
    u32 tlb_mode;         /* low bits of every TLB tag: 1 supervisor, 2 user (cpl 3), epoch << 4 */
    u32 jit_sr;           /* generated code: host address of guest RSP at block entry (stack region) */

    u64 icount_legacy;  /* instructions interpreted outside 64-bit mode (statistics) */
    u64 icount_legacy32; /* ... of which in 32-bit code */        /* scratch for generated code */

    Seg seg[6];
    struct { u64 base; u16 limit; } gdtr, idtr;
    Seg ldtr, tr;
    u64 cr0, cr2, cr3, cr4, cr8, efer;
    u64 gs_kernel, star, lstar, cstar, sfmask, pat, tsc_aux, apic_base, misc_enable;
    u64 sysenter_cs, sysenter_esp, sysenter_eip;
    u64 mtrr_def, mtrr_var[16], mtrr_fix[11];

    Xmm xmm[16];
    u32 mxcsr;
    u16 fcw, fsw;       /* fsw holds TOP in bits 11-13 */
    u16 ftw;            /* full tag word, 2 bits per physical register (3 = empty) */
    u8 fpr[8][10];      /* physical x87 registers, 80-bit */

    int mode64;         /* executing 64-bit code (LMA && CS.L) */
    int def32;          /* CS.D: default operand size 32 */
    /* (cpl, tlb_mode: near the start, for short offsets in generated code) */
    u32 tlb_epoch;        /* bumped by a flush instead of clearing the TLB (8 bits) */
    int halted;
    int shadow;         /* interrupt shadow after STI / MOV SS */

    TlbEntry tlb_r[TLB_SIZE], tlb_w[TLB_SIZE], tlb_x[TLB_SIZE];
    /* entries holding global pages survive CR3 writes: listed here, re-tagged */
    u8 tlb_g[3][TLB_SIZE];
    u16 glist[3 * TLB_SIZE];
    u32 nglist;
    /* paging-structure cache (long mode): 2 MiB region -> page-table base and the
     * accumulated R/W, U/S, NX of the upper levels; valid for one TLB epoch */
    struct { u64 tag; u64 table; u32 bits, epoch; } psc[256];
    int walk_global;    /* last page walk ended in a global page */

    int exc_vec, exc_has_err, delivering;
    int sys_access;     /* implicit supervisor access (descriptor tables, TSS, interrupt frames) */
    u32 exc_err;

    u8 *fetch_host;     /* host page of the current fetch page */
    u64 fetch_la;       /* its linear address; 1 = invalid */
    u64 run_i, run_n;   /* batch loop state (kept in memory across longjmp) */
    int jit_on, in_block;
    u32 jit_skip;       /* interpret this many more instructions before asking the JIT (cold code) */
    u64 blk_icount0;
    u64 icount_base;    /* icount + chain_budget when C entered generated code */
    u32 chain_budget;   /* steps left for chained blocks */
    u32 chain_from;     /* host address of the exit jump that left the last block (0 = not chainable) */
    u32 jit_sr_fail;    /* the last block left because its stack-region check failed */
    u32 jit_br[16][2];  /* generated code: per base register, host address and page offset of the block's first access */
    int apic_pending;   /* this CPU's local APIC IRR is not empty */
    int cpu_index;      /* 0 = bootstrap processor */
    int wait_sipi;      /* application processor waiting for a startup IPI */
    int yield_req;      /* PAUSE with several CPUs: end this CPU's turn after the instruction */

    u64 icount;
    struct Machine *m;
    void *jmp[5];       /* __builtin_setjmp buffer for exceptions */
    int fatal;          /* nonzero: stop, emulator cannot continue */
} Cpu;

/* ---------- devices ---------- */
typedef struct {
    u8 irr, imr, isr, icw_step, icw4, vector, read_isr, auto_eoi, elcr;
} Pic;

typedef struct {
    u16 reload, latch;
    u8 mode, access, rw_lo, latched, w_lo, gate, out_pending;
    u64 start_ns;
} PitChannel;

typedef struct {
    u8 data[64];
    u8 count, head;
} ByteQueue;

#define MAX_CPUS 2
typedef struct Machine {
    Cpu cpu;            /* CPU 0, the bootstrap processor */
    Cpu *cpus[MAX_CPUS];
    Cpu *cur;           /* the CPU executing right now */
    int ncpu;
    u32 smp_quantum;    /* instructions per CPU turn with several CPUs */
    Cpu cpu_ap[MAX_CPUS - 1];   /* application processors */
    Host host;

    u8 *ram;            /* guest physical [0, ram_size) */
    u64 ram_size;
    u8 *fb;             /* guest framebuffer, mapped at fb_phys */
    u64 fb_phys;
    u32 fb_width, fb_height, fb_pitch, fb_size, fb_bpp;
    u32 fb_start;           /* byte offset of the visible image (VBE panning) */

    u64 ns;             /* virtual time at the start of the current batch */
    u64 batch_icount0;  /* cpu.icount at that point: now = ns + ((icount - batch_icount0) * ps_per_insn + ps_rem) / 1000 */
    u32 ps_per_insn;    /* guest clock: picoseconds per instruction (1000 = 1 GHz) */
    u64 ps_rem;         /* sub-nanosecond remainder carried between batches */

    Pic pic[2];
    PitChannel pit[3];
    u8 port61;
    u8 cmos_index, cmos_nmi;
    u8 cmos[128];
    u64 boot_unix;

    u8 uart[8];         /* IER,IIR/FCR,LCR,MCR,LSR,MSR,SCR (+DLL/DLM) */
    u8 uart_dll, uart_dlm, uart_rbr, uart_rx_full;

    u8 kbc_cfg, kbc_cmd, kbc_out_port;
    ByteQueue kbc_q;      /* bytes waiting for port 0x60 */
    u8 kbc_q_aux[64];     /* per queued byte: 1 = from mouse */
    u8 kbd_expect, mouse_expect, mouse_rate_seq[3], mouse_id, mouse_stream;
    u8 kbd_scanset;

    u32 pci_addr;

    struct Jit *jit;     /* NULL: interpreter only */

    /* firmware boot (NULL bios: Multiboot direct boot, no chipset devices) */
    const u8 *bios_rom;
    u8 *bios;            /* ROM image (writable copy is never written) */
    u32 bios_size;
    const u8 *vga_rom;
    u32 vga_rom_size;
    int chipset;
    int trace_bios;
    u32 smbios_cores;       /* nonzero: own SMBIOS tables (fw_cfg) with this processor core count */
    u64 break_rip;       /* debugging: log registers when execution reaches this address (0 = off) */
    int break_max;       /* hits to log (>3: one compact line each) */      /* log real-mode INT 13h/15h requests (debugging) */

    char log_line[160];
} Machine;

/* ---------- API ---------- */
void machine_init(Machine *m, u8 *ram, u64 ram_size, Host host);
void machine_set_framebuffer(Machine *m, u8 *fb, u64 phys, u32 w, u32 h, u32 pitch);
/* Load a Multiboot2 kernel (ELF64/ELF32 image in host memory) plus modules and
 * put the CPU into the Multiboot2 handoff state. Returns 0 on success. */
typedef struct { const u8 *data; u32 size; const char *name; } BootModuleSrc;
int mb2_boot(Machine *m, const u8 *elf, u32 elf_size, const char *cmdline,
             const BootModuleSrc *mods, int nmods);
/* Boot from an ISO 9660 image via its /boot/grub/grub.cfg menu. entry < 0 = the
 * menu's default. extra_args is appended to the kernel command line. */
int iso_boot(Machine *m, const u8 *iso, u64 size, int entry, const char *extra_args);
/* Enable the x86-64 -> i386 JIT (i386 hosts only). mem must be writable and
 * executable; it holds generated code and the JIT's tables. Returns 0 on success. */
int jit_init(Machine *m, u8 *mem, u32 size);
/* Power on with firmware: ROM at the top of 4 GiB (and E0000-FFFFF through the
 * i440FX PAM registers), CPU at the reset vector F000:FFF0. */
void machine_bios_reset(Machine *m, u8 *bios, u32 bios_size, const u8 *vga_rom, u32 vga_rom_size);
/* Execute up to n instructions. Returns the number executed. */
u64 machine_run(Machine *m, u64 n);
/* Number of CPUs (call after machine_init, before machine_bios_reset; firmware boots only). */
void machine_set_cpus(Machine *m, int n);
/* Guest reset (CPU fatal == 3 after machine_run): CPUs, devices and firmware restart, RAM and disks stay. */
void machine_reboot(Machine *m);
void emu_reset_request(Cpu *c);
void machine_key(Machine *m, u8 scancode);           /* set-1 scancode byte */
void machine_mouse(Machine *m, int dx, int dy, int buttons);

/* ---------- internal ---------- */
void emu_log(Machine *m, const char *fmt, ...);
__attribute__((noreturn)) void emu_fatal(Cpu *c, const char *fmt, ...);
int emu_vsnprintf(char *out, u32 cap, const char *fmt, __builtin_va_list ap);

u32 io_in(Machine *m, u16 port, int size);
void io_out(Machine *m, u16 port, u32 v, int size);
void pic_raise(Machine *m, int irq);
void pic_lower(Machine *m, int irq);
int pic_pending(Machine *m);          /* -1 none, else vector (acknowledges) */
int pic_has_pending(Machine *m);
void devices_init(Machine *m);
void chipset_init(Machine *m);
int chipset_map(Machine *m, u64 pa, int write, u8 **out);
const u8 *chipset_vga_rom(Machine *m, u64 pa);
int chipset_pm_io(Machine *m, u16 port, int write, u32 *v, int size);
int ide_io(Machine *m, u16 port, int write, u32 *v, int size);
int ide_bm_io(Machine *m, u16 port, int write, u32 *v, int size);
u32 ide_pio_read_bulk(Machine *m, u16 port, u8 *dst, u32 bytes, int elem);
u32 chipset_ide_bm_base(Machine *m);
int fwcfg_io(Machine *m, u16 port, int write, u32 *v, int size);
int vbe_io(Machine *m, u16 port, int write, u32 *v, int size);
/* firmware boots: make the whole framebuffer buffer (VRAM) addressable through the VGA BAR */
void machine_set_vram_size(Machine *m, u32 bytes);
void acpi_init(Machine *m);
void apic_reset(Machine *m);
int apic_has_interrupt(Machine *m);
int apic_ack(Machine *m);
void apic_isa_irq(Machine *m, int irq);
void ioapic_set_irq(Machine *m, int gsi, int high);
u64 apic_next_event(Machine *m);
void apic_tick(Machine *m);
int apic_mmio(Machine *m, u64 pa, int write, u64 *v, int sz);
void apic_set_tpr_from_cr8(Machine *m, u64 cr8);
/* attach an IDE device: unit 0-3, atapi = CD-ROM (2048-byte blocks) */
void ide_attach(Machine *m, int unit, int atapi, u64 bytes);
u32 pci_cfg_read(Machine *m, u32 addr, int size);
void pci_cfg_write(Machine *m, u32 addr, u32 v, int size);
void devices_tick(Machine *m);
u64 devices_next_event(Machine *m);
u64 machine_now(Machine *m);           /* exact virtual time, also inside a batch */
void machine_event_changed(Machine *m); /* end the current batch after this instruction */

void cpu_reset(Cpu *c, Machine *m);
void cpu_set_flat32(Cpu *c);
void cpu_step(Cpu *c);
u64 cpu_run(Cpu *c, u64 n);       /* up to n steps; stops early when halted */
void cpu_interrupt(Cpu *c, int vector);
void tlb_flush(Cpu *c);

void *e_memcpy(void *d, const void *s, size_t n);
void *e_memmove(void *d, const void *s, size_t n);
void *e_memset(void *d, int v, size_t n);
size_t e_strlen(const char *s);

#endif
