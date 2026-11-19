/* Machine wiring, run loop, and the tiny freestanding runtime the core needs. */
#include "emu.h"

void *e_memcpy(void *d, const void *s, size_t n) { return __builtin_memcpy(d, s, n); }
void *e_memmove(void *d, const void *s, size_t n) { return __builtin_memmove(d, s, n); }
void *e_memset(void *d, int v, size_t n) { return __builtin_memset(d, v, n); }
size_t e_strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }

/* ---------- minimal formatted output (%s %d %u %x %llx %llu %lld %p %c %%) ---------- */

static void put(char *o, u32 cap, u32 *n, char ch) { if (*n + 1 < cap) o[*n] = ch; (*n)++; }

static void put_num(char *o, u32 cap, u32 *n, u64 v, int base, int neg, int width, char padc) {
    char buf[24];
    int i = 0;
    do {
        u32 dg = (u32)(v % (u64)base);
        buf[i++] = (char)(dg < 10 ? '0' + dg : 'a' + dg - 10);
        v /= (u64)base;
    } while (v);
    if (neg) buf[i++] = '-';
    while (i < width) buf[i++] = padc;
    while (i) put(o, cap, n, buf[--i]);
}

int emu_vsnprintf(char *o, u32 cap, const char *f, __builtin_va_list ap) {
    u32 n = 0;
    for (; *f; f++) {
        if (*f != '%') { put(o, cap, &n, *f); continue; }
        f++;
        char padc = ' ';
        int width = 0, longs = 0;
        if (*f == '0') { padc = '0'; f++; }
        while (*f >= '0' && *f <= '9') width = width * 10 + (*f++ - '0');
        while (*f == 'l') { longs++; f++; }
        switch (*f) {
        case 's': {
            const char *s = __builtin_va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s) put(o, cap, &n, *s++);
            break;
        }
        case 'c': put(o, cap, &n, (char)__builtin_va_arg(ap, int)); break;
        case 'd': {
            i64 v = longs >= 2 ? __builtin_va_arg(ap, long long) : longs ? __builtin_va_arg(ap, long) : __builtin_va_arg(ap, int);
            put_num(o, cap, &n, v < 0 ? (u64)0 - (u64)v : (u64)v, 10, v < 0, width, padc);
            break;
        }
        case 'u': case 'x': {
            u64 v = longs >= 2 ? __builtin_va_arg(ap, unsigned long long)
                  : longs ? __builtin_va_arg(ap, unsigned long) : __builtin_va_arg(ap, unsigned);
            put_num(o, cap, &n, v, *f == 'x' ? 16 : 10, 0, width, padc);
            break;
        }
        case 'p': put(o, cap, &n, '0'); put(o, cap, &n, 'x');
                  put_num(o, cap, &n, (u64)(uintptr_t)__builtin_va_arg(ap, void *), 16, 0, 0, ' '); break;
        case '%': put(o, cap, &n, '%'); break;
        default: put(o, cap, &n, '?'); break;
        }
    }
    if (cap) o[n < cap ? n : cap - 1] = 0;
    return (int)n;
}

void emu_log(Machine *m, const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    emu_vsnprintf(m->log_line, sizeof m->log_line, fmt, ap);
    __builtin_va_end(ap);
    if (m->host.log) m->host.log(m->host.ctx, m->log_line);
}

void emu_fatal(Cpu *c, const char *fmt, ...) {
    Machine *m = c->m;
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    emu_vsnprintf(m->log_line, sizeof m->log_line, fmt, ap);
    __builtin_va_end(ap);
    if (m->host.log) {
        m->host.log(m->host.ctx, "FATAL:");
        m->host.log(m->host.ctx, m->log_line);
    }
    c->fatal = 1;
    c->exc_vec = -1;
    __builtin_longjmp(c->jmp, 1);
}

/* the guest asked for a reset: leave the run loops (fatal 3), the embedder calls machine_reboot */
void emu_reset_request(Cpu *c) {
    emu_log(c->m, "guest reset");
    c->fatal = 3;
    c->exc_vec = -1;
    __builtin_longjmp(c->jmp, 1);
}

/* ---------- machine ---------- */

void machine_init(Machine *m, u8 *ram, u64 ram_size, Host host) {
    e_memset(m, 0, sizeof *m);
    m->ram = ram;
    m->ram_size = ram_size;
    m->host = host;
    m->ps_per_insn = 1000;
    m->ps_rem = 0;
    cpu_reset(&m->cpu, m);
    m->cpus[0] = &m->cpu; m->cur = &m->cpu; m->ncpu = 1; m->cpu.cpu_index = 0;
    devices_init(m);
}

void cpu_set_reset_vector(Cpu *c);

void machine_bios_reset(Machine *m, u8 *bios, u32 bios_size, const u8 *vga_rom, u32 vga_rom_size) {
    m->bios = bios;
    m->bios_size = bios_size;
    m->vga_rom = vga_rom;
    m->vga_rom_size = vga_rom_size;
    m->chipset = 1;
    chipset_init(m);
    acpi_init(m);
    apic_reset(m);
    m->cmos[0x3d] = 0x23;          /* boot order (Bochs/QEMU CMOS convention): CD-ROM, then hard disk */
    m->cmos[0x38] = 0x01;          /* skip the floppy boot signature check */
    cpu_set_reset_vector(&m->cpu);
}

void cpu_init_ap(Cpu *c, Machine *m, int index);
void ide_reset(Machine *m);
void jit_flush_code(Machine *m);
void machine_reboot(Machine *m) {
    u64 ic = m->cpu.icount;
    int jit = m->cpu.jit_on;
    cpu_reset(&m->cpu, m);
    m->cpu.icount = ic; m->cpu.jit_on = jit; m->cpu.cpu_index = 0;
    for (int i = 1; i < m->ncpu; i++) { u64 ica = m->cpus[i]->icount; cpu_init_ap(m->cpus[i], m, i); m->cpus[i]->icount = ica; }
    m->cur = &m->cpu;
    e_memset(m->pic, 0, sizeof m->pic);
    e_memset(m->pit, 0, sizeof m->pit);
    m->kbc_q.count = m->kbc_q.head = 0;
    devices_init(m);
    ide_reset(m);
    if (m->bios) machine_bios_reset(m, m->bios, m->bios_size, m->vga_rom, m->vga_rom_size);
    if (m->jit) jit_flush_code(m);
    m->batch_icount0 = m->cpu.icount;
}

void machine_set_framebuffer(Machine *m, u8 *fb, u64 phys, u32 w, u32 h, u32 pitch) {
    m->fb = fb;
    m->fb_phys = phys;
    m->fb_width = w;
    m->fb_height = h;
    m->fb_pitch = pitch;
    m->fb_size = (pitch * h + 0xfffu) & ~0xfffu;
    m->fb_bpp = 32;
    m->fb_start = 0;
    for (int ci = 0; ci < m->ncpu; ci++) tlb_flush(m->cpus[ci]);
}

void machine_set_vram_size(Machine *m, u32 bytes) { m->fb_size = bytes; for (int ci = 0; ci < m->ncpu; ci++) tlb_flush(m->cpus[ci]); }

u64 machine_now(Machine *m) { return m->ns + ((m->cur->icount - m->batch_icount0) * m->ps_per_insn + m->ps_rem) / 1000; }

void machine_event_changed(Machine *m) {
    /* the run loop counts this instruction after it completes */
    if (m->ncpu > 1) return;   /* several CPUs: turns end only at the quantum, PAUSE or HLT (the JIT cannot stop mid-chain) */
    if (m->cur->run_n > m->cur->run_i + 1) m->cur->run_n = m->cur->run_i + 1;
}

void cpu_init_ap(Cpu *c, Machine *m, int index);
void machine_set_cpus(Machine *m, int n) {
    if (n < 1) n = 1;
    if (n > MAX_CPUS) n = MAX_CPUS;
    m->ncpu = n;
    for (int i = 1; i < n; i++) {
        m->cpus[i] = &m->cpu_ap[i - 1];
        m->cpu_ap[i - 1].jit_on = m->cpu.jit_on;
        cpu_init_ap(m->cpus[i], m, i);
    }
}

/* several CPUs on one host thread: rounds of up to one slice per CPU in the same
 * stretch of virtual time (deterministic: switches at instruction counts) */
#define SMP_QUANTUM 262144u
static u64 machine_run_smp(Machine *m, u64 n) {
    u64 done = 0;
    while (done < n) {
        for (int i = 0; i < m->ncpu; i++) if (m->cpus[i]->fatal) { m->cpu.fatal = m->cpus[i]->fatal; m->cur = &m->cpu; return done; }
        u64 next = devices_next_event(m);
        u64 slice = m->smp_quantum ? m->smp_quantum : SMP_QUANTUM;
        if (next != ~0ull) {
            u64 until = next > m->ns ? ((next - m->ns) * 1000 - m->ps_rem) / m->ps_per_insn + 1 : 1;
            if (until < slice) slice = until;
        }
        int ran_any = 0;
        for (int i = 0; i < m->ncpu; i++) {
            Cpu *c = m->cpus[i];
            m->cur = c;
            if (c->wait_sipi) continue;
            u64 start = c->icount;
            m->batch_icount0 = start;
            if (c->halted) {
                if (!(c->rflags & F_IF)) continue;                       /* stopped until INIT/NMI */
                if (!apic_has_interrupt(m)) continue;                    /* idle */
                cpu_step(c);                                            /* deliver the interrupt */
            }
            if (!c->halted && c->icount - start < slice) cpu_run(c, slice - (c->icount - start));
            u64 ran = c->icount - start;
            if (ran) ran_any = 1;
            done += ran;
            if (c->fatal) break;
        }
        m->cur = &m->cpu;
        if (!ran_any) {
            /* everybody waits: jump to the next device event */
            if (next == ~0ull) {
                int stopped = 1;
                for (int i = 0; i < m->ncpu; i++) if (!m->cpus[i]->wait_sipi && (m->cpus[i]->rflags & F_IF)) stopped = 0;
                emu_log(m, stopped ? "all CPUs halted with interrupts disabled" : "all CPUs halted with no timer armed");
                m->cpu.fatal = 2;
                return done;
            }
            if (next > m->ns) { m->ns = next; m->ps_rem = 0; }
            devices_tick(m);
            done += 64;
            continue;
        }
        { u64 ps = slice * m->ps_per_insn + m->ps_rem; m->ns += ps / 1000; m->ps_rem = ps % 1000; }
        devices_tick(m);
    }
    return done;
}

u64 machine_run(Machine *m, u64 n) {
    if (m->ncpu > 1) return machine_run_smp(m, n);
    Cpu *c = &m->cpu;
    u64 done = 0;
    while (done < n && !c->fatal) {
        if (c->halted) {
            if (!(c->rflags & F_IF)) {
                emu_log(m, "CPU halted with interrupts disabled at %llx", (unsigned long long)c->rip);
                c->fatal = 2;
                break;
            }
            if (!(m->chipset ? apic_has_interrupt(m) : pic_has_pending(m))) {
                u64 next = devices_next_event(m);
                if (next == ~0ull) {
                    emu_log(m, "CPU halted with no timer armed");
                    c->fatal = 2;
                    break;
                }
                if (next > m->ns) { m->ns = next; m->ps_rem = 0; }
                m->batch_icount0 = c->icount;
                devices_tick(m);
                done += 64;            /* idle time still counts against the budget */
                continue;
            }
            cpu_step(c);
            m->batch_icount0 = c->icount;
            continue;
        }
        u64 next = devices_next_event(m);
        u64 slice = 1u << 20;     /* batches still end exactly at the next device event */
        if (next != ~0ull) {
            u64 until = next > m->ns ? ((next - m->ns) * 1000 - m->ps_rem) / m->ps_per_insn + 1 : 1;
            if (until < slice) slice = until;
        }
        if (slice > n - done) slice = n - done;
        u64 start = c->icount;
        m->batch_icount0 = start;
        cpu_run(c, slice);
        u64 ran = c->icount - start;
        if (!ran) ran = 1;
        done += ran;
        { u64 ps = ran * m->ps_per_insn + m->ps_rem; m->ns += ps / 1000; m->ps_rem = ps % 1000; }
        m->batch_icount0 = c->icount;
        devices_tick(m);
    }
    return done;
}
