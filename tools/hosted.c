/* Hosted development harness: runs the emulator core as a Linux process.
 *
 *   hosted [options] kernel.elf [name=path ...]
 *   hosted [options] --iso image.iso [--entry N]
 *   hosted [options] --bios bios.bin [--vgabios vgabios.bin]   (firmware boot)
 *     --cmdline STR      kernel command line (default "mode=live")
 *     --ram MIB          guest RAM (default 512)
 *     --insns N          stop after N guest instructions (default unlimited)
 *     --seconds S        stop after S seconds of host time
 *     --shot FILE.png    write the guest framebuffer on exit
 *     --shot-every N     also write it every N instructions (FILE.png is overwritten)
 *     --no-jit           interpreter only (the JIT exists on i386 builds only)
 *     --text             print the VGA text screen (B8000, 80x25) on exit
 *     --hash             print a hash of guest RAM, framebuffer and registers on exit
 *                        (virtual time is instruction-driven, so runs are deterministic)
 *
 * Guest COM1 goes to stdout, emulator diagnostics to stderr.
 * This is a development tool; the product runtime is the bare-metal image.
 */
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include "../src/emu/emu.h"

static void h_serial(void *ctx, u8 c) { (void)ctx; fputc(c, stdout); if (c == '\n') fflush(stdout); }
static void h_debug(void *ctx, u8 c) { (void)ctx; fputc(c, stderr); }
static void h_log(void *ctx, const char *s) { (void)ctx; fprintf(stderr, "[w32] %s\n", s); }
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
static int disk_fd[4] = { -1, -1, -1, -1 };
/* --snapshot: writes to the hard disk go to an unnamed sparse overlay file;
 * the image itself is never modified. One bit per 512-byte sector. */
static int ovl_fd = -1;
static u8 *ovl_map;
static int ovl_has(u64 s) { return (ovl_map[s >> 3] >> (s & 7)) & 1; }
static int h_disk_read(void *ctx, int unit, u64 off, u32 len, u8 *buf) {
    (void)ctx;
    if (unit < 0 || unit > 3 || disk_fd[unit] < 0) return -1;
    if (unit == 0 && ovl_fd >= 0 && !(off & 511) && !(len & 511)) {
        for (u32 i = 0; i < len; i += 512) {
            u64 s = (off + i) >> 9;
            int fd = ovl_has(s) ? ovl_fd : disk_fd[0];
            if (pread(fd, buf + i, 512, (off_t)(off + i)) != 512) return -1;
        }
        return 0;
    }
    return pread(disk_fd[unit], buf, len, (off_t)off) == (ssize_t)len ? 0 : -1;
}
static int h_disk_write(void *ctx, int unit, u64 off, u32 len, const u8 *buf) {
    (void)ctx;
    if (unit < 0 || unit > 3 || disk_fd[unit] < 0) return -1;
    if (unit == 0 && ovl_fd >= 0) {
        if ((off & 511) || (len & 511)) return -1;
        if (pwrite(ovl_fd, buf, len, (off_t)off) != (ssize_t)len) return -1;
        for (u32 i = 0; i < len; i += 512) { u64 s = (off + i) >> 9; ovl_map[s >> 3] |= (u8)(1u << (s & 7)); }
        return 0;
    }
    return pwrite(disk_fd[unit], buf, len, (off_t)off) == (ssize_t)len ? 0 : -1;
}
static u64 fixed_wall;
static int m_trace_bios, want_events;
static double guest_ghz;   /* instructions per guest nanosecond (default 1) */
static u64 break_rip;
static int break_max = 3;
static u64 peek_va;
static u64 h_wall(void *ctx) { (void)ctx; return fixed_wall ? fixed_wall : (u64)time(NULL); }

static u8 *read_file(const char *path, u32 *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    u8 *p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { perror(path); exit(1); }
    fclose(f);
    *size = (u32)n;
    return p;
}

/* ---- minimal PNG writer (stored deflate blocks) ---- */
static u32 crc_table[256];
static void crc_init(void) {
    for (u32 n = 0; n < 256; n++) { u32 c = n; for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1; crc_table[n] = c; }
}
static u32 crc(u32 c, const u8 *p, size_t n) { c = ~c; while (n--) c = crc_table[(c ^ *p++) & 0xff] ^ (c >> 8); return ~c; }
static void be32(u8 *p, u32 v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void chunk(FILE *f, const char *type, const u8 *data, u32 n) {
    u8 h[8]; be32(h, n); memcpy(h + 4, type, 4);
    fwrite(h, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    u32 c = crc(0, (const u8 *)type, 4); c = crc(c, data, n);
    u8 t[4]; be32(t, c); fwrite(t, 1, 4, f);
}

static u32 shot_bpp = 32;
static void write_png(const char *path, const u8 *fb, u32 w, u32 h, u32 pitch) {
    static int inited;
    if (!inited) { crc_init(); inited = 1; }
    size_t raw_n = (size_t)h * (1 + 3 * w);
    u8 *raw = malloc(raw_n);
    for (u32 y = 0; y < h; y++) {
        u8 *o = raw + (size_t)y * (1 + 3 * w);
        *o++ = 0;
        const u8 *s = fb + (size_t)y * pitch;
        for (u32 x = 0; x < w; x++) {
            if (shot_bpp == 32) { o[0] = s[4 * x + 2]; o[1] = s[4 * x + 1]; o[2] = s[4 * x]; }
            else if (shot_bpp == 24) { o[0] = s[3 * x + 2]; o[1] = s[3 * x + 1]; o[2] = s[3 * x]; }
            else if (shot_bpp == 16) { u16 p = (u16)(s[2 * x] | s[2 * x + 1] << 8); o[0] = (u8)((p >> 11) << 3); o[1] = (u8)(((p >> 5) & 63) << 2); o[2] = (u8)((p & 31) << 3); }
            else { o[0] = o[1] = o[2] = s[x]; }
            o += 3;
        }
    }
    size_t nblk = (raw_n + 65534) / 65535;
    size_t z_n = 2 + raw_n + nblk * 5 + 4;
    u8 *z = malloc(z_n), *o = z;
    *o++ = 0x78; *o++ = 0x01;
    u32 a = 1, b = 0;
    for (size_t off = 0; off < raw_n;) {
        size_t len = raw_n - off > 65535 ? 65535 : raw_n - off;
        *o++ = off + len == raw_n;
        *o++ = len & 0xff; *o++ = len >> 8; *o++ = ~len & 0xff; *o++ = (~len >> 8) & 0xff;
        memcpy(o, raw + off, len);
        for (size_t i = 0; i < len; i++) { a = (a + raw[off + i]) % 65521; b = (b + a) % 65521; }
        o += len; off += len;
    }
    be32(o, (b << 16) | a); o += 4;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); free(raw); free(z); return; }
    fwrite("\x89PNG\r\n\x1a\n", 1, 8, f);
    u8 ihdr[13]; be32(ihdr, w); be32(ihdr + 4, h); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    chunk(f, "IHDR", ihdr, 13);
    chunk(f, "IDAT", z, (u32)(o - z));
    chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(raw); free(z);
}

/* ---- live control: commands from a FIFO, one per line ----
 *   key 1c 9c         raw set-1 scancode bytes
 *   press enter|tab|space|esc|up|down|left|right|bksp|alt+X|shift+tab|f1..f10
 *   type some text    ASCII as make/break pairs
 *   mouse DX DY BTN   one PS/2 packet
 *   shot FILE         screenshot now
 *   status            print instruction count and rip
 *   stop              end the run
 * Input bytes are fed to the keyboard controller one per run slice (1 ms guest). */
static u8 in_q[4096]; static u32 in_head, in_tail;
static void q1(u8 b) { if (in_tail - in_head < sizeof in_q) in_q[in_tail++ % sizeof in_q] = b; }
static void tap(u8 sc, int shift) { if (shift) q1(0x2a); q1(sc); q1(sc | 0x80); if (shift) q1(0xaa); }
static const char *k_lo = "\x1b" "1234567890-=\b\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./";
static const char *k_hi = "\x1b" "!@#$%^&*()_+\b\tQWERTYUIOP{}\n\0ASDFGHJKL:\"~\0|ZXCVBNM<>?";
static void type_char(char ch) {
    if (ch == ' ') { tap(0x39, 0); return; }
    for (int i = 0; i < 0x35; i++) {
        if (k_lo[i] == ch && ch) { tap((u8)(i + 1), 0); return; }
        if (k_hi[i] == ch && ch) { tap((u8)(i + 1), 1); return; }
    }
}
static int named_key(const char *n) {
    static const struct { const char *n; int sc; } t[] = {
        {"enter",0x1c},{"tab",0x0f},{"space",0x39},{"esc",0x01},{"bksp",0x0e},
        {"up",0xe048},{"down",0xe050},{"left",0xe04b},{"right",0xe04d},{"home",0xe047},{"end",0xe04f},
        {"pgup",0xe049},{"pgdn",0xe051},{"del",0xe053},
        {"f1",0x3b},{"f2",0x3c},{"f3",0x3d},{"f4",0x3e},{"f5",0x3f},{"f6",0x40},{"f7",0x41},{"f8",0x42},{"f9",0x43},{"f10",0x44},
    };
    for (unsigned i = 0; i < sizeof t / sizeof t[0]; i++) if (!strcmp(n, t[i].n)) return t[i].sc;
    if (n[0] && !n[1]) for (int i = 0; i < 0x35; i++) if (k_lo[i] == n[0]) return i + 1;
    return -1;
}
static void press(const char *n) {
    int mod = 0;   /* 0x38 alt, 0x2a shift, 0x1d ctrl */
    for (;;) {
        if (!strncmp(n, "alt+", 4)) { mod = 0x38; n += 4; }
        else if (!strncmp(n, "shift+", 6)) { mod = 0x2a; n += 6; }
        else if (!strncmp(n, "ctrl+", 5)) { mod = 0x1d; n += 5; }
        else break;
    }
    int sc = named_key(n);
    if (sc < 0) { fprintf(stderr, "[w32] ctl: unknown key %s\n", n); return; }
    if (mod) q1((u8)mod);
    if (sc > 0xff) q1(0xe0);
    q1((u8)sc);
    if (sc > 0xff) q1(0xe0);
    q1((u8)(sc | 0x80));
    if (mod) q1((u8)(mod | 0x80));
}
/* ---- live profile: "prof N" samples rip at the next N slice ends, grouped by PE image ---- */
typedef struct { u64 base, end; u32 hits; char name[40]; } ProfImg;
static ProfImg prof_img[96]; static int prof_nimg;
static u32 prof_left, prof_total, prof_user, prof_unknown;
static u64 peek(Cpu *c, u64 va, int *ok) { u64 cpu_peek64(Cpu *, u64, int *); return cpu_peek64(c, va, ok); }
static void prof_name(Cpu *c, u64 base, char *out, int cap) {
    int ok; out[0] = 0;
    u32 lfa = (u32)peek(c, base + 0x3c, &ok); if (!ok) return;
    u32 exp = (u32)peek(c, base + lfa + 24 + 112, &ok); if (!ok || !exp) { snprintf(out, (size_t)cap, "?"); return; }
    u32 nrva = (u32)peek(c, base + exp + 12, &ok); if (!ok) return;
    for (int i = 0; i < cap - 1; i++) { u64 v = peek(c, base + nrva + (u64)i, &ok); char ch = ok ? (char)v : 0; out[i] = ch; out[i + 1] = 0; if (!ch) break; }
}
static void prof_sample(Machine *m) {
    u64 cpu_image_base(Cpu *, u64);
    u64 r = m->cpu.rip;
    prof_total++;
    if (!(r >> 63)) prof_user++;
    for (int i = 0; i < prof_nimg; i++) if (r >= prof_img[i].base && r < prof_img[i].end) { prof_img[i].hits++; return; }
    u64 base = cpu_image_base(&m->cpu, r);
    int ok;
    u32 lfa = base ? (u32)peek(&m->cpu, base + 0x3c, &ok) : 0;
    u32 size = base ? (u32)peek(&m->cpu, base + lfa + 24 + 56, &ok) : 0;
    if (!base || r >= base + size || prof_nimg >= 96) { prof_unknown++; return; }
    ProfImg *e = &prof_img[prof_nimg++];
    e->base = base; e->end = base + size; e->hits = 1;
    prof_name(&m->cpu, base, e->name, (int)sizeof e->name);
}
static void prof_report(void) {
    fprintf(stderr, "[w32] prof: %u samples, %u%% user mode, %u outside any PE image (JIT code / heap)\n",
            prof_total, prof_total ? prof_user * 100 / prof_total : 0, prof_unknown);
    for (int k = 0; k < 15; k++) {
        int best = -1;
        for (int i = 0; i < prof_nimg; i++) if (prof_img[i].hits && (best < 0 || prof_img[i].hits > prof_img[best].hits)) best = i;
        if (best < 0) break;
        fprintf(stderr, "[w32] prof: %5.1f%%  %-28s %llx\n", prof_img[best].hits * 100.0 / prof_total, prof_img[best].name,
                (unsigned long long)prof_img[best].base);
        prof_img[best].hits = 0;
    }
}

static int ctl_fd = -1;
static char ctl_buf[1024]; static u32 ctl_len;
static int ctl_poll(Machine *m, u8 *fb) {
    if (in_head != in_tail) machine_key(m, in_q[in_head++ % sizeof in_q]);
    if (ctl_fd < 0) return 0;
    ssize_t r = read(ctl_fd, ctl_buf + ctl_len, sizeof ctl_buf - 1 - ctl_len);
    if (r <= 0) return 0;
    ctl_len += (u32)r;
    int stop = 0;
    char *nl;
    ctl_buf[ctl_len] = 0;
    while ((nl = strchr(ctl_buf, '\n'))) {
        *nl = 0;
        char *l = ctl_buf;
        fprintf(stderr, "[w32] ctl @%llu: %s\n", (unsigned long long)m->cpu.icount, l);
        if (!strncmp(l, "key ", 4)) { char *p = l + 4, *e; for (;;) { long v = strtol(p, &e, 16); if (e == p) break; q1((u8)v); p = e; } }
        else if (!strncmp(l, "press ", 6)) { char *p = l + 6, *tok; while ((tok = strtok(p, " "))) { press(tok); p = NULL; } }
        else if (!strncmp(l, "type ", 5)) { for (char *p = l + 5; *p; p++) type_char(*p); }
        else if (!strncmp(l, "mouse ", 6)) { int dx = 0, dy = 0, b = 0; sscanf(l + 6, "%d %d %d", &dx, &dy, &b); machine_mouse(m, dx, dy, b); }
        else if (!strncmp(l, "shot ", 5)) { shot_bpp = m->fb_bpp; write_png(l + 5, fb + m->fb_start, m->fb_width, m->fb_height, m->fb_pitch); }
        else if (!strcmp(l, "status")) fprintf(stderr, "[w32] status: %llu insns, guest %.3fs, rip %llx\n",
            (unsigned long long)m->cpu.icount, m->ns / 1e9, (unsigned long long)m->cpu.rip);
        else if (!strcmp(l, "jit")) { if (m->jit) { void jit_stats(Machine *, char *, u32); char b[4]; jit_stats(m, b, sizeof b); } }
        else if (!strcmp(l, "vga")) fprintf(stderr, "[w32] fb: %ux%ux%u pitch %u start %u\n", m->fb_width, m->fb_height, m->fb_bpp, m->fb_pitch, m->fb_start);
        else if (!strncmp(l, "prof", 4)) { prof_left = (u32)(atoi(l + 4) > 0 ? atoi(l + 4) : 3000); prof_total = prof_user = prof_unknown = 0; prof_nimg = 0; }
        else if (!strcmp(l, "stop")) stop = 1;
        u32 used = (u32)(nl + 1 - ctl_buf);
        memmove(ctl_buf, nl + 1, ctl_len - used + 1);
        ctl_len -= used;
    }
    return stop;
}

static u64 state_hash(Machine *m, const u8 *ram, u64 ram_size, const u8 *fb, u32 fb_bytes) {
    u64 h = 1469598103934665603ull;
    for (u64 i = 0; i < ram_size; i += 8) { u64 v; memcpy(&v, ram + i, 8); if (v) { h ^= v + i; h *= 1099511628211ull; } }
    for (u32 i = 0; i < fb_bytes; i++) { h ^= fb[i]; h *= 1099511628211ull; }
    for (int k = 0; k < m->ncpu; k++) {
        const Cpu *c = m->cpus[k];
        for (int i = 0; i < 16; i++) { h ^= c->r[i]; h *= 1099511628211ull; }
        h ^= c->rip; h *= 1099511628211ull;
    }
    return h;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

/* W32_HOSTPROF=1: sample the host PC every 0.5 ms of CPU time. Samples in the
 * JIT code cache count as "jit"; others are written to hostprof.txt as offsets
 * from main (resolve with nm). */
#include <signal.h>
#include <ucontext.h>
#include <sys/time.h>
#define HP_N 65536
static uintptr_t hp_pc[HP_N];
static u32 hp_cnt[HP_N];
static u32 hp_jit, hp_total, hp_lost;
static uintptr_t hp_jit_lo, hp_jit_hi;
static Machine *hp_m;
static uintptr_t hpj_st[4096]; static u64 hpj_rip[4096]; static u32 hpj_n[4096], hpj_cnt[4096];
static void hp_handler(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si;
    ucontext_t *uc = uc_;
#if defined(__i386__)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_EIP];
#else
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
#endif
    hp_total++;
    if (pc >= hp_jit_lo && pc < hp_jit_hi) {
        hp_jit++;
        u64 rip; uintptr_t st; u32 n;
        int jit_pc_block(Machine *, uintptr_t, u64 *, uintptr_t *, u32 *);
        { void jit_prof_sample(Machine *, uintptr_t); if (hp_m) jit_prof_sample(hp_m, pc); }
        if (hp_m && jit_pc_block(hp_m, pc, &rip, &st, &n)) {
            u32 h = (u32)(st * 2654435761u) >> 20;
            for (u32 k = 0; k < 64; k++, h = (h + 1) & 4095) {
                if (hpj_st[h] == st) { hpj_cnt[h]++; break; }
                if (!hpj_st[h]) { hpj_st[h] = st; hpj_rip[h] = rip; hpj_n[h] = n; hpj_cnt[h] = 1; break; }
            }
        }
        return;
    }
    u32 h = (u32)(pc * 2654435761u) >> 16;
    for (u32 k = 0; k < 64; k++, h = (h + 1) & (HP_N - 1)) {
        if (hp_pc[h] == pc) { hp_cnt[h]++; return; }
        if (!hp_pc[h]) { hp_pc[h] = pc; hp_cnt[h] = 1; return; }
    }
    hp_lost++;
}
void jit_dump_bytes(const u8 *p, u32 n, u32 key);
void jit_dump_bytes(const u8 *p, u32 n, u32 key) { char fn[64]; snprintf(fn, sizeof fn, "key%x.bin", key); FILE *f = fopen(fn, "wb"); if (f) { fwrite(p, 1, n, f); fclose(f); } }
void *jit_prof_alloc(u32 n);
void *jit_prof_alloc(u32 n) { return malloc(n); }
static void hp_start(void) {
    extern Machine *hp_machine_ptr; hp_m = hp_machine_ptr;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = hp_handler; sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGPROF, &sa, 0);
    struct itimerval it = { { 0, 500 }, { 0, 500 } };
    setitimer(ITIMER_PROF, &it, 0);
}
/* W32_PERF=<icount>: hardware counters (P-core PMU) from that instruction count on */
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
static int pf_fd[24], pf_n, pf_group = -1;
static const char *pf_name[24];
static int pf_open(u32 type, u64 config, const char *name, int group) {
    struct perf_event_attr a; memset(&a, 0, sizeof a);
    a.size = sizeof a; a.type = type; a.config = config;
    a.disabled = group < 0; a.exclude_kernel = 1; a.exclude_hv = 1;
    if (group < 0) a.read_format = PERF_FORMAT_GROUP;
    int fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, group, 0);
    if (fd < 0) { fprintf(stderr, "[w32] perf: %s unavailable\n", name); return -1; }
    return fd;
}
static void pf_start(void) {
    u32 core = 4;   /* cpu_core PMU type (hybrid) */
    FILE *f = fopen("/sys/bus/event_source/devices/cpu_core/type", "r"); if (f) { if (fscanf(f, "%u", &core) != 1) core = 4; fclose(f); }
    /* group 1: top-down slots */
    static const struct { u64 cfg; const char *nm; } td[] = {
        { 0x400, "slots" }, { 0x8000, "retiring" }, { 0x8100, "bad-spec" }, { 0x8200, "fe-bound" }, { 0x8300, "be-bound" },
        { 0x8400, "heavy-ops" }, { 0x8500, "br-mispredict" }, { 0x8600, "fetch-lat" }, { 0x8700, "mem-bound" } };
    pf_group = pf_open(core, td[0].cfg, td[0].nm, -1);
    if (pf_group >= 0) for (u32 i = 1; i < 9; i++) pf_open(core, td[i].cfg, td[i].nm, pf_group);
    /* group 2: plain counters */
    u64 ext = (u64)core << 32;
    static const struct { u32 type; u64 cfg; const char *nm; } ev_all[] = {
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles" },
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions" },
        { 0xffffffffu, 0x20c4, "taken-branches" },       /* BR_INST_RETIRED.NEAR_TAKEN */
        { 0xffffffffu, 0x0160, "baclears" },             /* BACLEARS.ANY (Golden Cove 0x60): front-end resteers */
        { 0xffffffffu, 0x0480, "icache-stall-cyc" },     /* ICACHE_DATA.STALLS */
        { 0xffffffffu, 0xe424, "L2-code-miss" },         /* L2_RQSTS.CODE_RD_MISS */
        { PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch-misses" },
        { 0xffffffffu, 0x0879, "dsb-uops" },             /* IDQ.DSB_UOPS */
        { 0xffffffffu, 0x0479, "mite-uops" } };          /* IDQ.MITE_UOPS */
    const char *sel = getenv("W32_PERFSET");
    u32 nall = sizeof ev_all / sizeof ev_all[0];
    struct { u32 type; u64 cfg; const char *nm; } ev[8];
    u32 nev = 0;
    for (u32 i = 0; i < nall; i++)
        if (!sel || i < 2 || strchr(sel, '0' + (int)i)) { ev[nev].type = ev_all[i].type; ev[nev].cfg = ev_all[i].cfg; ev[nev].nm = ev_all[i].nm; nev++; }
        /* L2_RQSTS.CODE_RD_HIT */
    for (u32 i = 0; i < nev; i++) {
        struct perf_event_attr a; memset(&a, 0, sizeof a);
        a.size = sizeof a; a.exclude_kernel = 1; a.exclude_hv = 1;
        if (ev[i].type == 0xffffffffu) { a.type = core; a.config = ev[i].cfg; }   /* raw P-core event */
        else { a.type = ev[i].type; a.config = ev[i].cfg | ext; }
        int fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
        if (fd < 0) { fprintf(stderr, "[w32] perf: %s unavailable\n", ev[i].nm); continue; }
        pf_fd[pf_n] = fd; pf_name[pf_n++] = ev[i].nm;
    }
    if (pf_group >= 0) ioctl(pf_group, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}
static void pf_report(void) {
    if (pf_group >= 0) {
        u64 buf[16]; memset(buf, 0, sizeof buf);
        if (read(pf_group, buf, sizeof buf) > 0) {
            static const char *nm[] = { "slots", "retiring", "bad-spec", "fe-bound", "be-bound", "heavy-ops", "br-mispredict", "fetch-lat", "mem-bound" };
            u64 n = buf[0], slots = buf[1];
            fprintf(stderr, "[w32] perf top-down (%% of slots):");
            for (u64 i = 1; i < n && i < 9; i++) fprintf(stderr, " %s %.1f", nm[i], slots ? 100.0 * (double)buf[1 + i] / (double)slots : 0.0);
            fprintf(stderr, "\n");
        }
    }
    u64 v[24];
    for (int i = 0; i < pf_n; i++) { v[i] = 0; if (read(pf_fd[i], &v[i], 8) != 8) v[i] = 0; }
    for (int i = 0; i < pf_n; i++) fprintf(stderr, "[w32] perf %-14s %llu\n", pf_name[i], (unsigned long long)v[i]);
    if (pf_n >= 2 && v[0]) fprintf(stderr, "[w32] perf IPC %.2f\n", (double)v[1] / (double)v[0]);
}

int main(int argc, char **argv);
static void hp_dump(void) {
    struct itimerval it; memset(&it, 0, sizeof it); setitimer(ITIMER_PROF, &it, 0);
    FILE *f = fopen("hostprof.txt", "w");
    if (!f) return;
    fprintf(f, "total %u jit %u lost %u main %lx\n", hp_total, hp_jit, hp_lost, (unsigned long)(uintptr_t)main);
    for (u32 i = 0; i < HP_N; i++) if (hp_pc[i]) fprintf(f, "%lx %u\n", (unsigned long)(hp_pc[i] - (uintptr_t)main), hp_cnt[i]);
    fclose(f);
    /* hottest generated blocks: guest rip, insn count, and their host code */
    f = fopen("hostprof_jit.txt", "w");
    if (f) {
        for (int r = 0; r < 40; r++) {
            u32 best = 0;
            for (u32 i = 0; i < 4096; i++) if (hpj_cnt[i] > hpj_cnt[best]) best = i;
            if (!hpj_cnt[best]) break;
            fprintf(f, "%u %llx %u %lx\n", hpj_cnt[best], (unsigned long long)hpj_rip[best], hpj_n[best], (unsigned long)hpj_st[best]);
            char fn[64]; snprintf(fn, sizeof fn, "jitblk%02d.bin", r);
            FILE *g = fopen(fn, "wb"); if (g) { fwrite((void *)hpj_st[best], 1, 4096, g); fclose(g); }
            hpj_cnt[best] = 0;
        }
        fclose(f);
    }
    fprintf(stderr, "[w32] host profile: %u samples, %u%% in JIT code -> hostprof.txt\n", hp_total, hp_total ? hp_jit * 100 / hp_total : 0);
}

Machine *hp_machine_ptr;
int main(int argc, char **argv) {
    const char *cmdline = "mode=live", *shot = NULL, *kernel = NULL, *iso = NULL, *bios = NULL, *vgabios = NULL;
    const char *cdrom = NULL, *dump_ram = NULL, *hda = NULL, *ctl = NULL;
    int snapshot = 0;
    int entry = -1, want_hash = 0, no_jit = 0, want_text = 0;
    u64 ram_mb = 512, insns = ~0ull, shot_every = 0, hash_every = 0, hash_from = 0;
    double seconds = 0;
    BootModuleSrc mods[8];
    int nmods = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cmdline") && i + 1 < argc) cmdline = argv[++i];
        else if (!strcmp(argv[i], "--ram") && i + 1 < argc) ram_mb = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--insns") && i + 1 < argc) insns = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--shot") && i + 1 < argc) shot = argv[++i];
        else if (!strcmp(argv[i], "--shot-every") && i + 1 < argc) shot_every = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--iso") && i + 1 < argc) iso = argv[++i];
        else if (!strcmp(argv[i], "--hash")) want_hash = 1;
        else if (!strcmp(argv[i], "--bios") && i + 1 < argc) bios = argv[++i];
        else if (!strcmp(argv[i], "--vgabios") && i + 1 < argc) vgabios = argv[++i];
        else if (!strcmp(argv[i], "--cdrom") && i + 1 < argc) cdrom = argv[++i];
        else if (!strcmp(argv[i], "--hda") && i + 1 < argc) hda = argv[++i];
        else if (!strcmp(argv[i], "--snapshot")) snapshot = 1;
        else if (!strcmp(argv[i], "--hash-every") && i + 1 < argc) hash_every = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--hash-from") && i + 1 < argc) hash_from = strtoull(argv[++i], 0, 0);
        else if (!strcmp(argv[i], "--ctl") && i + 1 < argc) ctl = argv[++i];
        else if (!strcmp(argv[i], "--trace-bios")) m_trace_bios = 1;
        else if (!strcmp(argv[i], "--trace-ide")) m_trace_bios = 2;
        else if (!strcmp(argv[i], "--events")) want_events = 1;
        else if (!strcmp(argv[i], "--guest-ghz") && i + 1 < argc) guest_ghz = atof(argv[++i]);
        else if (!strcmp(argv[i], "--break") && i + 1 < argc) break_rip = strtoull(argv[++i], 0, 16);
        else if (!strcmp(argv[i], "--break-max") && i + 1 < argc) break_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--peek") && i + 1 < argc) peek_va = strtoull(argv[++i], 0, 16);
        else if (!strcmp(argv[i], "--dump-ram") && i + 1 < argc) dump_ram = argv[++i];
        else if (!strcmp(argv[i], "--no-jit")) no_jit = 1;
        else if (!strcmp(argv[i], "--text")) want_text = 1;
        else if (!strcmp(argv[i], "--entry") && i + 1 < argc) entry = atoi(argv[++i]);
        else if (!kernel) kernel = argv[i];
        else if (nmods < 8) {
            char *eq = strchr(argv[i], '=');
            if (!eq) { fprintf(stderr, "module must be name=path: %s\n", argv[i]); return 2; }
            *eq = 0;
            mods[nmods].name = argv[i];
            mods[nmods].data = read_file(eq + 1, &mods[nmods].size);
            nmods++;
        }
    }
    if (want_hash || hash_every) fixed_wall = 1767225600ull;   /* 2026-01-01: deterministic RTC */
    if (!kernel && !iso && !bios) { fprintf(stderr, "usage: %s [options] kernel.elf [name=path ...]\n", argv[0]); return 2; }

    u64 ram_size = ram_mb << 20;
    u8 *ram = mmap(NULL, ram_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ram != MAP_FAILED && !getenv("W32_NOHUGE")) madvise(ram, ram_size, MADV_HUGEPAGE);   /* fewer TLB misses (defrag=madvise hosts) */
    if (ram == MAP_FAILED) { perror("mmap"); return 1; }
    const u32 fw = 1280, fh = 800, pitch = fw * 4;
    u8 *fb = calloc(1, 16u << 20);      /* 16 MiB: also the VRAM of the emulated VGA on firmware boots */

    static Machine m;
    hp_machine_ptr = &m;
    Host host = { h_serial, h_debug, h_log, h_wall, h_disk_read, h_disk_write, NULL };
    machine_init(&m, ram, ram_size, host);
    machine_set_framebuffer(&m, fb, 0xfd000000ull, fw, fh, pitch);
    if (guest_ghz > 0) { m.ps_per_insn = (u32)(1000.0 / guest_ghz + 0.5); if (!m.ps_per_insn) m.ps_per_insn = 1;
                         fprintf(stderr, "[w32] guest clock: %u ps per instruction\n", m.ps_per_insn); }
    if (!no_jit) {
        /* big code cache: Windows' working set overflows 64 MiB and thrashes (flush + retranslate) */
        u32 jit_size = getenv("W32_JITMB") ? (u32)atoi(getenv("W32_JITMB")) << 20 : 1536u << 20;
        u8 *jm = MAP_FAILED;
        while (jit_size >= (64u << 20) &&
               (jm = mmap(NULL, jit_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED)
            jit_size >>= 1;
        { extern int jit_prof_on; extern u32 jit_off; if (getenv("W32_JITPROF")) jit_prof_on = 1; if (getenv("W32_JITOFF")) jit_off = (u32)strtoul(getenv("W32_JITOFF"), 0, 0); { extern u32 jit_off2; if (getenv("W32_JITOFF2")) jit_off2 = (u32)strtoul(getenv("W32_JITOFF2"), 0, 0); } { extern u32 jit_hot; if (getenv("W32_JITHOT")) jit_hot = (u32)strtoul(getenv("W32_JITHOT"), 0, 0); } extern u64 jit_dump_rip; if (getenv("W32_JITDUMP")) jit_dump_rip = strtoull(getenv("W32_JITDUMP"), 0, 16); }
        if (jm != MAP_FAILED && !getenv("W32_NOHUGE")) madvise(jm, jit_size, MADV_HUGEPAGE);
        if (jm != MAP_FAILED && getenv("W32_HOSTPROF")) { hp_jit_lo = (uintptr_t)jm; hp_jit_hi = (uintptr_t)jm + jit_size; }
        if (jm != MAP_FAILED && jit_init(&m, jm, jit_size) == 0) fprintf(stderr, "[w32] JIT enabled (%u MiB)\n", jit_size >> 20);
    }
    if (bios) {
        u32 bsize, vsize = 0;
        u8 *b = read_file(bios, &bsize);
        u8 *v = vgabios ? read_file(vgabios, &vsize) : NULL;
        if (getenv("W32_SMBIOS_CORES")) m.smbios_cores = (u32)atoi(getenv("W32_SMBIOS_CORES"));
        if (getenv("W32_CPUS")) machine_set_cpus(&m, atoi(getenv("W32_CPUS")));
        if (getenv("W32_QUANTUM")) m.smp_quantum = (u32)atoi(getenv("W32_QUANTUM"));
        machine_bios_reset(&m, b, bsize, v, vsize);
        machine_set_vram_size(&m, 16u << 20);
        m.trace_bios = m_trace_bios;
        m.break_rip = break_rip;
        m.break_max = break_max;
        if (cdrom) {
            disk_fd[2] = open(cdrom, O_RDONLY);
            if (disk_fd[2] < 0) { perror(cdrom); return 1; }
            struct stat st; fstat(disk_fd[2], &st);
            ide_attach(&m, 2, 1, (u64)st.st_size);
        }
        if (hda) {
            disk_fd[0] = open(hda, snapshot ? O_RDONLY : O_RDWR);
            if (disk_fd[0] < 0) { perror(hda); return 1; }
            struct stat st; fstat(disk_fd[0], &st);
            if (snapshot) {
                char dir[4096]; snprintf(dir, sizeof dir, "%s", hda);
                char *sl = strrchr(dir, '/'); if (sl) *sl = 0; else snprintf(dir, sizeof dir, ".");
                ovl_fd = open(dir, O_TMPFILE | O_RDWR, 0600);
                ovl_map = calloc((size_t)(st.st_size >> 12) + 1, 1);
                if (ovl_fd < 0 || !ovl_map) { perror("snapshot overlay"); return 1; }
            }
            ide_attach(&m, 0, 0, (u64)st.st_size);
        }
    } else if (iso) {
        u32 isize;
        u8 *img = read_file(iso, &isize);
        if (iso_boot(&m, img, isize, entry, "")) { fprintf(stderr, "boot failed\n"); return 1; }
    } else {
        u32 ksize;
        u8 *k = read_file(kernel, &ksize);
        if (mb2_boot(&m, k, ksize, cmdline, mods, nmods)) { fprintf(stderr, "boot failed\n"); return 1; }
    }

    if (ctl) {
        ctl_fd = open(ctl, O_RDWR | O_NONBLOCK);   /* RDWR: no EOF when writers come and go */
        if (ctl_fd < 0) { perror(ctl); return 1; }
    }
    double t0 = now();
    u64 hp_from = getenv("W32_HOSTPROF") ? strtoull(getenv("W32_HOSTPROF"), 0, 0) : 0;
    int hp_on = 0;
    u64 pf_from = getenv("W32_PERF") ? strtoull(getenv("W32_PERF"), 0, 0) : 0;
    int pf_on = 0;
    if (getenv("W32_HOSTPROF") && !hp_from) { hp_start(); hp_on = 1; }
    u64 next_shot = shot_every;
    u64 next_hash = hash_from + hash_every;
    u64 rip_ring[32]; int rip_n = 0;
    for (;;) {
        if (m.cpu.fatal == 3) { fprintf(stderr, "[w32] guest reset: restarting the emulated PC\n"); machine_reboot(&m); }
        if (m.cpu.fatal || m.cpu.icount >= insns) break;
        u64 slice = insns - m.cpu.icount < 1000000 ? insns - m.cpu.icount : 1000000;
        if (hash_every && next_hash - m.cpu.icount < slice) slice = next_hash - m.cpu.icount;
        { static u64 tick, next_tick; static int ti; if (!ti) { ti = 1; tick = getenv("W32_TICK") ? strtoull(getenv("W32_TICK"), 0, 0) : 0; next_tick = tick; }
          if (tick && m.cpu.icount >= next_tick) { fprintf(stderr, "[w32] tick %llu %.4f\n", (unsigned long long)m.cpu.icount, now() - t0); next_tick += tick; }
          if (tick && next_tick - m.cpu.icount < slice) slice = next_tick - m.cpu.icount; }
        machine_run(&m, slice);
        if (pf_from && !pf_on && m.cpu.icount >= pf_from) { pf_start(); pf_on = 1; }
        if (hp_from && !hp_on && m.cpu.icount >= hp_from) {
            extern int jit_prof_on; extern void jit_prof_reset(void);
            hp_start(); hp_on = 1; if (jit_prof_on) jit_prof_reset();
        }
        if (hash_every && m.cpu.icount >= next_hash) {
            fprintf(stderr, "[w32] hash @%llu %016llx rip %llx  t=%.1fs\n", (unsigned long long)m.cpu.icount,
                    (unsigned long long)state_hash(&m, ram, ram_size, fb, pitch * fh), (unsigned long long)m.cpu.rip, now() - t0);
            next_hash += hash_every;
        }
        if (ctl_poll(&m, fb)) break;
        rip_ring[rip_n++ & 31] = m.cpu.rip;
        if (prof_left && !--prof_left) prof_report(); else if (prof_left) prof_sample(&m);
        if (shot && shot_every && m.cpu.icount >= next_shot) { shot_bpp = m.fb_bpp; write_png(shot, fb + m.fb_start, m.fb_width, m.fb_height, m.fb_pitch); next_shot += shot_every; }
        if (seconds > 0 && now() - t0 > seconds) break;
    }
    double dt = now() - t0;
    fflush(stdout);
    fprintf(stderr, "[w32] stopped: %llu instructions in %.2fs (%.1f MIPS), guest time %.3fs, rip %llx, fatal %d\n",
            (unsigned long long)m.cpu.icount, dt, m.cpu.icount / dt / 1e6, m.ns / 1e9,
            (unsigned long long)m.cpu.rip, m.cpu.fatal);
    if (want_events) { void cpu_dump_events(Cpu *); cpu_dump_events(&m.cpu);
        if (m.chipset) { void apic_dump(Machine *); void chipset_dump(Machine *); apic_dump(&m); chipset_dump(&m); }
        u64 cpu_image_base(Cpu *, u64);
        for (int i = 0; i < 32 && i < rip_n; i++) {
            u64 r = rip_ring[(rip_n - 1 - i) & 31], base = (r >> 63) ? cpu_image_base(&m.cpu, r) : 0;
            fprintf(stderr, "[w32] rip sample %2d: %llx  image %llx + %llx\n", i, (unsigned long long)r,
                    (unsigned long long)base, (unsigned long long)(r - base));
        }
        fprintf(stderr, "[w32] cpl %d rip %llx cr0 %llx cr3 %llx cr4 %llx efer %llx rflags %llx\n", m.cpu.cpl,
                (unsigned long long)m.cpu.rip, (unsigned long long)m.cpu.cr0, (unsigned long long)m.cpu.cr3,
                (unsigned long long)m.cpu.cr4, (unsigned long long)m.cpu.efer, (unsigned long long)m.cpu.rflags); }
    if (peek_va) {
        u64 cpu_peek64(Cpu *, u64, int *);
        for (int i = 0; i < 6; i++) { int ok; u64 v = cpu_peek64(&m.cpu, peek_va + 8 * (u64)i, &ok);
            fprintf(stderr, "[w32] peek %llx: %016llx%s\n", (unsigned long long)(peek_va + 8 * (u64)i), (unsigned long long)v, ok ? "" : " (unmapped)"); }
    }
    if (getenv("W32_HOSTPROF")) hp_dump();
    if (pf_on) pf_report();
    if (m.jit) { void jit_stats(Machine *, char *, u32); char b[4]; jit_stats(&m, b, sizeof b); }
    if (shot) { shot_bpp = m.fb_bpp; write_png(shot, fb + m.fb_start, m.fb_width, m.fb_height, m.fb_pitch); }
    if (dump_ram) {
        FILE *f = fopen(dump_ram, "wb");
        if (f) { for (u64 o = 0; o < ram_size; o += 1u << 24) fwrite(ram + o, 1, ram_size - o < (1u << 24) ? (size_t)(ram_size - o) : (1u << 24), f); fclose(f); }
    }
    if (want_text) {
        fprintf(stderr, "+--------------------------------------------------------------------------------+\n");
        for (int y = 0; y < 25; y++) {
            fputc('|', stderr);
            for (int x = 0; x < 80; x++) { u8 ch = ram[0xb8000 + 2 * (y * 80 + x)]; fputc(ch >= 32 && ch < 127 ? ch : ' ', stderr); }
            fputs("|\n", stderr);
        }
        fprintf(stderr, "+--------------------------------------------------------------------------------+\n");
    }
    for (int k = 1; k < m.ncpu; k++) fprintf(stderr, "[w32] cpu%d: rip %llx cs %x mode64 %d halted %d wait_sipi %d IF %d icount %llu\n", k,
        (unsigned long long)m.cpus[k]->rip, m.cpus[k]->seg[CS].sel, m.cpus[k]->mode64, m.cpus[k]->halted, m.cpus[k]->wait_sipi,
        (int)((m.cpus[k]->rflags >> 9) & 1), (unsigned long long)m.cpus[k]->icount);
    if (m.ncpu > 1) fprintf(stderr, "[w32] cpu0: rip %llx cs %x halted %d IF %d icount %llu\n", (unsigned long long)m.cpu.rip, m.cpu.seg[CS].sel, m.cpu.halted, (int)((m.cpu.rflags >> 9) & 1), (unsigned long long)m.cpu.icount);
    if (want_hash) fprintf(stderr, "[w32] state hash %016llx\n", (unsigned long long)state_hash(&m, ram, ram_size, fb, pitch * fh));
    return m.cpu.fatal == 1 ? 1 : 0;
}
