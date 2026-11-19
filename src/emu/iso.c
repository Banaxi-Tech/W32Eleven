/* Boot a Multiboot2 OS straight from an ISO 9660 image, the way GRUB would:
 * read /boot/grub/grub.cfg, pick a menu entry, load its `multiboot2` kernel
 * and `module2` modules from the image, and hand off. Only the subset of the
 * GRUB script language that such menus use is understood (set default=,
 * menuentry blocks, multiboot2, module2); everything else is ignored. */
#include "emu.h"

static u32 le32(const u8 *p) { return (u32)p[0] | (u32)p[1] << 8 | (u32)p[2] << 16 | (u32)p[3] << 24; }

static int lower(int ch) { return ch >= 'A' && ch <= 'Z' ? ch + 32 : ch; }

static int name_eq(const u8 *a, u32 an, const char *b, u32 bn) {
    if (an != bn) return 0;
    for (u32 i = 0; i < an; i++) if (lower(a[i]) != lower((u8)b[i])) return 0;
    return 1;
}

/* find a component in a directory extent; returns 1 and the child's extent */
static int dir_lookup(const u8 *iso, u64 size, u32 lba, u32 len, const char *comp, u32 clen, u32 *olba, u32 *olen) {
    u64 start = (u64)lba * 2048, end = start + len;
    if (end > size) return 0;
    for (u64 p = start; p < end;) {
        u32 rl = iso[p];
        if (!rl) { p = (p / 2048 + 1) * 2048; continue; }
        if (p + rl > end || rl < 34) return 0;
        const u8 *r = iso + p;
        u32 nl = r[32];
        const u8 *nm = r + 33;
        int match = 0;
        /* Rock Ridge NM entries in the system use area */
        u32 su = 33 + nl + ((nl & 1) ? 0 : 1);
        u8 rr[256];
        u32 rrn = 0;
        while (su + 4 <= rl) {
            const u8 *e = r + su;
            u32 el = e[2];
            if (el < 4 || su + el > rl) break;
            if (e[0] == 'N' && e[1] == 'M' && el > 5) {
                for (u32 i = 5; i < el && rrn < sizeof rr; i++) rr[rrn++] = e[i];
            }
            su += el;
        }
        if (rrn) match = name_eq(rr, rrn, comp, clen);
        if (!match) {
            u32 n = nl;
            for (u32 i = 0; i < nl; i++) if (nm[i] == ';') { n = i; break; }
            if (n && nm[n - 1] == '.') n--;
            match = name_eq(nm, n, comp, clen);
        }
        if (match) { *olba = le32(r + 2); *olen = le32(r + 10); return 1; }
        p += rl;
    }
    return 0;
}

static int iso_find(const u8 *iso, u64 size, const char *path, const u8 **data, u32 *len) {
    if (size < 17 * 2048) return 0;
    const u8 *pvd = iso + 16 * 2048;
    if (pvd[0] != 1 || pvd[1] != 'C' || pvd[2] != 'D') return 0;
    u32 lba = le32(pvd + 156 + 2), l = le32(pvd + 156 + 10);
    while (*path) {
        while (*path == '/') path++;
        if (!*path) break;
        u32 n = 0;
        while (path[n] && path[n] != '/') n++;
        if (!dir_lookup(iso, size, lba, l, path, n, &lba, &l)) return 0;
        path += n;
    }
    if ((u64)lba * 2048 + l > size) return 0;
    *data = iso + (u64)lba * 2048;
    *len = l;
    return 1;
}

/* ---------- grub.cfg ---------- */

typedef struct {
    char kernel[128], args[256];
    char mod_path[8][128], mod_name[8][64];
    int nmods;
    char title[96];
} Entry;

static const char *skip_ws(const char *p, const char *e) { while (p < e && (*p == ' ' || *p == '\t')) p++; return p; }

/* copy one (optionally quoted) word */
static const char *word(const char *p, const char *e, char *out, u32 cap) {
    u32 n = 0;
    p = skip_ws(p, e);
    char q = 0;
    if (p < e && (*p == '"' || *p == '\'')) q = *p++;
    while (p < e && (q ? *p != q : (*p != ' ' && *p != '\t'))) { if (n + 1 < cap) out[n++] = *p; p++; }
    if (q && p < e) p++;
    out[n] = 0;
    return p;
}

static int starts(const char *p, const char *e, const char *kw) {
    u32 n = (u32)e_strlen(kw);
    if ((u32)(e - p) < n) return 0;
    for (u32 i = 0; i < n; i++) if (p[i] != kw[i]) return 0;
    return p + n == e || p[n] == ' ' || p[n] == '\t' || p[n] == '=';
}

static int parse_grub(const char *cfg, u32 len, Entry *ents, int max, int *def) {
    int n = 0, in = 0;
    *def = 0;
    const char *p = cfg, *end = cfg + len;
    while (p < end) {
        const char *le = p;
        while (le < end && *le != '\n') le++;
        const char *s = skip_ws(p, le);
        if (starts(s, le, "set") && !in) {
            const char *t = skip_ws(s + 3, le);
            if (starts(t, le, "default")) {
                t += 7;
                if (t < le && *t == '=') t++;
                if (t < le && (*t == '"' || *t == '\'')) t++;
                int v = 0;
                while (t < le && *t >= '0' && *t <= '9') v = v * 10 + (*t++ - '0');
                *def = v;
            }
        } else if (starts(s, le, "menuentry") && n < max) {
            Entry *e = &ents[n];
            e_memset(e, 0, sizeof *e);
            word(s + 9, le, e->title, sizeof e->title);
            in = 1;
        } else if (in && s < le && *s == '}') {
            in = 0;
            n++;
        } else if (in && starts(s, le, "multiboot2")) {
            Entry *e = &ents[n];
            const char *t = word(s + 10, le, e->kernel, sizeof e->kernel);
            t = skip_ws(t, le);
            u32 k = 0;
            while (t < le && *t != '\r' && k + 1 < sizeof e->args) e->args[k++] = *t++;
            while (k && (e->args[k - 1] == ' ' || e->args[k - 1] == '\t')) k--;
            e->args[k] = 0;
        } else if (in && starts(s, le, "module2")) {
            Entry *e = &ents[n];
            if (e->nmods < 8) {
                const char *t = word(s + 7, le, e->mod_path[e->nmods], sizeof e->mod_path[0]);
                word(t, le, e->mod_name[e->nmods], sizeof e->mod_name[0]);
                e->nmods++;
            }
        }
        p = le + 1;
    }
    return n;
}

int iso_boot(Machine *m, const u8 *iso, u64 size, int entry, const char *extra_args) {
    const u8 *cfg;
    u32 cfg_len;
    if (!iso_find(iso, size, "/boot/grub/grub.cfg", &cfg, &cfg_len)) {
        emu_log(m, "iso: /boot/grub/grub.cfg not found");
        return -1;
    }
    static Entry ents[16];
    int def;
    int n = parse_grub((const char *)cfg, cfg_len, ents, 16, &def);
    for (int i = 0; i < n; i++) emu_log(m, "iso: menu entry %d: \"%s\" (%s %s)", i, ents[i].title, ents[i].kernel, ents[i].args);
    if (entry < 0) entry = def;
    if (entry >= n || !ents[entry].kernel[0]) { emu_log(m, "iso: no bootable entry %d", entry); return -1; }
    Entry *e = &ents[entry];
    emu_log(m, "iso: booting entry %d \"%s\"", entry, e->title);
    const u8 *k;
    u32 klen;
    if (!iso_find(iso, size, e->kernel, &k, &klen)) { emu_log(m, "iso: %s not found", e->kernel); return -1; }
    BootModuleSrc mods[8];
    for (int i = 0; i < e->nmods; i++) {
        if (!iso_find(iso, size, e->mod_path[i], &mods[i].data, &mods[i].size)) {
            emu_log(m, "iso: %s not found", e->mod_path[i]);
            return -1;
        }
        mods[i].name = e->mod_name[i];
    }
    static char cmd[512];
    u32 cn = 0;
    for (const char *s = e->args; *s && cn + 1 < sizeof cmd; s++) cmd[cn++] = *s;
    if (extra_args && *extra_args) {
        if (cn && cn + 1 < sizeof cmd) cmd[cn++] = ' ';
        for (const char *s = extra_args; *s && cn + 1 < sizeof cmd; s++) cmd[cn++] = *s;
    }
    cmd[cn] = 0;
    return mb2_boot(m, k, klen, cmd, mods, e->nmods);
}
