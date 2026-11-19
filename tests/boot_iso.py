"""End-to-end boot test: the bare-metal W32Eleven image boots a Multiboot2
x86-64 OS from an ISO (via its GRUB menu) and must reach the desktop.

usage: python3 tests/boot_iso.py IMAGE.iso [--entry N] [--tcg486] [--nojit] [--timeout S]

Host CPU for the 32-bit runtime:
  default   qemu-system-i386 with KVM (-cpu host): the runtime's own code runs
            at hardware speed, so this measures the emulator, not QEMU
  --tcg486  QEMU's software 486 model: slow, but proves that the runtime and
            all code the JIT generates use nothing newer than i386 + x87

Pass criteria (KleinOS live session):
  1. the guest's serial log reports "compositor ready" and "logged in";
  2. a later screenshot of the host display shows the desktop: the top menu
     bar is light and at least 30% of pixels are not black.
"""
import os, socket, struct, subprocess, sys, tempfile, time, zlib

args = sys.argv[1:]
iso = args[0] if args and not args[0].startswith("--") else None
entry = args[args.index("--entry") + 1] if "--entry" in args else "1"
timeout = int(args[args.index("--timeout") + 1]) if "--timeout" in args else 900
tcg = "--tcg486" in args
append = f"entry={entry}" + (" jit=0" if "--nojit" in args else "")
if not iso or not os.path.exists(iso):
    sys.exit(f"ISO not found: {iso}")
if not tcg and not os.access("/dev/kvm", os.R_OK | os.W_OK):
    sys.exit("no usable /dev/kvm: run with --tcg486")

tmp = tempfile.mkdtemp(prefix="w32boot-")
log, mon, shot = (os.path.join(tmp, n) for n in ("debugcon.log", "mon.sock", "screen.ppm"))
cpu = ["-machine", "pc,accel=tcg", "-cpu", "486"] if tcg else ["-machine", "pc,accel=kvm", "-cpu", "host"]
qemu = subprocess.Popen(["qemu-system-i386", *cpu, "-m", "1G",
    "-kernel", "build/emulator.elf", "-append", append, "-initrd", iso,
    "-vga", "std", "-display", "none", "-serial", "none",
    "-monitor", f"unix:{mon},server,nowait", "-debugcon", f"file:{log}",
    "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04", "-no-reboot"])

def text():
    try: return open(log, errors="replace").read()
    except FileNotFoundError: return ""

def screendump():
    s = socket.socket(socket.AF_UNIX); s.connect(mon); time.sleep(0.3); s.recv(65536)
    if os.path.exists(shot): os.remove(shot)
    s.send(f"screendump {shot}\n".encode())
    for _ in range(100):
        time.sleep(0.1)
        if os.path.exists(shot) and os.path.getsize(shot) > 1000: break
    time.sleep(0.5); s.close()
    d = open(shot, "rb").read()
    hdr = d.split(b"\n", 3)
    w, h = map(int, hdr[1].split())
    return w, h, hdr[3]

def fail(msg):
    qemu.kill()
    print(text()[-3000:])
    sys.exit(f"FAIL: {msg} (logs in {tmp})")

start = time.time()
try:
    while True:
        t = text()
        if "logged in" in t: break
        if qemu.poll() is not None: fail(f"QEMU exited early ({qemu.returncode})")
        if "FATAL" in t or "panic" in t.lower(): fail("emulator or guest reported a fatal error")
        if time.time() - start > timeout: fail("timed out before the guest session started")
        time.sleep(0.05)
    if "compositor ready" not in text(): fail("no compositor")
    print(f"guest session started {time.time() - start:.2f}s after QEMU launch")
    deadline = time.time() + 300
    while True:
        time.sleep(2 if not tcg else 15)
        w, h, px = screendump()
        step = 3 * 97
        lit = sum(1 for i in range(0, len(px) - 2, step) if px[i] + px[i + 1] + px[i + 2] > 60)
        frac = lit / (len(px) // step)
        bar = px[5 * w * 3: 5 * w * 3 + 3 * 400]
        bar_light = sum(bar) / len(bar) > 150
        if frac > 0.3 and bar_light: break
        if time.time() > deadline: fail(f"desktop never appeared (lit {frac:.2f}, menu bar light {bar_light})")
    raw = b"".join(b"\0" + px[y * w * 3:(y + 1) * w * 3] for y in range(h))
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b))
    png = os.path.join(tmp, "desktop.png")
    open(png, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                          + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    print(f"desktop on screen; screenshot {png}")
    print("PASS")
finally:
    if qemu.poll() is None: qemu.kill()
