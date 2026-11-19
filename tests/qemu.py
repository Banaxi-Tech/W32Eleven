import subprocess
p = subprocess.run(['qemu-system-i386', '-machine', 'pc,accel=tcg',
    '-cpu', '486', '-m', '32M', '-kernel', 'build/emulator.elf',
    '-display', 'none', '-serial', 'none', '-monitor', 'none',
    '-debugcon', 'stdio', '-device', 'isa-debug-exit,iobase=0xf4,iosize=0x04',
    '-no-reboot'], capture_output=True, text=True, timeout=30)
print(p.stdout, end='')
if p.returncode != 33 or not p.stdout.endswith('PASS\n'):
    raise SystemExit(f'QEMU failed ({p.returncode}): {p.stderr}')
