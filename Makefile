CC = gcc
# The bare-metal image targets i386 integer code; an x87 FPU is required.
CFLAGS = -m32 -march=i386 -O2 -std=c11 -ffreestanding -fno-pie -fno-stack-protector \
         -fno-asynchronous-unwind-tables -fno-math-errno -fno-strict-aliasing \
         -Wall -Wextra -Werror
HOSTED_CFLAGS = -O2 -g -std=c11 -fno-strict-aliasing -fno-math-errno -Wall -Wextra -Werror

EMU_SRC = $(wildcard src/emu/*.c)
EMU_OBJ = $(patsubst src/emu/%.c,build/emu/%.o,$(EMU_SRC))
EMU_HDR = $(wildcard src/emu/*.h)

ISO ?= ../KleinOS/build/build_1.2.13_pre_release_8813.iso

.PHONY: all test boot-test boot-test-486 jit-check cpufuzz win-boot hosted clean
all: build/emulator.elf build/hosted build/hosted32

build build/emu build/host:
	mkdir -p $@
build/thunk.bin: src/host/thunk.asm | build
	nasm -f bin $< -o $@

build/start.o: src/start.asm build/thunk.bin | build
	nasm -f elf32 $< -o $@
build/%.o: src/%.c src/jit.h | build
	$(CC) $(CFLAGS) -c $< -o $@
build/emu/%.o: src/emu/%.c $(EMU_HDR) | build/emu
	$(CC) $(CFLAGS) -c $< -o $@
build/host/%.o: src/host/%.c $(EMU_HDR) | build/host
	$(CC) $(CFLAGS) -c $< -o $@
build/emulator.elf: build/start.o build/selftest.o build/jit.o build/host/main.o build/host/rt.o build/host/ata.o build/host/ahci.o build/host/bios.o $(EMU_OBJ) src/link.ld
	ld -m elf_i386 -T src/link.ld -o $@ build/start.o build/selftest.o build/jit.o build/host/main.o build/host/rt.o build/host/ata.o build/host/ahci.o build/host/bios.o $(EMU_OBJ)

# Linux development harness (not part of the bare-metal product)
hosted: build/hosted
build/hosted: tools/hosted.c $(EMU_SRC) $(EMU_HDR) | build
	$(CC) $(HOSTED_CFLAGS) -o $@ tools/hosted.c $(EMU_SRC)
# 32-bit build: same code generator as the bare-metal image, so it runs the JIT
build/hosted32: tools/hosted.c $(EMU_SRC) $(EMU_HDR) | build
	$(CC) -m32 -mincoming-stack-boundary=2 -static-libgcc -fno-pie -no-pie $(HOSTED_CFLAGS) -o $@ tools/hosted.c $(EMU_SRC)

test: build/emulator.elf
	python3 tests/qemu.py

# Boots the ISO on the bare-metal emulator inside qemu-system-i386 (KVM host CPU).
boot-test: build/emulator.elf
	python3 tests/boot_iso.py $(ISO)
# Same on QEMU's software 486: proves the runtime and generated code are i386 + x87 only.
boot-test-486: build/emulator.elf
	python3 tests/boot_iso.py $(ISO) --tcg486

# JIT conformance: interpreter and JIT must leave identical guest state
# (RAM, framebuffer, registers) after the same number of instructions.
JIT_CHECK_INSNS ?= 400000000
jit-check: build/hosted32
	@a=$$(./build/hosted32 --no-jit --iso $(ISO) --entry 1 --insns $(JIT_CHECK_INSNS) --hash 2>&1 >/dev/null | grep 'state hash'); \
	 b=$$(./build/hosted32 --iso $(ISO) --entry 1 --insns $(JIT_CHECK_INSNS) --hash 2>&1 >/dev/null | grep 'state hash'); \
	 echo "interpreter: $$a"; echo "JIT:         $$b"; \
	 if [ "$$a" = "$$b" ]; then echo PASS; else echo FAIL; exit 1; fi

# CPU conformance: random instructions on the host CPU vs the emulator (32-bit mode)
cpufuzz: build/cpufuzz
	./build/cpufuzz 2000000
build/cpufuzz: tests/cpufuzz.c $(EMU_SRC) $(EMU_HDR) | build
	$(CC) -m32 -mincoming-stack-boundary=2 -static-libgcc -O1 -std=c11 -fno-strict-aliasing -fno-math-errno -o $@ tests/cpufuzz.c $(EMU_SRC)

# Firmware boot of the Windows ISO in the hosted harness (SeaBIOS + QEMU VGA BIOS)
WIN_ISO ?= /mnt/hdd/Windows11_Client_x64_en-us_26300_9457.iso
win-boot: build/hosted
	./build/hosted --ram 2048 --bios /usr/share/qemu/bios.bin --vgabios /usr/share/qemu/vgabios-stdvga.bin \
	    --cdrom $(WIN_ISO) --insns 25000000000 --text

clean:
	rm -rf build
