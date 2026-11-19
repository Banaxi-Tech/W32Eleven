; Real-mode thunk for BIOS calls from the 32-bit runtime (assembled as a flat
; binary for address 0x7000 and copied there at start-up).
;
; Parameter block at 0x6F00 (all in segment 0):
;   +0  eax  +4 ebx  +8 ecx  +12 edx  +16 esi  +20 edi   (in and out)
;   +24 eflags (out)
;   +28 interrupt number (in, byte)
;   +32 GDT pointer of the 32-bit runtime (6 bytes, set by the caller)
; Selectors of the runtime's GDT: 0x08 flat code32, 0x10 flat data32,
; 0x18 code16 (base 0, 64 KiB), 0x20 data16 (base 0, 64 KiB).
; DS = ES = 0 during the call (buffers must lie in the first 64 KiB).

PARAMS equ 0x6f00          ; (DAPs and other small buffers: 0x6000..; stack 0x6100..0x6ef0)

[org 0x7000]
bits 32
entry:
    pushad
    mov [saved_esp], esp
    sidt [saved_idt]
    mov al, [PARAMS + 28]
    mov [intno], al                 ; patch the INT instruction below
    jmp 0x18:pm16

bits 16
pm16:
    mov ax, 0x20
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov fs, ax
    mov gs, ax
    mov eax, cr0
    and eax, 0xfffffffe
    mov cr0, eax
    jmp 0x0000:rm

rm:
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov sp, 0x6ef0
    lidt [rm_idt]
    mov eax, [PARAMS + 0]
    mov ebx, [PARAMS + 4]
    mov ecx, [PARAMS + 8]
    mov edx, [PARAMS + 12]
    mov esi, [PARAMS + 16]
    mov edi, [PARAMS + 20]
    sti
    db 0xcd                         ; int imm8
intno:
    db 0x13
    cli
    xor bp, bp
    mov ds, bp
    mov es, bp
    mov [PARAMS + 0], eax
    mov [PARAMS + 4], ebx
    mov [PARAMS + 8], ecx
    mov [PARAMS + 12], edx
    mov [PARAMS + 16], esi
    mov [PARAMS + 20], edi
    pushfd
    pop dword [PARAMS + 24]
    o32 lgdt [PARAMS + 32]
    mov eax, cr0
    or eax, 1
    mov cr0, eax
    jmp dword 0x08:pm32

bits 32
pm32:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov fs, ax
    mov gs, ax
    lidt [saved_idt]
    mov esp, [saved_esp]
    popad
    ret

align 4
saved_esp: dd 0
saved_idt: dw 0
           dd 0
rm_idt:    dw 0x3ff
           dd 0
