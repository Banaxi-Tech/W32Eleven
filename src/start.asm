BITS 32
; Multiboot (v1) header: page-align modules, provide memory information, and ask
; the loader for a linear framebuffer (real PCs: GRUB sets a VESA mode for us).
MB_FLAGS equ 0x7
section .multiboot
align 4
dd 0x1badb002, MB_FLAGS, -(0x1badb002 + MB_FLAGS)
dd 0, 0, 0, 0, 0          ; load addresses (unused: flag 16 clear)
dd 0, 1024, 768, 32       ; linear graphics, preferred 1024x768x32
section .text
global _start
extern kernel_main
_start:
 cli
 cld
 lgdt [gdt_desc]                 ; our own GDT (the BIOS thunk needs 16-bit selectors)
 jmp 0x08:.flat
.flat:
 mov cx, 0x10
 mov ds, cx
 mov es, cx
 mov ss, cx
 mov fs, cx
 mov gs, cx
 mov esp, stack_top
 ; x87 is required (the SSE emulation computes with it): EM=0, MP=1, TS=0, NE=1
 mov ecx, cr0
 and ecx, ~0xc
 or ecx, 0x22
 mov cr0, ecx
 fninit
 push ebx
 push eax
 call kernel_main
.hang: hlt
 jmp .hang
section .data
align 8
global gdt_desc
gdt:
 dq 0
 dq 0x00cf9a000000ffff           ; 0x08 code32, flat
 dq 0x00cf92000000ffff           ; 0x10 data32, flat
 dq 0x00009a000000ffff           ; 0x18 code16, base 0, 64 KiB
 dq 0x000092000000ffff           ; 0x20 data16, base 0, 64 KiB
gdt_desc:
 dw gdt_desc - gdt - 1
 dd gdt
global thunk_blob, thunk_blob_end
thunk_blob:
 incbin "build/thunk.bin"
thunk_blob_end:
section .bss
align 16
resb 65536
stack_top:
section .note.GNU-stack noalloc noexec nowrite progbits
