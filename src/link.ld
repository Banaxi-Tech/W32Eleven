ENTRY(_start)
SECTIONS {
 . = 1M;
 .multiboot : { KEEP(*(.multiboot)) }
 .text : { *(.text*) }
 .rodata : { *(.rodata*) }
 .data : { *(.data*) }
 .bss : { *(COMMON) *(.bss*) }
 . = ALIGN(4096);
 __image_end = .;
 /DISCARD/ : { *(.eh_frame*) *(.note*) *(.comment) }
}
