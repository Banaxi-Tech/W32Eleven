#ifndef JIT_H
#define JIT_H
typedef unsigned char u8;
typedef unsigned int u32;
typedef struct { u32 lo, hi; } Word64;
typedef struct { Word64 r[16]; } Cpu;
typedef enum { JIT_OK, JIT_TRUNCATED, JIT_UNSUPPORTED, JIT_FULL } Status;
/* Input is one bounded long-mode arithmetic block, terminated by HLT.
 * No guest instruction is executed directly. Output uses the i386 cdecl ABI.
 * Guest architectural flags and privileged execution are not yet implemented.
 */
Status translate(const u8 *guest, u32 length, u8 *out, u32 capacity, u32 *written);
#endif
