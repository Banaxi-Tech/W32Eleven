#include "jit.h"
/* JIT translation self-tests, run by the bare-metal runtime at boot. */
void print(const char*s);
static u8 code[4096] __attribute__((aligned(16)));
static u32 rng=12345;
static u32 random32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
int jit_selftest(void){
 print("W32Eleven: protected-mode i386 JIT tests\n");
 /* Verify every register pair, aliasing, carry and borrow against independent
    unsigned 64-bit C arithmetic. Native compiler lowers it to i386 pairs. */
 for(u32 op=0;op<4;op++)for(u32 s=0;s<16;s++)for(u32 d=0;d<16;d++)for(u32 trial=0;trial<16;trial++) {
  Cpu cpu;for(int i=0;i<16;i++){cpu.r[i].lo=random32();cpu.r[i].hi=random32();}
  if(trial==0){cpu.r[s].lo=0xffffffff;cpu.r[d].lo=1;}
  Cpu before=cpu;
  unsigned long long a=((unsigned long long)cpu.r[d].hi<<32)|cpu.r[d].lo;
  unsigned long long b=((unsigned long long)cpu.r[s].hi<<32)|cpu.r[s].lo;
  unsigned long long expected=op==0?a+b:op==1?a-b:op==2?a^b:b;
  u8 ops[]={0x01,0x29,0x31,0x89};
  u8 guest[]={0x48|((s>>3)<<2)|(d>>3),ops[op],0xc0|((s&7)<<3)|(d&7),0xf4};
  u32 n;if(translate(guest,4,code,sizeof(code),&n)!=JIT_OK)return 0;
  ((void(*)(Cpu*))code)(&cpu);
  if(cpu.r[d].lo!=(u32)expected||cpu.r[d].hi!=(u32)(expected>>32))return 0;
  for(u32 i=0;i<16;i++)if(i!=d&&(cpu.r[i].lo!=before.r[i].lo||cpu.r[i].hi!=before.r[i].hi))return 0;
 }
 print("16384 arithmetic/register differential cases passed\n");
 for(u32 r=0;r<16;r++){
  u8 guest[]={0x48|(r>>3),0xb8|(r&7),0x78,0x56,0x34,0x12,0xef,0xcd,0xab,0x90,0xf4};
  Cpu cpu={0};u32 n;
  if(translate(guest,sizeof(guest),code,sizeof(code),&n)!=JIT_OK)return 0;
  ((void(*)(Cpu*))code)(&cpu);
  if(cpu.r[r].lo!=0x12345678||cpu.r[r].hi!=0x90abcdef)return 0;
  for(u32 len=0;len<sizeof(guest);len++)if(translate(guest,len,code,sizeof(code),&n)!=JIT_TRUNCATED)return 0;
  if(translate(guest,sizeof(guest),code,1,&n)!=JIT_FULL||n)return 0;
 }
 u8 bad[]={0x0f,0x0b};u32 n;
 if(translate(bad,2,code,sizeof(code),&n)!=JIT_UNSUPPORTED)return 0;
 print("Immediate, truncation, capacity and rejection tests passed\n");return 1;
}
