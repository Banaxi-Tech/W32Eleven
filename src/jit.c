#include "jit.h"
typedef struct { u8 *p; u32 n, cap; int full; } Writer;
static void byte(Writer *w,u8 x) { if(w->n<w->cap)w->p[w->n++]=x;else w->full=1; }
static void dword(Writer *w,u32 x) { for(int i=0;i<4;i++)byte(w,x>>(8*i)); }
/* EDX points at guest state. All guest operands use disp32 addressing. */
static void mem(Writer *w,u8 op,u8 reg,u32 offset) {
 byte(w,op);byte(w,0x82|(reg<<3));dword(w,offset);
}
Status translate(const u8 *g,u32 len,u8 *out,u32 cap,u32 *written) {
 Writer w={out,0,cap,0};u32 p=0;*written=0;
 byte(&w,0x8b);byte(&w,0x54);byte(&w,0x24);byte(&w,4);
 while(p<len) {
  u8 rex=g[p++];
  if(rex==0xf4) { byte(&w,0xc3);if(w.full)return JIT_FULL;*written=w.n;return JIT_OK; }
  if((rex&0xf8)!=0x48 || (rex&2))return JIT_UNSUPPORTED;
  if(p==len)return JIT_TRUNCATED;
  u8 op=g[p++];
  if(op>=0xb8&&op<=0xbf) {
   if(len-p<8)return JIT_TRUNCATED;
   u32 dst=((op-0xb8)|((rex&1)<<3))*8;
   for(int h=0;h<2;h++) {
    byte(&w,0xc7);byte(&w,0x82);dword(&w,dst+4*h);
    for(int i=0;i<4;i++)byte(&w,g[p++]);
   }
  } else if(op==0x01||op==0x29||op==0x31||op==0x89) {
   if(p==len)return JIT_TRUNCATED;
   u8 m=g[p++];if((m&0xc0)!=0xc0)return JIT_UNSUPPORTED;
   u32 src=(((m>>3)&7)|((rex&4)<<1))*8;
   u32 dst=((m&7)|((rex&1)<<3))*8;
   /* Load both halves before writing: source and destination can alias. */
   mem(&w,0x8b,0,src);mem(&w,0x8b,1,src+4);
   mem(&w,op,0,dst);
   mem(&w,op==0x01?0x11:op==0x29?0x19:op,1,dst+4);
  } else return JIT_UNSUPPORTED;
  if(w.full)return JIT_FULL;
 }
 return JIT_TRUNCATED;
}
