/* SPDX-License-Identifier: MIT */
#include <assert.h>
#include <stdio.h>
#include "src/tether_protocol.h"

static void put32(uint8_t *p, uint32_t value) {
  for(unsigned i=0;i<4;i++) p[i]=(uint8_t)(value>>(8*i));
}
int main(void) {
  uint8_t packet[40]={0};
  TetherObservation result;
  memcpy(packet,"THR1",4); packet[4]=1; packet[6]=2;
  put32(packet+8,42); put32(packet+20,10);
  put32(packet+24,(uint32_t)-3); put32(packet+28,10);
  put32(packet+32,20); put32(packet+36,0);
  assert(tether_decode(packet,40,100,&result));
  assert(result.sequence==42 && result.logits[0]==-3 && result.class_id==2);
  assert(tether_recent(&result,299));
  assert(!tether_recent(&result,300));
  assert(!tether_decode(packet,39,100,&result));
  packet[5]=2; assert(!tether_decode(packet,40,100,&result));
  packet[5]=1; assert(tether_decode(packet,40,100,&result));
  assert(!tether_recent(&result,100));
  packet[5]=0; packet[6]=0; assert(!tether_decode(packet,40,100,&result));
  packet[6]=2; put32(packet+20,9); assert(!tether_decode(packet,40,100,&result));
  assert(tether_newer(0,UINT32_MAX));
  assert(!tether_newer(42,42));
  assert(!tether_newer(41,42));
  assert(!tether_newer(UINT32_C(0x80000000),0));
  result.synthetic=0; result.received_ms=UINT32_MAX-10;
  assert(tether_recent(&result,10));
  puts("Protocol checks passed: decoding, corruption, synthetic flag, expiry, sequence wrap");
  return 0;
}
