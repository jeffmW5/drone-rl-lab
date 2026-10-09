/* SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  uint32_t sequence, capture_us, inference_us, received_ms;
  int32_t margin, logits[4];
  uint8_t class_id, synthetic;
} TetherObservation;

static inline uint32_t tether_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

static inline bool tether_decode(const uint8_t *data, size_t size, uint32_t now_ms,
                                 TetherObservation *out) {
  if (size != 40 || memcmp(data, "THR1", 4) || data[4] != 1 ||
      (data[5] & ~1u) || data[6] >= 4 || data[7] != 0) return false;
  TetherObservation candidate = {0};
  candidate.sequence = tether_u32(data+8);
  candidate.capture_us = tether_u32(data+12);
  candidate.inference_us = tether_u32(data+16);
  candidate.margin = (int32_t)tether_u32(data+20);
  candidate.received_ms = now_ms;
  candidate.class_id = data[6]; candidate.synthetic = data[5] & 1;
  for(unsigned i=0;i<4;i++) candidate.logits[i] = (int32_t)tether_u32(data+24+4*i);
  unsigned best=0, second;
  for(unsigned i=1;i<4;i++) if(candidate.logits[i]>candidate.logits[best]) best=i;
  second = best==0 ? 1 : 0;
  for(unsigned i=0;i<4;i++) if(i!=best && candidate.logits[i]>candidate.logits[second]) second=i;
  int64_t margin = (int64_t)candidate.logits[best]-candidate.logits[second];
  if(margin>INT32_MAX) margin=INT32_MAX;
  if(candidate.class_id!=best || candidate.margin!=(int32_t)margin) return false;
  *out=candidate;
  return true;
}

/* This checks recent RECEIPT, not sensor-exposure age across unsynchronized clocks. */
static inline bool tether_recent(const TetherObservation *observation, uint32_t now_ms) {
  return !observation->synthetic && (uint32_t)(now_ms-observation->received_ms)<200u;
}

static inline bool tether_newer(uint32_t next, uint32_t previous) {
  uint32_t difference = next - previous;
  return difference != 0 && difference < UINT32_C(0x80000000);
}
