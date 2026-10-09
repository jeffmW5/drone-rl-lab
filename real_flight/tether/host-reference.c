/* SPDX-License-Identifier: MIT
 * Portable integer oracle, deliberately scalar and independent of PULP kernels.
 * Input: batches of 4096 raw bytes on stdin. Output: four int32 logits per frame.
 * Host binary protocol is little endian; intended for this x86_64 training host.
 */
#include <stdint.h>
#include <stdio.h>
#include "integer_model.h"

static uint8_t a[64*64*16], b[64*64*16];

static void convolution(const uint8_t *in, uint8_t *out, int h, int ci, int co,
                        const int8_t *weights, const int32_t *bias, unsigned shift) {
  for (int y=0; y<h; y++) for (int x=0; x<h; x++) for (int o=0; o<co; o++) {
    int32_t sum=bias[o];
    for (int c=0; c<ci; c++) for (int ky=0; ky<3; ky++) for (int kx=0; kx<3; kx++) {
      int sy=y+ky-1, sx=x+kx-1;
      if (sy>=0 && sy<h && sx>=0 && sx<h)
        sum += (int32_t)in[(sy*h+sx)*ci+c] * weights[((o*ci+c)*3+ky)*3+kx];
    }
    sum = sum > 0 ? sum >> shift : 0;
    out[(y*h+x)*co+o] = sum>255 ? 255 : sum;
  }
}

static void pool(const uint8_t *in, uint8_t *out, int h, int channels, int k, int average) {
  int oh=h/k;
  for (int y=0; y<oh; y++) for (int x=0; x<oh; x++) for (int c=0; c<channels; c++) {
    unsigned value=0;
    for (int ky=0; ky<k; ky++) for (int kx=0; kx<k; kx++) {
      unsigned pixel=in[(((y*k+ky)*h+x*k+kx)*channels)+c];
      if (average) value+=pixel; else if(pixel>value) value=pixel;
    }
    out[(y*oh+x)*channels+c] = average ? value/(k*k) : value;
  }
}

int main(void) {
  uint8_t input[4096]; int32_t output[4];
  size_t size;
  while ((size=fread(input, 1, sizeof input, stdin))!=0) {
    if (size!=sizeof input) return 2;
    convolution(input,a,64,1,8,weight_0,bias_0,shift_1);
    pool(a,b,64,8,2,0);
    convolution(b,a,32,8,16,weight_3,bias_3,shift_4);
    pool(a,b,32,16,2,0);
    convolution(b,a,16,16,16,weight_6,bias_6,shift_7);
    pool(a,b,16,16,4,1);
    for(int o=0;o<4;o++) {
      output[o]=bias_10[o];
      /* Dense weights are CHW; activation buffers are HWC. */
      for(int c=0;c<16;c++) for(int y=0;y<4;y++) for(int x=0;x<4;x++)
        output[o] += (int32_t)b[(y*4+x)*16+c]*weight_10[o*256+c*16+y*4+x];
    }
    if(fwrite(output,sizeof output,1,stdout)!=1) return 3;
  }
  return ferror(stdin) ? 4 : 0;
}
