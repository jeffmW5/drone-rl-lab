"""Expand the exact per-layer GAP8 test to synthetic and adversarial inputs."""
from pathlib import Path
import subprocess
import sys
import numpy as np
from train import SyntheticLines

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / 'prepare_golden.py')], check=True)
target = root / 'firmware/tether-golden'
oracle = (root / 'host-reference.c').read_text().split('int main(void) {')[0]
oracle = oracle.replace('a[64*64*16], b[64*64*16]', 'a[32768], b[8192]')
oracle += '''
extern const uint8_t *tether_test_input;
extern unsigned tether_test_case;
static int32_t oracle_output[4];
static void tether_check_layer(unsigned layer, const uint8_t *actual, unsigned size) {
  const uint8_t *expected=0;
  unsigned expected_size=0;
  switch(layer) {
    case 0: convolution(tether_test_input,a,64,1,8,weight_0,bias_0,shift_1); expected=a; expected_size=32768; break;
    case 1: pool(a,b,64,8,2,0); expected=b; expected_size=8192; break;
    case 2: convolution(b,a,32,8,16,weight_3,bias_3,shift_4); expected=a; expected_size=16384; break;
    case 3: pool(a,b,32,16,2,0); expected=b; expected_size=4096; break;
    case 4: convolution(b,a,16,16,16,weight_6,bias_6,shift_7); expected=a; expected_size=4096; break;
    case 5: pool(a,b,16,16,4,1); expected=b; expected_size=256; break;
    case 6:
      for(int o=0;o<4;o++) {
        oracle_output[o]=bias_10[o];
        for(int c=0;c<16;c++) for(int y=0;y<4;y++) for(int x=0;x<4;x++)
          oracle_output[o]+=(int32_t)b[(y*4+x)*16+c]*weight_10[o*256+c*16+y*4+x];
      }
      expected=(const uint8_t *)oracle_output; expected_size=16; break;
  }
  if(!expected || size!=expected_size) { tether_failures++; return; }
  unsigned mismatches=0;
  for(unsigned i=0;i<size;i++) if(actual[i]!=expected[i]) mismatches++;
  if(mismatches) printf("CASE_FAIL case=%u layer=%u bytes=%u\\n",tether_test_case,layer,mismatches);
  tether_failures+=mismatches;
  if(tether_test_case==0) golden_static_check(layer,actual,size);
}
'''
header = target / 'inc/golden.h'
header.write_text(header.read_text().replace('tether_check_layer(', 'golden_static_check(') + '\n' + oracle)
(target / 'inc/integer_model.h').write_text((root / 'runs/quantized/integer_model.h').read_text())
samples = SyntheticLines(8, 300000)
frames = [np.round(samples[i][0].numpy()*255).astype(np.uint8).flatten() for i in range(8)]
(target / 'inc/cases.h').write_text('static const uint8_t test_frames[8][4096] = {' +
    ','.join('{' + ','.join(map(str, frame)) + '}' for frame in frames) + '};\n')
(target / 'src/main.c').write_text('''#include "pmsis.h"
#include "input.h"
#include "network.h"
#include "cases.h"
#include <stdio.h>
#include <string.h>
volatile uint32_t tether_failures;
const uint8_t *tether_test_input;
unsigned tether_test_case;
static uint8_t input[4096];
static void application(void *arg) {
  (void)arg;
  void *arena=pi_l2_malloc(96*1024);
  int32_t output[4];
  if(!arena) { pmsis_exit(2); return; }
  for(tether_test_case=0;tether_test_case<16;tether_test_case++) {
    uint32_t rng=0x31415926;
    for(unsigned i=0;i<4096;i++) {
      rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
      switch(tether_test_case) {
        case 0: input[i]=L2_input_h[i]; break;
        case 1: input[i]=0; break;
        case 2: input[i]=255; break;
        case 3: input[i]=((i/64+i%64)&1)?255:0; break;
        case 4: input[i]=i%256; break;
        case 5: input[i]=(i==0 || i==4095 || i==2080)?255:0; break;
        case 6: input[i]=127; break;
        case 7: input[i]=rng&255; break;
        default: input[i]=test_frames[tether_test_case-8][i]; break;
      }
    }
    tether_test_input=input;
    network_run(arena,96*1024,output,0,1,input);
    printf("CASE_DONE case=%u cumulative_mismatched_bytes=%u\\n",tether_test_case,(unsigned)tether_failures);
  }
  printf("GOLDEN_%s mismatched_bytes=%u cases=16 layers=112\\n",tether_failures?"FAIL":"PASS",(unsigned)tether_failures);
  pmsis_exit(tether_failures?1:0);
}
int main(void) { return pmsis_kickoff((void *)application); }
''')
network = target / 'src/network.c'
network.write_text(network.read_text().replace('#define VERBOSE 1', '/* Per-case oracle replaces static checksum reporting. */'))
print('Prepared 16 cases, exact comparisons after all 112 layer executions')
