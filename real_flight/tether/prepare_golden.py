"""Prepare full-byte per-layer GVSOC comparison for the generated PULP kernels."""
from pathlib import Path
import shutil
import numpy as np
from sdk_compat import patch_bias

root = Path(__file__).resolve().parent
target = root / 'firmware/tether-golden'
target.mkdir(parents=True, exist_ok=True)
for directory in ('src', 'inc'):
    shutil.copytree(root / 'generated/tether-l2' / directory, target / directory, dirs_exist_ok=True)
for file in list((target / 'src').glob('*.c')) + list((target / 'inc').glob('*.h')):
    file.write_text(file.read_text().replace('L2_DATA', 'PI_L2').replace('L1_DATA', 'PI_L1').replace('<hal/pulp.h>', '"pmsis.h"'))
(target / 'inc/pulp.h').write_text('#pragma once\n/* Legacy DORY include on PMSIS/FreeRTOS. */\n#include "pmsis.h"\n')
dma = target / 'src/dory_dma.c'
dma.write_text('#include <math.h>\n' + dma.read_text().replace('ARCHI_MCHAN_DEMUX_ADDR', 'CL_DEMUX_DMA_ADDR').replace('ARCHI_CL_EVT_DMA0', 'CL_IRQ_DMA0'))
mchan = target / 'inc/mchan.h'
mchan.write_text(mchan.read_text().replace('eu_evt_maskWaitAndClr', 'hal_eu_evt_mask_wait_and_clr'))
patch_bias(target)
header = ['#pragma once', '#include <stdint.h>', 'extern volatile uint32_t tether_failures;']
sizes = []
for i in range(7):
    values = np.loadtxt(root / f'runs/quantized/out_layer{i}.txt', dtype=np.int64).flatten()
    data = values.astype('<i4').view(np.uint8) if i == 6 else values.astype(np.uint8)
    sizes.append(len(data))
    header.append(f'static const uint8_t expected_{i}[{len(data)}] = {{' + ','.join(map(str, data)) + '};')
header += ['static const uint8_t *expected_layers[7] = {' + ','.join(f'expected_{i}' for i in range(7)) + '};',
           'static const unsigned expected_sizes[7] = {' + ','.join(map(str, sizes)) + '};',
           'static void tether_check_layer(unsigned layer, const uint8_t *actual, unsigned size) {',
           '  if (layer>=7 || size!=expected_sizes[layer]) { tether_failures++; return; }',
           '  for(unsigned i=0;i<size;i++) if(actual[i]!=expected_layers[layer][i]) tether_failures++;',
           '}']
(target / 'inc/golden.h').write_text('\n'.join(header))
path = target / 'src/network.c'
source = path.read_text()
assert source.count('unsigned int args[5];') == 1
source = source.replace('unsigned int args[5];', 'unsigned int args[6];')
source = source.replace('if (pi_cluster_open(&cluster_dev))\n    return;',
                        'if (pi_cluster_open(&cluster_dev)) { pmsis_exit(-1); return (struct network_run_token){0}; }')
source = '#include "pmsis.h"\n#include "golden.h"\n' + source
marker = '    // Free memory'
assert source.count(marker) == 1
source = source.replace(marker, '    tether_check_layer(i, L2_output, activations_out_size[i]);\n\n' + marker)
path.write_text(source)
(target / 'src/main.c').write_text('''#include "pmsis.h"
#include "input.h"
#include "network.h"
#include <stdio.h>
volatile uint32_t tether_failures;
static void application(void *arg) {
  (void)arg;
  void *arena=pi_l2_malloc(96*1024);
  int32_t output[4]={0};
  if(!arena) { pmsis_exit(2); return; }
  network_run(arena,96*1024,output,0,1,L2_input_h);
  printf("GOLDEN_%s mismatched_bytes=%u logits=%d,%d,%d,%d\\n",
         tether_failures?"FAIL":"PASS",(unsigned)tether_failures,output[0],output[1],output[2],output[3]);
  pmsis_exit(tether_failures?1:0);
}
int main(void) { return pmsis_kickoff((void *)application); }
''')
makefile = (root / 'generated/tether-l2/Makefile').read_text()
makefile = 'PMSIS_OS = freertos\n' + makefile
shutil.copy2(root / 'generated/tether-l2/vars.mk', target / 'vars.mk')
(target / 'Makefile').write_text(makefile)
print(target)
