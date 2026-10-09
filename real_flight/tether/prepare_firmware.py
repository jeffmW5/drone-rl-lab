"""Prepare a camera+CPX app from DORY GAP8_L2 output and pinned Bitcraze sources."""
import argparse
import json
from pathlib import Path
import shutil
from sdk_compat import patch_bias

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path.cwd())
args = p.parse_args()
root = args.root.resolve()
generated = root / 'generated/tether-l2'
target = root / 'firmware/tether-camera'
target.mkdir(parents=True, exist_ok=True)
for directory in ('src', 'inc'):
    shutil.copytree(generated / directory, target / directory, dirs_exist_ok=True)
for file in list((target / 'src').glob('*.c')) + list((target / 'inc').glob('*.h')):
    file.write_text(file.read_text().replace('L2_DATA', 'PI_L2').replace('L1_DATA', 'PI_L1').replace('<hal/pulp.h>', '"pmsis.h"'))
(target / 'inc/pulp.h').write_text('#pragma once\n/* Legacy DORY include on PMSIS/FreeRTOS. */\n#include "pmsis.h"\n')
dma = target / 'src/dory_dma.c'
dma.write_text('#include <math.h>\n' + dma.read_text().replace('ARCHI_MCHAN_DEMUX_ADDR', 'CL_DEMUX_DMA_ADDR').replace('ARCHI_CL_EVT_DMA0', 'CL_IRQ_DMA0'))
mchan = target / 'inc/mchan.h'
mchan.write_text(mchan.read_text().replace('eu_evt_maskWaitAndClr', 'hal_eu_evt_mask_wait_and_clr'))
patch_bias(target)
network_file = target / 'src/network.c'
network = network_file.read_text()
if network.count('unsigned int args[5];') != 1 or 'args[5] = (unsigned int) L2_input_h;' not in network:
    raise RuntimeError('DORY network template changed: review argument-buffer patch')
network = network.replace('unsigned int args[5];', 'unsigned int args[6];')
old = 'if (pi_cluster_open(&cluster_dev))\n    return;'
if network.count(old) != 1:
    raise RuntimeError('Review changed cluster-open error path')
network = network.replace(old, 'if (pi_cluster_open(&cluster_dev)) {\n    pmsis_exit(-1);\n    return (struct network_run_token){0};\n  }')
network = network.replace('#define VERBOSE 1', '/* Golden-frame checks are disabled for live camera inputs. */')
network = network.replace('  print_perf("Final", cycle_network_execution, 2065408);', '  /* Timing is reported in CPX perception packets. */')
network_file.write_text(network)
vendor = root / 'vendor/aideck-gap8-examples'
camera = vendor / 'examples/other/wifi-img-streamer'
for filename in ('camera_pipeline.c', 'himax_timing.c'):
    shutil.copy2(camera / filename, target / 'src' / filename)
for filename in ('camera_pipeline.h', 'himax_timing.h', 'stream_config.h'):
    shutil.copy2(camera / filename, target / 'inc' / filename)
for filename in ('com.c', 'cpx.c'):
    shutil.copy2(vendor / 'lib/cpx/src' / filename, target / 'src' / filename)
for header in (vendor / 'lib/cpx/inc').glob('*.h'):
    shutil.copy2(header, target / 'inc' / header.name)
shutil.copy2(root / 'firmware-main.c', target / 'src/main.c')
report = json.loads((root / 'runs/quantized/quantization.json').read_text())
(target / 'inc/tether_model.h').write_text('#pragma once\n#define TETHER_SYNTHETIC ' + str(int(report['synthetic'])) + '\n')
makefile = '''PMSIS_OS = freertos
io = uart
CORE ?= 8
APP = tether_camera
APP_SRCS := $(wildcard src/*.c)
APP_INC = inc
APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP_SDK=1 -DTARGET_CHIP_FAMILY_GAP8
APP_CFLAGS += -DUSE_HYPERRAM -DUSE_HYPERFLASH -DRAM_TYPE=HYPERRAM -DFLASH_TYPE=HYPERFLASH
APP_CFLAGS += -O2 -Iinc -DconfigUSE_TIMERS=1 -DINCLUDE_xTimerPendFunctionCall=1
APP_LDFLAGS += -lm -Wl,--print-memory-usage
include $(RULES_DIR)/pmsis_rules.mk
'''
(target / 'Makefile').write_text(makefile)
(target / 'MODEL.json').write_text(json.dumps(report, indent=2))
(target / 'PATCHES.md').write_text('DORY network argument array corrected from 5 to 6 entries.\nCluster-open failure terminates rather than returning an undefined token.\nBitcraze camera/CPX source licenses retained in copied files.\n')
print(target)
