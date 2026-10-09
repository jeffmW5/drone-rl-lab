#!/usr/bin/env bash
set -euo pipefail
root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$root"
"$HOME/ai/crazyflie-tether-venv/bin/python" quantize.py runs/smoke/model.pt --output runs/quantized
mkdir -p generated/tether-l2/src generated/tether-l2/inc
cd vendor/dory
"$HOME/ai/crazyflie-dory-venv/bin/python" network_generate.py NEMO PULP.GAP8_L2 \
  "$root/runs/quantized/config.json" --app_dir "$root/generated/tether-l2"
cd "$root"
"$HOME/ai/crazyflie-tether-venv/bin/python" prepare_firmware.py
