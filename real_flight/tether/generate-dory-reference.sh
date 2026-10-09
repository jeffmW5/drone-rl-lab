#!/usr/bin/env bash
set -euo pipefail
root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "$root/generated/dory-reference/src" "$root/generated/dory-reference/inc"
cd "$root/vendor/dory"
"$HOME/ai/crazyflie-dory-venv/bin/python" network_generate.py NEMO PULP.GAP8 \
  dory/dory_examples/config_files/config_NEMO_dronet.json \
  --app_dir "$root/generated/dory-reference"
