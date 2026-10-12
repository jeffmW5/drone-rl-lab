#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${PUFFER_CUDA_HOME:-/home/n33du/ai/cuda-12.6-puffer}"
mkdir -p tmp/cuda-tether-v001
exec 9>tmp/tether-training.lock
flock -n 9 || { echo 'GPU trainer owns lease; defer compilation'; exit 1; }
"$CUDA_HOME/bin/nvcc" -O3 -arch=sm_61 --cubin \
  tether_sim/cuda/physics_v001.cu -o tmp/cuda-tether-v001/physics.cubin
gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -fPIC -I"$CUDA_HOME/include" \
  tether_sim/cuda/host.c -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib -lcuda \
  -o tmp/cuda-tether-v001/libtether.so
sha256sum tether_sim/cuda/physics_v001.cu tether_sim/cuda/host.c > tmp/cuda-tether-v001/sources.sha256
