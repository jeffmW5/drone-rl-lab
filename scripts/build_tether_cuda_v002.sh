#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${PUFFER_CUDA_HOME:-/home/n33du/ai/cuda-12.6-puffer}"
exec 9>tmp/tether-training.lock
flock -n 9 || { echo 'GPU trainer owns lease; defer compilation'; exit 1; }
test ! -e tmp/cuda-tether-v002
mkdir -p tmp/cuda-tether-v002
"$CUDA_HOME/bin/nvcc" -O3 -arch=sm_61 --cubin tether_sim/cuda/physics_v002.cu -o tmp/cuda-tether-v002/physics.cubin
gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -fPIC -I"$CUDA_HOME/include" tether_sim/cuda/host.c -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib -lcuda -o tmp/cuda-tether-v002/libtether.so
sha256sum tether_sim/cuda/physics_v002.cu tether_sim/cuda/host.c > tmp/cuda-tether-v002/sources.sha256
