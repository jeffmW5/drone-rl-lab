#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
runtime="${PUFFER_TETHER_PYTHON:-/home/n33du/ai/puffer-tether-venv/bin/python}"
toolkit="${PUFFER_CUDA_HOME:-/home/n33du/ai/cuda-12.6-puffer}"
if [[ -x "$toolkit/bin/nvcc" ]]; then
    export CUDA_HOME="$toolkit"
    export PATH="$toolkit/bin:$PATH"
fi
exec "$runtime" -u -m tether_sim.puffer_train "$@"
