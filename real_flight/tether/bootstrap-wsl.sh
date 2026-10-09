#!/usr/bin/env bash
set -euo pipefail
root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
venv="$HOME/ai/crazyflie-tether-venv"
mkdir -p "$root"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
"$venv/bin/python" -m pip install numpy==2.4.6 cflib==0.1.34 cfclient==2026.8.1 onnx==1.17.0 onnxruntime==1.20.1
"$venv/bin/python" -m pip freeze > "$root/requirements-resolved.txt"
echo "Activate: source $venv/bin/activate"
