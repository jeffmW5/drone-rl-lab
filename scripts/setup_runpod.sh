#!/bin/bash
# =============================================================================
# Drone RL Lab — RunPod GPU Setup
# =============================================================================
# Run this on a fresh RunPod instance (PyTorch template recommended).
# Sets up everything needed for GPU-accelerated drone racing training.
#
# Usage: bash scripts/setup_runpod.sh
# =============================================================================

set -e  # Exit on any error

POD_ROOT="/root"
LAB_DIR="${POD_ROOT}/drone-rl-lab"
LSY_DIR="${POD_ROOT}/lsy_drone_racing"
SETUP_MARKER="${POD_ROOT}/.drone_rl_runpod_setup_complete"

# ── Safety: Auto-shutdown timer ──────────────────────────────────────────────
MAX_HOURS=4
echo ""
echo "================================================"
echo "  AUTO-SHUTDOWN in ${MAX_HOURS} hours"
echo "  Pod will stop automatically to prevent billing."
echo "  Cancel with: kill %1"
echo "================================================"
echo ""
(sleep ${MAX_HOURS}h && echo "AUTO-SHUTDOWN: Time limit reached!" && runpodctl stop pod $RUNPOD_POD_ID 2>/dev/null || shutdown -h now) &

# ── Clone repos ──────────────────────────────────────────────────────────────
cd "${POD_ROOT}"

if [ ! -d "drone-rl-lab" ]; then
    echo "[1/5] Cloning drone-rl-lab..."
    git clone https://github.com/jeffmW5/drone-rl-lab.git
else
    echo "[1/5] drone-rl-lab already exists, pulling latest..."
    cd drone-rl-lab && git pull && cd ..
fi

if [ ! -d "lsy_drone_racing" ]; then
    echo "[2/5] Cloning lsy_drone_racing..."
    git clone https://github.com/jeffmW5/lsy_drone_racing.git
else
    echo "[2/5] lsy_drone_racing already exists, pulling latest..."
    cd lsy_drone_racing && git pull && cd ..
fi

# ── Install Pixi + dependencies ──────────────────────────────────────────────
echo "[3/5] Setting up lsy_drone_racing via Pixi GPU environment..."
if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh
fi
export PATH="${HOME}/.pixi/bin:${PATH}"

cd "${LSY_DIR}"
# Resolve the base GPU environment from the local editable fork checkout.
pixi run -e gpu python -c "import jax; print(jax.devices())" >/dev/null
# Add RL extras inside the same Pixi-managed environment.
pixi run -e gpu pip install -e ".[rl]" --quiet

echo "[4/5] Installing drone-rl-lab deps into the Pixi GPU environment..."
pixi run -e gpu pip install pyyaml stable-baselines3 gym-pybullet-drones matplotlib --quiet

# Helper wrappers keep future pod commands short and guarantee they use the
# editable Pixi environment from the local fork checkout.
cat > /usr/local/bin/drone-rl-gpu-python <<'INNER'
#!/bin/bash
set -e
cd /root/lsy_drone_racing
exec /root/.pixi/bin/pixi run -e gpu python "$@"
INNER
chmod +x /usr/local/bin/drone-rl-gpu-python

cat > /usr/local/bin/drone-rl-gpu-pip <<'INNER'
#!/bin/bash
set -e
cd /root/lsy_drone_racing
exec /root/.pixi/bin/pixi run -e gpu pip "$@"
INNER
chmod +x /usr/local/bin/drone-rl-gpu-pip

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "[5/5] Verifying GPU..."
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
drone-rl-gpu-python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
drone-rl-gpu-python -c "import jax; print(f'JAX devices: {jax.devices()}')"
drone-rl-gpu-python -c "import lsy_drone_racing; print(f'lsy_drone_racing: {lsy_drone_racing.__file__}')"

# ── Configure git ─────────────────────────────────────────────────────────────
git config --global user.name "JefferyWhitmire"
git config --global user.email "jeffwhitmire33@gmail.com"

# ── Set up deploy key for git push ────────────────────────────────────────────
echo "[+] Setting up deploy key for GitHub push..."
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

if [ -f ~/.ssh/id_ed25519 ]; then
    echo "  Deploy key already in place"
elif [ ! -f ~/.ssh/id_ed25519 ]; then
    # Fallback: generate a new key (requires manual GitHub setup)
    ssh-keygen -t ed25519 -C "runpod-drone-rl-lab" -f ~/.ssh/id_ed25519 -N ""
    echo ""
    echo "  NEW KEY — add to https://github.com/jeffmW5/drone-rl-lab/settings/keys"
    cat ~/.ssh/id_ed25519.pub
    echo ""
fi

# Switch remote to SSH so git push works
cd "${LAB_DIR}"
git remote set-url origin git@github.com:jeffmW5/drone-rl-lab.git

cat > "${SETUP_MARKER}" <<INNER
setup_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
lab_dir=${LAB_DIR}
lsy_dir=${LSY_DIR}
pixi_env=gpu
pixi_bin=/root/.pixi/bin/pixi
INNER

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  SETUP COMPLETE"
echo ""
echo "  cd ${LAB_DIR}"
echo "  drone-rl-gpu-python train.py configs/exp_011_racing_gpu.yaml"
echo ""
echo "  Benchmark helper: cd ${LSY_DIR} && pixi run -e gpu python scripts/sim.py -r"
echo "  After training: git add -A && git commit && git push"
echo "  Auto-shutdown in ${MAX_HOURS}h. Billing ~\$0.30/hr."
echo "================================================"
echo ""
