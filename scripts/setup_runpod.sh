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
cd /root

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

# ── Install dependencies ─────────────────────────────────────────────────────
echo "[3/5] Installing lsy_drone_racing (GPU)..."
cd /root/lsy_drone_racing
pip install -e ".[sim,rl,gpu]" --quiet

echo "[4/5] Installing drone-rl-lab deps..."
cd /root/drone-rl-lab
pip install pyyaml stable-baselines3 gym-pybullet-drones matplotlib --quiet

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "[5/5] Verifying GPU..."
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# ── Configure git ─────────────────────────────────────────────────────────────
git config --global user.name "JefferyWhitmire"
git config --global user.email "jeffwhitmire33@gmail.com"

# ── Set up deploy key for git push ────────────────────────────────────────────
echo "[+] Setting up deploy key for GitHub push..."
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

if [ -f /root/drone-rl-lab/ssh_key_runpod ]; then
    # Use the pre-committed deploy key (already added to GitHub)
    cp /root/drone-rl-lab/ssh_key_runpod ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
    echo "  Using pre-configured deploy key from repo"
elif [ ! -f ~/.ssh/id_ed25519 ]; then
    # Fallback: generate a new key (requires manual GitHub setup)
    ssh-keygen -t ed25519 -C "runpod-drone-rl-lab" -f ~/.ssh/id_ed25519 -N ""
    echo ""
    echo "  NEW KEY — add to https://github.com/jeffmW5/drone-rl-lab/settings/keys"
    cat ~/.ssh/id_ed25519.pub
    echo ""
fi

# Switch remote to SSH so git push works
cd /root/drone-rl-lab
git remote set-url origin git@github.com:jeffmW5/drone-rl-lab.git

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  SETUP COMPLETE"
echo ""
echo "  cd /root/drone-rl-lab"
echo "  python train.py configs/exp_011_racing_gpu.yaml"
echo ""
echo "  After training: git add -A && git commit && git push"
echo "  Auto-shutdown in ${MAX_HOURS}h. Billing ~\$0.30/hr."
echo "================================================"
echo ""
