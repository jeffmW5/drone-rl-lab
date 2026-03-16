# INBOX — GPU Training on RunPod

**From:** Windows Claude (orchestrator)
**To:** Linux Claude (executor)
**Date:** 2026-03-15

Refer to `MEMORY.md` for context on past experiments.

## Context

Phase 1 (inference-time trajectory swapping) failed — the policy is tightly coupled to its training trajectory shape (see MEMORY.md Hard Rule #6). We need to **retrain with diverse trajectories on GPU**.

A RunPod GPU pod is running. You need to SSH into it, set up the environment, and run experiments.

**Target: sub-5s average on Level 2 (Kaggle top 3)**

## Step 0: Set up SSH key

```bash
mkdir -p ~/.ssh
cp /media/drone-rl-lab/ssh_key_runpod ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
```

## Step 1: SSH into RunPod

```bash
ssh nwcko9i3b6tz65-64410b43@ssh.runpod.io -i ~/.ssh/id_ed25519
```

If prompted about host authenticity, type `yes`.

## Step 2: Set up the pod

Once SSH'd in, run:

```bash
cd /root
git clone https://github.com/jeffmW5/drone-rl-lab.git
git clone https://github.com/jeffmW5/lsy_drone_racing.git

cd /root/lsy_drone_racing
pip install -e ".[sim,rl]"

cd /root/drone-rl-lab
pip install pyyaml

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

**If CUDA/JAX don't see the GPU, STOP and write the error to outbox.**

## Step 3: Run exp_014 — L0 GPU validation (n_obs=2, 1024 envs)

```bash
cd /root/drone-rl-lab
python train.py configs/exp_014_gpu_level0_nobs2.yaml
```

**Expected:** ~5-10 min. Reward should exceed 7.0. This validates that n_obs=2 works with enough compute (exp_013 failed on CPU with only 64 envs).

Write `results/exp_014.../EXPERIMENT.md` per program.md standard.

## Step 4: Run exp_015 — L2 competition (3M steps)

```bash
python train.py configs/exp_015_gpu_level2.yaml
```

**Expected:** ~10-20 min. This is the first real shot at Level 2. The agent will train directly on randomized gates. Watch for:
- Does reward climb steadily or plateau early?
- Any reward collapses?

Write `results/exp_015.../EXPERIMENT.md`.

## Step 5: Run exp_016 — L2 extended (10M steps, conditional)

**Only run this if** exp_015's reward was still climbing at 3M steps:

```bash
python train.py configs/exp_016_gpu_level2_long.yaml
```

Write `results/exp_016.../EXPERIMENT.md`.

## Step 6: Copy results back

```bash
cd /root/drone-rl-lab
git config user.email "linux-claude@drone-rl-lab"
git config user.name "Linux Claude"
git add results/ outbox/
git commit -m "GPU training: exp_014-016 results"
```

**To push**, you'll need to set up git credentials on the pod. If git push fails, just leave the results on the pod and write a summary to `/media/drone-rl-lab/outbox/gpu_results.md` instead (via the shared folder after exiting SSH).

## Step 7: Exit and report

```bash
exit
```

Back on the VM, write results to:
- `outbox/gpu_results.md` — summary of all GPU experiments
- Update `MEMORY.md` per program.md Step 8

## Important Notes

- The pod costs ~$0.30/hr. Don't leave it running idle.
- Do NOT modify `train.py`, `train_racing.py`, `compare.py`, or `plot.py`
- If any experiment fails, write the error to outbox and continue to the next one
- The pod has a 4-hour auto-shutdown as a safety net (if setup_runpod.sh was run), but try to finish within 1-2 hours
- Read MEMORY.md Hard Rules before starting — especially #1 and #6
