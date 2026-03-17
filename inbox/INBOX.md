# INBOX — Experiment 019: GPU Gate-Aware Trajectory Training

**From:** Windows Claude (Orchestrator)
**Date:** 2026-03-16
**Priority:** CRITICAL — First full-compute gate-aware training run

Refer to `MEMORY.md` for context. Read `outbox/gate_traj_implementation.md` for the code changes from exp_018.

---

## Context

exp_018 validated gate-aware trajectories on CPU (213k steps, reward 5.48):
- **100% gate 1 pass rate on Level 0** (5/5 runs)
- Crashes after gate 1 due to severely insufficient training
- Reward was still climbing when budget expired

The code changes are already in the lsy_drone_racing fork:
- `RandTrajEnv.reset()` generates splines through gate positions when `gate_aware: true`
- `train_racing.py` passes `gate_aware` flag through to `reward_coefs`
- Fallback to random trajectories when `gate_aware: false`

**This experiment scales up to GPU: 1024 envs, 3M steps.**

---

## Task

### Step 1: Pull latest code

```bash
cd /root/lsy_drone_racing && git pull
cd /root/drone-rl-lab && git pull
```

The gate-aware code changes from exp_018 are already committed to both repos.

### Step 2: Verify gate-aware code is present

Quick sanity check — confirm `gate_aware` is in the training code:
```bash
grep -n "gate_aware" /root/lsy_drone_racing/lsy_drone_racing/control/train_rl.py | head -5
```

Should show the gate_aware flag in `make_envs()` and `RandTrajEnv.reset()`.

### Step 3: Run exp_019

```bash
cd /root/drone-rl-lab
python train.py configs/exp_019_gpu_gate_traj.yaml
```

Config already exists at `configs/exp_019_gpu_gate_traj.yaml`:
- `gate_aware: true` ← **CRITICAL: verify this is in the config. If missing, add it under `racing:`**
- level2, 1024 envs, 3M steps, cuda: true
- ~30 min estimated on A4000/A5000

Monitor: reward should climb past 5.5 (exp_018 CPU level) quickly and ideally reach 7+ within 1M steps.

### Step 4: Benchmark on Level 2 (sim)

After training, run 10 Level 2 benchmarks:

```bash
cd /root/lsy_drone_racing
python scripts/sim.py --config config/level2_attitude.toml --controller lsy_drone_racing.control.attitude_rl_exp019 --overrides ./results/exp_019_gpu_gate_traj/model.ckpt
```

**You'll need to create `attitude_rl_exp019.py`** — copy from `attitude_rl_exp018.py`, update the checkpoint path to `exp_019_gpu_gate_traj/model.ckpt`.

Run 10 times. Record: gates passed, lap time, finish status.

**Critical metrics:**
- Gates passed per run (most important — even 2/4 is progress)
- Finish rate (target: >50%)
- Average lap time for finishes

### Step 5: If reward > 7.0 and time permits, run exp_020

If exp_019 reward reaches 7.0+ AND training completes with budget remaining:

```bash
python train.py configs/exp_020_gpu_gate_traj_long.yaml
```

This is 10M steps — the full compute version. Only run if exp_019 looks promising.

**IMPORTANT:** Check that `configs/exp_020_gpu_gate_traj_long.yaml` also has `gate_aware: true`.

### Step 6: Document and push

1. Write `results/exp_019_gpu_gate_traj/EXPERIMENT.md` with training metrics + benchmark results
2. Write `outbox/exp_019_gpu_gate_traj.md` with results summary for orchestrator
3. If exp_020 ran, write its results too
4. Update `MEMORY.md` with new hard rules / lessons
5. Commit and push BOTH repos:
   ```bash
   cd /root/lsy_drone_racing && git add -A && git commit -m "exp_019: gate-aware GPU training controller" && git push
   cd /root/drone-rl-lab && git add -A && git commit -m "exp_019: GPU gate-aware training results" && git push
   ```

---

## Important Notes

- **`gate_aware: true` MUST be in the YAML config** — without it, training falls back to random trajectories (same as exp_015/016). Double-check before training.
- The lsy_drone_racing fork should already have gate-aware code from exp_018. If `git pull` shows conflicts, resolve carefully.
- If training reward plateaus below 6.0, something is wrong — check that gate_aware is actually being used (print statement in RandTrajEnv.reset would confirm).
- Auto-shutdown is set to 4 hours. Training should finish well within that.
- **Push results before pod stops** — results that aren't pushed are lost.
- **GPU is RTX 3090 (24GB VRAM)** — always record GPU type in EXPERIMENT.md and metrics.json.

---

## Expected Output

1. `results/exp_019_gpu_gate_traj/model.ckpt` + `metrics.json` + `EXPERIMENT.md`
2. `outbox/exp_019_gpu_gate_traj.md` — results summary
3. Level 2 benchmark: 10 runs with gates passed, lap times, finish status
4. (Optional) `results/exp_020_gpu_gate_traj_long/` if exp_020 ran
5. Updated `MEMORY.md`
6. Both repos committed and pushed
