# drone-rl-lab

An agentic loop for autonomously training and iterating on RL drone controllers,
inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Two backends:
- **Hover** — [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) + [SB3 PPO](https://github.com/DLR-RM/stable-baselines3)
- **Racing** — [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) + CleanRL PPO (JAX)

---

## How it works

```
Windows Claude (Orchestrator)
    | writes experiment config YAML + INBOX.md
Shared Folder (bridge)
    | reads outbox/ results + MEMORY.md
Linux Claude (Executor)
    -> runs training -> benchmarks in sim -> documents results
```

Each experiment is a frozen YAML config. The history forms a readable
research log of what worked, what didn't, and why. `MEMORY.md` preserves
lessons across sessions so agents never repeat mistakes.

---

## Current status

**Goal: sub-5s average lap time on Level 2** (top 3 on TUM Kaggle leaderboard)

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |
| — | **Our best (exp_016)** | **13.49** |

### Progress

| Phase | Experiments | Key finding |
|-------|-----------|-------------|
| Hover (PyBullet) | 001-005 | ONE_D_RPM action space caps reward at ~474. Quartic reward is optimal. |
| Racing CPU (L0) | 010, 013 | PPO learns trajectory following. 13.36s lap, beats PID. n_obs=2 needs more compute. |
| Racing GPU (L0-L2) | 014-016 | n_obs=2 works with 1024 envs. First RL to finish Level 2 (2/10, 13.49s). Reward plateaus at ~7.7. |
| Dynamic trajectory | dyn | Inference-time trajectory swap fails — policy coupled to training trajectory shape. |

### The gap

Our exp_016 (10M GPU steps) is the **first RL model to finish Level 2 at all** — the reference RL goes 0/5. But we're 3-4x slower than competition winners. The bottleneck is **trajectory generation**: all controllers follow a fixed spline that doesn't adapt to randomized gate positions.

---

## Quick start

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab

# Run an experiment
python train.py configs/exp_016_gpu_level2_long.yaml

# View leaderboard
python compare.py

# Generate training curves
python plot.py
```

### GPU training (RunPod)

```bash
# On a fresh RunPod pod (RTX 3090, PyTorch template):
bash scripts/setup_runpod.sh        # installs everything, 4h auto-shutdown
python train.py configs/exp_016_gpu_level2_long.yaml
bash scripts/sync_results.sh "description"
# STOP YOUR POD
```

---

## Project structure

```
drone-rl-lab/
+-- train.py              # Dispatcher (routes to hover or racing trainer)
+-- train_hover.py        # Hover training (SB3 PPO)
+-- train_racing.py       # Racing training (CleanRL PPO, JAX)
+-- compare.py            # Leaderboard tool
+-- plot.py               # Training curve plotter
+-- program.md            # Research goals, rules, documentation standard
+-- MEMORY.md             # Institutional memory (hard rules, experiment log)
+-- INBOX.md              # Windows Claude -> Linux Claude instructions
+-- configs/              # Experiment configs (one YAML per experiment)
+-- results/              # Per-experiment outputs (metrics, checkpoints, docs)
+-- outbox/               # Linux Claude -> Windows Claude reports
+-- scripts/              # RunPod setup, deploy key, sync scripts
```

---

## Experiment log

| # | Name | Backend | Level | Reward | Lap (s) | Key finding |
|---|------|---------|-------|:------:|:-------:|-------------|
| 001 | quartic baseline | hover | — | 474 | — | Ceiling for 1D RPM action space |
| 002 | extended budget | hover | — | 474 | — | More time doesn't help |
| 003 | quadratic reward | hover | — | 369 | — | Worse — quartic is optimal |
| 004 | velocity penalty | hover | — | 407 | — | Penalty destabilizes |
| 005 | conservative PPO | hover | — | 437 | — | Stability vs performance tradeoff |
| 010 | racing baseline | racing | L0 | 7.36 | 13.36 | Beats PID, 0.024s off reference |
| 013 | n_obs=2 fix | racing | L0 | 5.02 | DNF | Undertrained — needs GPU |
| 014 | GPU n_obs=2 | racing | L0 | 7.29 | — | Validates n_obs=2 with enough compute |
| 015 | GPU Level 2 | racing | L2 | 7.53 | — | First L2 training, still climbing |
| **016** | **GPU L2 extended** | **racing** | **L2** | **7.71** | **13.49** | **First RL to finish L2 (2/10)** |
