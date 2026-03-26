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
    | writes experiment config YAML + inbox/INBOX.md
    |
Shared Folder / Git (bridge)
    |
Linux Claude (Executor)
    | reads inbox/ queue → trains → benchmarks → documents
    | writes outbox/STATUS.md + results/
    |
Windows Claude
    | reads outbox/, updates memory, queues next experiment
```

Each experiment is a frozen YAML config. The executor processes a task queue
(`inbox/INBOX.md`) with `[NEXT]`/`[QUEUED]`/`[DONE]` status tags. Memory is
split across `memory/` files so agents never repeat mistakes.

---

## Current status

**Goal: sub-5s average lap time on Level 2** (top 3 on TUM Kaggle leaderboard)

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |
| — | **Our best (exp_020)** | **~13.5** |

### Progress

| Phase | Experiments | Key finding |
|-------|-----------|-------------|
| Hover (PyBullet) | 001-005 | ONE_D_RPM action space caps reward at ~474. Quartic reward is optimal. |
| Racing CPU (L0) | 010, 013 | PPO learns trajectory following. 13.36s lap, beats PID. n_obs=2 needs more compute. |
| Racing GPU (L0-L2) | 014-016 | n_obs=2 works with 1024 envs. First RL to finish Level 2 (2/10, 13.49s). |
| Gate-aware traj | 018-020 | Gate-aware trajectories improve reward. Best reward 7.79 (exp_020, 10M GPU steps). |
| Yaw-aware traj | 021 | Yaw approach/departure vectors fix gate 1→2 crashes. 2.4x more gates passed. |

### The gap

We're 3-4x slower than competition winners. The bottleneck is the **trajectory-following approach** itself — the policy follows a fixed spline and can't adapt fast enough to randomized gate positions. Next step: train directly on `RaceCoreEnv` with waypoint rewards instead of trajectory tracking.

---

## Quick start

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab

# Run an experiment
python train.py configs/exp_020_gpu_gate_traj_long.yaml

# View leaderboard
python compare.py
python compare.py --csv
python compare.py --filter level=level2
python compare.py --generate-log    # auto-update memory/EXPERIMENT_LOG.md

# Benchmark in MuJoCo sim
python scripts/benchmark.py -e exp_020_gpu_gate_traj_long -n 5

# Generate training curves
python plot.py
```

### GPU training (RunPod)

```bash
# On a fresh RunPod pod (RTX 3090, PyTorch template):
bash scripts/setup_runpod.sh        # installs Pixi GPU env + RL extras, 4h auto-shutdown
drone-rl-gpu-python train.py configs/exp_020_gpu_gate_traj_long.yaml
bash scripts/sync_results.sh "description"
# STOP YOUR POD
```

### Autonomous operation

```bash
# Process the full INBOX queue via Claude Code
bash scripts/run_experiment.sh

# Or use the pipeline (train → benchmark → document → push per task)
bash scripts/run_experiment.sh --pipeline

# Watch for results on Windows side
bash scripts/watch_results.sh
```

---

## Project structure

```
drone-rl-lab/
├── train.py              # Dispatcher (routes to hover or racing trainer)
├── train_hover.py        # Hover training (SB3 PPO)
├── train_racing.py       # Racing training (CleanRL PPO, JAX) + early stopping
├── compare.py            # Leaderboard (--csv, --filter, --generate-log)
├── plot.py               # Training curve plotter
├── program.md            # Research goals, rules, documentation standard
├── CLAUDE.md             # Claude Code session instructions
├── MEMORY.md             # Index → memory/ files
│
├── memory/
│   ├── HARD_RULES.md     # Absolute constraints (never violate)
│   ├── EXPERIMENT_LOG.md # Auto-generated experiment history
│   ├── INSIGHTS.md       # Kaggle targets, benchmarks, architecture notes
│   └── NEXT.md           # Prioritized research queue
│
├── inbox/
│   └── INBOX.md          # Task queue ([NEXT]/[QUEUED]/[DONE])
├── outbox/
│   ├── STATUS.md         # Quick summary for orchestrator
│   └── exp_NNN.md        # Per-experiment reports
│
├── configs/              # Experiment configs (one YAML per experiment)
├── results/              # Per-experiment outputs (metrics, checkpoints, docs)
│
└── scripts/
    ├── run_experiment.sh   # One-command launcher (Claude Code or pipeline)
    ├── pipeline.sh         # Full auto-chain: train → benchmark → document → push
    ├── benchmark.py        # Structured MuJoCo sim benchmarking
    ├── parse_queue.py      # INBOX queue parser
    ├── watch_results.sh    # Windows-side git polling
    ├── setup_runpod.sh     # RunPod GPU setup (Pixi-first)
    ├── setup_deploy_key.sh # GitHub deploy key setup
    └── sync_results.sh     # Pull results from RunPod
```

---

## Communication protocol

```
Orchestrator                    Executor
     │                              │
     ├─── inbox/INBOX.md ──────────>│  [NEXT] task with config
     │                              ├── train → benchmark → document
     │                              ├── mark [DONE], advance queue
     │<── outbox/STATUS.md ─────────┤  results summary
     │<── outbox/exp_NNN.md ────────┤  detailed report
     │                              │
     ├── review results             │
     ├── update memory/INSIGHTS.md  │
     ├── queue next [QUEUED] tasks  │
     └── git push                   │
```

---

## Experiment log

Run `python compare.py` for the live leaderboard, or see `memory/EXPERIMENT_LOG.md` for the full history.
