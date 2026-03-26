# drone-rl-lab

An agentic loop for autonomously training and iterating on RL drone controllers,
inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Two backends:
- **Hover** -- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) + [SB3 PPO](https://github.com/DLR-RM/stable-baselines3)
- **Racing** -- [lsy_drone_racing](https://github.com/utiasDSL/lsy_drone_racing) + CleanRL PPO (JAX)

---

## How it works

```text
Windows Claude (Orchestrator)
    | writes experiment config YAML + inbox/INBOX.md
    |
Shared Folder / Git (bridge)
    |
Linux Claude (Executor)
    | reads queue -> trains -> benchmarks -> documents
    | writes outbox/STATUS.md + results/
    |
Windows Claude
    | reads outbox/, updates memory, queues next experiment
```

Each experiment is a frozen YAML config. The executor processes a task queue
(`inbox/INBOX.md`) with actionable statuses such as `[IMPLEMENTED]`, `[READY]`,
or the legacy `[NEXT]` / `[QUEUED]` tags, plus `[IN PROGRESS]` / `[DONE]`
lifecycle states. Memory is split across `memory/` files so agents never repeat
mistakes. Automation also writes a machine-readable snapshot to
`state/current.json`.

---

## Current status

**Goal: sub-5s average lap time on Level 2** (top 3 on the TUM Kaggle leaderboard)

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |
| -- | **Our best legacy finish (exp_016)** | **13.49** |

### Where we are now

| Phase | Experiments | Key finding |
|-------|-------------|-------------|
| Hover (PyBullet) | 001-005 | `ONE_D_RPM` caps reward around 474. Quartic hover reward is the best hover result. |
| Trajectory-following racing | 010-021 | RL can finish Level 2 (`exp_016`: 2/10 at 13.49s), but the fixed-spline approach hit its ceiling. |
| Direct RaceCoreEnv bootstrap | 022-023 | Direct gate racing works without trajectory tracking; early runs reached up to 0.8 average gates on Level 2 benchmark. |
| Reward tuning + benchmark alignment | 028-056 | Training reward can improve dramatically without benchmark gains; start-state and stability remain the bottleneck. |
| Structural fixes | 057-059 | Body-frame observations, soft-collision curriculum, and asymmetric critic are the active line. |

### Current thread

- `exp_056` finished on 2026-03-25 with **28.92 final mean reward** (peak 40.96)
  but **0 gates** and a **0.64s average dive-crash** benchmark.
- `exp_057` (body-frame gate observations) is currently training on RunPod.
- `exp_058` (soft-collision curriculum) and `exp_059` (asymmetric actor-critic)
  are implemented and queued.
- `exp_046` remains the best short-flight benchmark reference in the current
  direct-racing line: **1.37s average flight**, **0 gates**, but consistent
  flights toward gate 0.

### The gap

The repo is no longer asking "can RL finish at all?" but "can a direct
`RaceCoreEnv` policy survive the standardized benchmark, start stringing gates
together from the real start state, and eventually turn that into fast laps?"

Benchmark outcomes are the primary signal. Training reward is still useful
within a local experiment family, but it is not a global leaderboard once
reward definitions change.

---

## Quick start

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab

# Run an experiment
python train.py configs/exp_NNN.yaml

# View leaderboard
python compare.py
python compare.py --csv
python compare.py --filter level=level2
python compare.py --generate-log    # auto-update memory/EXPERIMENT_LOG.md

# Benchmark in MuJoCo sim
python scripts/benchmark.py -e exp_NNN -n 5

# Generate training curves
python plot.py
```

### GPU training (RunPod)

```bash
# On a fresh RunPod pod (RTX 3090, PyTorch template):
bash scripts/setup_runpod.sh        # installs Pixi GPU env + RL extras, 4h auto-shutdown
drone-rl-gpu-python train.py configs/exp_NNN.yaml
bash scripts/sync_results.sh "description"
# STOP YOUR POD
```

### Autonomous operation

```bash
# Process the full INBOX queue via Claude Code
bash scripts/run_experiment.sh

# Or use the pipeline (train -> benchmark -> document -> push per task)
bash scripts/run_experiment.sh --pipeline

# Watch for results on Windows side
bash scripts/watch_results.sh
```

---

## Project structure

```text
drone-rl-lab/
|-- train.py              # Dispatcher (routes to hover or racing trainer)
|-- train_hover.py        # Hover training (SB3 PPO)
|-- train_racing.py       # Racing training (CleanRL PPO, JAX)
|-- compare.py            # Leaderboard (--csv, --filter, --generate-log)
|-- plot.py               # Training curve plotter
|-- program.md            # Research goals, rules, documentation standard
|-- CLAUDE.md             # Claude Code session instructions
|-- MEMORY.md             # Index -> memory/ files
|
|-- memory/
|   |-- HARD_RULES.md     # Absolute constraints (never violate)
|   |-- EXPERIMENT_LOG.md # Auto-generated experiment history
|   |-- INSIGHTS.md       # Kaggle targets, benchmarks, architecture notes
|   `-- NEXT.md           # Prioritized research queue
|
|-- state/
|   `-- current.json      # Machine-readable snapshot for autonomous agents
|
|-- inbox/
|   `-- INBOX.md          # Task queue (supports IMPLEMENTED/READY/IN PROGRESS/DONE)
|-- outbox/
|   |-- STATUS.md         # Quick summary for orchestrator
|   `-- exp_NNN.md        # Per-experiment reports
|
|-- configs/              # Experiment configs (one YAML per experiment)
|-- results/              # Per-experiment outputs (metrics, checkpoints, docs)
|
`-- scripts/
    |-- run_experiment.sh   # One-command launcher (Claude Code or pipeline)
    |-- pipeline.sh         # Full auto-chain: train -> benchmark -> document -> push
    |-- benchmark.py        # Structured MuJoCo sim benchmarking
    |-- parse_queue.py      # INBOX queue parser
    |-- task_queue.py       # Shared queue parsing/mutation helpers
    |-- lab_state.py        # Writes state/current.json
    |-- capture_provenance.py # Saves repo/env provenance per experiment
    |-- watch_results.sh    # Windows-side git polling
    |-- setup_runpod.sh     # RunPod GPU setup (Pixi-first)
    |-- setup_deploy_key.sh # GitHub deploy key setup
    `-- sync_results.sh     # Pull results from RunPod
```

---

## Communication protocol

```text
Orchestrator                    Executor
     |                              |
     |--- inbox/INBOX.md ---------> | actionable task with config
     |                              | train -> benchmark -> document
     |                              | mark [DONE], refresh state
     |<-- outbox/STATUS.md -------- | results summary
     |<-- state/current.json ------ | machine-readable state
     |<-- outbox/exp_NNN.md ------- | detailed report
     |                              |
     |--- review results            |
     |--- update memory/INSIGHTS.md |
     |--- queue next task           |
     `--- git push                  |
```

---

## Experiment log

Run `python compare.py` for the live leaderboard, or see
`memory/EXPERIMENT_LOG.md` for the full history.
