# drone-rl-lab

An agentic loop for autonomously training and iterating on RL drone controllers,
inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
with PPO from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## How it works

```
Windows Claude (Orchestrator)
    ↓ writes experiment config YAML + INBOX.md
Shared Folder (bridge)
    ↑ reads outbox/exp_NNN.md results
Linux Claude (Executor)
    → creates config → runs training → documents results
```

Each experiment is a frozen YAML config. The history forms a readable
research log of what worked, what didn't, and why.

---

## Quick start

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab

# Run an experiment
python train_rl.py configs/exp_001_baseline.yaml

# View leaderboard
python compare.py

# Generate training curve plots
python plot.py

# Per-step detail for one experiment
python plot.py --steps exp_001_baseline
```

---

## Project structure

```
drone-rl-lab/
├── train_rl.py          # Training infrastructure (don't edit)
├── compare.py           # Leaderboard tool
├── plot.py              # Training curve plotter
├── program.md           # Research goals & rules
├── INBOX.md             # Windows Claude → Linux Claude
├── configs/             # Experiment configs (one YAML per experiment)
│   ├── exp_001_baseline.yaml
│   ├── exp_002_extended_budget.yaml
│   └── ...
├── results/             # Per-experiment outputs
│   ├── exp_001_baseline/
│   │   ├── config.yaml      # Frozen copy of the config used
│   │   ├── metrics.json     # Quantitative results
│   │   ├── evaluations.npz  # Training curve data
│   │   ├── steps.csv        # Per-step distance/velocity (v2+)
│   │   ├── best_model.zip   # Best policy checkpoint
│   │   └── EXPERIMENT.md    # Documented explanation
│   └── ...
└── outbox/              # Linux Claude → Windows Claude (one per experiment)
    ├── exp_001_baseline.md
    └── ...
```

---

## Experiment log

| # | Name | Reward | Key finding |
|---|------|--------|-------------|
| 001 | quartic baseline | **474.171** | Reference point — 97.9% of theoretical max |
| 002 | extended budget (6min) | 474.206 | More time doesn't help — reward is the ceiling |
| 003 | quadratic reward | 465.792 | Stronger gradient = more crashes, worse performance |
| 004 | velocity penalty | 470.394 | Penalty destabilizes rather than helps |
| 005 | conservative PPO | 437.347 | No collapses, but lower ceiling — collapses may be useful |
