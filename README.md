# drone-rl-lab

An agentic loop for autonomously training and iterating on RL drone controllers,
inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
with PPO from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## How it works

```
Windows Claude (Orchestrator)
    ↓ writes experiment proposals to INBOX.md
Shared Folder (bridge)
    ↑ reads results from OUTBOX.md
Linux Claude (Executor)
    → modifies train_rl.py → runs training → documents results
```

Each successful experiment is a git commit. The history forms a readable
research log of what worked, what didn't, and why.

---

## Setup

### Requirements
- Ubuntu VM with gym-pybullet-drones installed
- Python venv with: `stable-baselines3`, `torch`, `pybullet`, `gymnasium`
- Shared folder accessible at `/media/sf_Shared/` (VirtualBox) or similar

### Running an experiment
```bash
source ~/repos/drones-venv/bin/activate
cd /path/to/drone-rl-lab
python train_rl.py
```

---

## Experiment Log

See `program.md` for the full experiment history and research goals.

Each experiment folder in `results/` contains:
- `metrics.json` — quantitative results
- `EXPERIMENT.md` — documented explanation (the lesson learned)

---

## Architecture

| File | Purpose |
|------|---------|
| `train_rl.py` | The training script — Linux Claude iterates on this |
| `program.md` | Research goals and rules for the agents |
| `INBOX.md` | Windows Claude → Linux Claude (next experiment) |
| `OUTBOX.md` | Linux Claude → Windows Claude (results) |
| `results/exp_NNN/` | Per-experiment artifacts and documentation |
