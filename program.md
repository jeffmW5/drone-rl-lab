# Drone RL Lab — Program

## Mission
Train a Crazyflie CF2X drone to hover stably at position [0, 0, 1.0] meters
using Proximal Policy Optimization (PPO) and the gym-pybullet-drones simulator.

We iterate experiment-by-experiment, guided by Windows Claude (orchestrator),
executed by Linux Claude (executor). Each experiment is a frozen YAML config.

---

## How to run an experiment

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab
python train_rl.py configs/exp_NNN.yaml
```

### After training completes:
1. Read the printed results
2. Write `results/exp_NNN/EXPERIMENT.md` (see documentation standard below)
3. Run `python compare.py` to see the leaderboard
4. Run `python plot.py` to generate training curves
5. Update `outbox/exp_NNN.md` with your analysis
6. `git add -A && git commit -m "exp_NNN: <short description>"`
7. `git push`

---

## How to create a new experiment

Linux Claude: create a new YAML file in `configs/`:

```yaml
name: exp_006_description
hypothesis: "What you're testing and why"
budget_seconds: 180

reward_code: |
  dist = np.linalg.norm(self.TARGET_POS_CUSTOM - state[0:3])
  return max(0, 2 - dist**4)

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95
```

**Do NOT edit train_rl.py** — all experiment parameters live in the config.

---

## Available tools

| Tool | What it does |
|------|-------------|
| `python train_rl.py configs/exp_NNN.yaml` | Run an experiment |
| `python compare.py` | Print sorted leaderboard of all experiments |
| `python plot.py` | Generate training curve plots |
| `python plot.py --steps exp_NNN` | Plot per-step distance/velocity detail |

---

## What Linux Claude is allowed to change

### ✅ YES
- Create new config YAML files in `configs/`
- Write `results/exp_NNN/EXPERIMENT.md`
- Write `outbox/exp_NNN.md`
- Git commit and push

### ❌ NO — do not modify
- `train_rl.py` (infrastructure — don't touch)
- `compare.py`, `plot.py` (tools — don't touch)
- Physics engine, episode length, drone model
- Observation type (keep `KIN`)
- Action type (keep `ONE_D_RPM`) — unless explicitly instructed by Windows Claude

---

## Evaluation Metric
**Primary:** `mean_reward` from 10 evaluation episodes after training
**Secondary:** `timesteps_trained` within the budget (sample efficiency proxy)

Higher mean_reward = better. An experiment is a success if mean_reward improves
over the previous best.

---

## EXPERIMENT.md Documentation Standard

Every experiment produces `results/exp_NNN/EXPERIMENT.md`:

```markdown
# Experiment NNN — <Short Title>

## What we changed
<Specific change — reference the config file>

## Why (the RL concept)
<What RL principle does this test? Explain simply and accurately.>

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | X.XX | Y.YY ✅/❌ |
| timesteps_trained | N | M |

## What this tells us
<Conclusions. Be honest about uncertainty.>

## Questions this opens up
- <Question 1>
- <Question 2>

## Suggested next experiment
<One specific hypothesis to test next>
```
