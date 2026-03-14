# Drone RL Lab — Program

## Mission
Train a Crazyflie CF2X drone to hover stably at position [0, 0, 1.0] meters
using Proximal Policy Optimization (PPO) and the gym-pybullet-drones simulator.

We iterate on `train_rl.py` experiment-by-experiment, guided by Windows Claude,
executed by Linux Claude. Each experiment is documented and committed to git.

---

## What Linux Claude is allowed to change in train_rl.py

### ✅ YES — modify these
- `EXPERIMENT_NAME` — increment and describe (e.g. `exp_002_linear_reward`)
- `EXPERIMENT_HYPOTHESIS` — one sentence describing what you're testing and why
- `PPO_KWARGS` — any PPO hyperparameters (learning_rate, n_steps, batch_size, gamma, gae_lambda)
- `CustomHoverAviary._computeReward()` — the reward function
- `CustomHoverAviary.TARGET_POS_CUSTOM` — the hover target position
- `TRAINING_BUDGET_SECONDS` — only increase if a specific experiment needs more time (ask first)

### ❌ NO — do not modify these
- Physics engine (keep `Physics.PYB`)
- Episode length (keep 8 seconds)
- Drone model (keep `CF2X`)
- Observation type (keep `KIN`)
- Action type (keep `ONE_D_RPM`) — unless explicitly instructed by Windows Claude
- The training infrastructure below the "do not modify" line
- The metrics.json and OUTBOX.md writing logic

---

## Evaluation Metric
**Primary:** `mean_reward` from 10 evaluation episodes after training
**Secondary:** `timesteps_trained` within the budget (sample efficiency proxy)

Higher mean_reward = better. An experiment is a success if mean_reward improves
over the previous best.

---

## How to run an experiment
```bash
source ~/repos/drones-venv/bin/activate
cd /media/sf_Shared/drone-rl-lab   # adjust path if different
python train_rl.py
```

After training completes:
1. Read the printed results
2. Write `results/exp_NNN/EXPERIMENT.md` (see documentation standard below)
3. If improved: `git add -A && git commit -m "exp_NNN: <short description> +X%"`
4. If not improved: `git checkout -- train_rl.py`
5. Update `OUTBOX.md` with a summary for Windows Claude

---

## EXPERIMENT.md Documentation Standard

Every experiment must produce `results/exp_NNN/EXPERIMENT.md` with this structure:

```markdown
# Experiment NNN — <Short Title>

## What we changed
<Specific code change, quoted or described precisely>

## Why (the RL concept)
<What RL principle does this test? Explain it simply and accurately.
If unsure about an explanation, say so — do not speculate.>

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | X.XX | Y.YY ✅/❌ |
| timesteps_trained | N | M |

## What this tells us
<What can we conclude? Be honest about uncertainty.>

## Questions this opens up
- <Question 1 for Windows Claude to consider>
- <Question 2>

## Suggested next experiment
<One specific hypothesis to test next>
```

---

## Experiment History

| # | Name | mean_reward | Status |
|---|------|-------------|--------|
| 001 | baseline_quartic | TBD | 🔄 running |
