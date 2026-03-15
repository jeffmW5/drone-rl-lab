# INBOX — Instructions for Linux Claude

> This file is written by Windows Claude. Read this for current instructions.

---

## Context

We now have Kaggle competition times to beat. The WS25 Level 2 leaderboard:
- 1st: 3.39s, 2nd: 4.89s, 3rd: 5.02s
- **Our target: sub-5s on Level 2** (top 3 territory)
- Our exp_010 on Level 0: 13.36s (competitive with default controllers)

A bug was found and fixed: `train_racing.py` now passes `n_obs` to `make_envs()`.
Previously our agent trained with `n_obs=0` (no observation stacking) while the
reference model uses `n_obs=2`. This should improve performance.

---

## Current Task: Retrain with n_obs fix + Level 1/2 Benchmark

### Phase 1 — Retrain exp_010 with the bug fix

Create `configs/exp_013_racing_nobs_fix.yaml`:

```yaml
name: exp_013_racing_nobs_fix
backend: racing
hypothesis: "Retrain Level 0 with n_obs=2 fix — observation stacking gives the agent trajectory history, should close the 0.024s gap vs reference model"
budget_seconds: 600
racing:
  level: level0
  total_timesteps: 500000
  num_envs: 64
  num_steps: 8
  learning_rate: 0.0015
  anneal_lr: true
  gamma: 0.94
  gae_lambda: 0.97
  update_epochs: 10
  num_minibatches: 8
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  max_grad_norm: 1.5
  n_obs: 2
  rpy_coef: 0.06
  cuda: false
  seed: 42
```

Run it:
```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab
git pull   # get the n_obs fix in train_racing.py
python train.py configs/exp_013_racing_nobs_fix.yaml
```

### Phase 2 — Full controller benchmark on Levels 0, 1, 2

After exp_013 trains, run ALL controllers on Levels 0, 1, and 2.
Use our NEW exp_013 model (not the old exp_010).

Create/update `attitude_rl_exp013.py` (same as exp_010 version but pointing to exp_013 checkpoint):
```bash
cd /media/lsy_drone_racing
cp lsy_drone_racing/control/attitude_rl_exp010.py lsy_drone_racing/control/attitude_rl_exp013.py
# Edit: change model path to /media/drone-rl-lab/results/exp_013_racing_nobs_fix/model.ckpt
```

Then run for each level (0, 1, 2), 5 runs each:
```bash
# For each LEVEL in level0, level1, level2:
# For each CONTROLLER in state_controller, attitude_controller, attitude_rl, attitude_rl_exp013:

python scripts/sim.py --config <LEVEL>.toml --controller <CONTROLLER>.py --n_runs 5
```

That's 4 controllers x 3 levels x 5 runs = 60 sim runs total. Should take ~10-15 min.

### Phase 3 — Write results

Write `outbox/benchmark_levels.md` with a table like:

```markdown
# Full Benchmark — Levels 0, 1, 2

## Level 0 (perfect knowledge)
| Controller | Avg Time (s) | Best (s) | Finished | Gates |
|-----------|:------------:|:--------:|:--------:|:-----:|
| state_controller | ... | ... | .../5 | 4/4 |
| attitude_controller | ... | ... | .../5 | 4/4 |
| attitude_rl (theirs) | ... | ... | .../5 | 4/4 |
| attitude_rl_exp013 (ours) | ... | ... | .../5 | 4/4 |

## Level 1 (randomized physics)
<same table>

## Level 2 (randomized physics + gates) — COMPETITION LEVEL
<same table>

## Analysis
- How does exp_013 (with n_obs fix) compare to exp_010?
- Which level causes the biggest performance drop?
- Where do we stand vs the Kaggle target of sub-5s?
- What's the bottleneck: more training, better hyperparams, or harder architecture?
```

### Phase 4 — Git commit

```bash
cd /media/drone-rl-lab
git add -A
git commit -m "exp_013: retrain with n_obs fix + Level 0/1/2 benchmark"
git push
```

### Important Notes
- The `n_obs` fix is already in `train_racing.py` — just `git pull`
- Do NOT modify `train_racing.py` — all params go through the config
- If Level 2 controllers crash a lot, that's expected and useful data
- If `attitude_rl.py` (their pretrained) fails, skip it and note why
- The sim comparison for exp_010 is done — this task is about exp_013 + multi-level benchmark
