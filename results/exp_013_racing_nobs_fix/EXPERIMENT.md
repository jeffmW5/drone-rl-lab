# Experiment 013 — Racing n_obs Fix (Level 0)

## What we changed
Retrained Level 0 with `n_obs=2` (observation history stacking) after fixing the bug where `train_racing.py` wasn't passing `n_obs` to `make_envs()`. Same hyperparameters as exp_010 otherwise. Config: `configs/exp_013_racing_nobs_fix.yaml`.

## Why (the RL concept)
Observation stacking (`n_obs=2`) gives the agent access to 2 frames of trajectory history (position, quaternion, velocity, angular velocity). This should help the agent infer velocity and acceleration from position history, potentially enabling smoother and faster flight. The reference model uses n_obs=2.

## Results
| Metric | exp_010 (n_obs=0) | This experiment (n_obs=2) |
|--------|-------------------|--------------------------|
| mean_reward | 7.359 +/- 0.014 | 5.022 +/- 0.199 |
| timesteps_trained | 499,712 | 297,472 |
| wall_time | 588.8s | 604.7s |
| obs_shape | (47,) | (73,) |
| Level 0 sim (gates) | 4/4 (13.36s) | 0-1/4 (crashes) |

## What this tells us
The n_obs=2 fix **backfired** in this configuration:

1. **Larger obs space = slower training**: obs_shape went from 47 to 73 (+55%), making each iteration slower. Only 297k of 500k target steps completed within the 600s budget.
2. **Severely undertrained**: The agent hit the time budget at iteration 582/976. The reward was still climbing (5.03 at cutoff), suggesting it needed significantly more training time.
3. **Sim failure**: Despite reward=5.02, the agent can't navigate gates — it crashes or misses on every Level 0 run. The reward metric from training (random trajectory following) doesn't directly translate to gate-passing ability at this training stage.
4. **The reference model was trained with 1024 envs and likely much more compute** — our 64 envs + 600s budget is not enough for the larger observation space.

## Questions this opens up
- Would 1200s budget (or 1M steps) allow n_obs=2 to converge?
- Should we increase num_envs to 128+ to get more samples per wall-clock second?
- Is n_obs=2 actually needed, or is n_obs=0 sufficient with better hyperparams?
- The exp_010 model (n_obs=0) was competitive — maybe focus on tuning that instead?

## Suggested next experiment
Either: (A) retrain n_obs=2 with 1200s budget + 128 envs, or (B) abandon n_obs=2 and iterate on the working n_obs=0 approach with more timesteps.
