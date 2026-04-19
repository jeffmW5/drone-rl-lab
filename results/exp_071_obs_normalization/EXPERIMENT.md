# Experiment 071 -- Observation Normalization

## What we changed
Single change from `exp_069`: enabled running observation normalization.

Config: `configs/exp_071_obs_normalization.yaml`

Concrete diff vs `exp_069` baseline:
- `obs_normalize: true`
- `obs_norm_clip: 10.0`

## What was held constant
All other `exp_069` settings were held fixed:
- 2x128 MLP (`hidden_size: 128`)
- `cuda: true`, `num_envs: 512`, `num_steps: 8`
- no logstd clamp
- `body_frame_obs: true`
- `bilateral_progress: true`, `progress_coef: 50.0`
- `soft_collision: true`
- `survive_coef: 0.05`
- `budget_seconds: 7200`

## Why (the RL concept)
This tests whether poorly conditioned inputs are still limiting policy learning. The motivating hypothesis was that the actor mean may be harder to optimize when raw observation scales drift across dimensions and training phases. The intervention is intentionally narrow: normalize observations online without changing reward structure, model size, rollout size, or exploration settings.

## Results
| Metric | exp_069 | This experiment |
|--------|:---:|:---:|
| mean_reward | 42.29 | **45.63** |
| timesteps_trained | 3.47M | **4.63M** |
| wall_time | 7200s | 7200.8s |
| benchmark | 7 total gates in 45 runs | **not verified** |

Additional artifact status:
- `metrics.json`: present
- `benchmark.json`: present but contains zero parsed runs
- `provenance.json`: present

## Observations
- Training completed the full 7200s budget with no early stop.
- Final mean reward increased from `42.29` in `exp_069` to `45.627`.
- Total timesteps increased to `4,628,480`.
- A post-run benchmark attempt was executed, but the current generic benchmark path produced `benchmarks: []` with no parsed runs. That means deployment outcome is not measured for this experiment yet.
- Throughput measurement on the same pod/config showed rollout is env-step dominated:
  - `env_samples_per_s`: `803.54`
  - `rollout_samples_per_s`: `750.71`
  - `policy_samples_per_s`: `11549.66`
  - `ppo_update_samples_per_s`: `14443.32`

## Inference
Observation normalization improved training reward in this experiment family. That supports the narrower claim that optimization/training dynamics benefited under the current setup.

What it does **not** establish is the thing we actually care about for racing decisions: benchmark improvement. Because the benchmark path produced no parsed runs for this experiment, the deployment-side conclusion is still `not verified`.

**Confidence:** medium for "training reward improved", low for any deployment interpretation.

## What this does NOT prove
- Does not prove observation normalization improved deterministic gate passage.
- Does not prove observation normalization improved sim deployment at all.
- Does not prove the higher reward reflects better mean-policy quality rather than better stochastic training behavior.
- Does not prove `exp_071` should outrank `exp_069` for racing selection, because benchmark evidence is missing.

## Next falsification test
1. Fix or replace the current benchmark/controller path so `exp_071` can be evaluated with real parsed runs.
2. Re-run the benchmark on the finished `exp_071` checkpoint and compare directly against `exp_069`.
3. Only then decide whether observation normalization belongs in the forward line or should remain a training-only positive.

## Suggested next experiment
Before making strong queue decisions from `exp_071`, repair the benchmark path for this direct-racing line. If benchmarking is restored quickly, compare `exp_071` to `exp_069` first; otherwise proceed with the next clean intervention (`exp_072`) but keep `exp_071` benchmark status explicitly unresolved.
