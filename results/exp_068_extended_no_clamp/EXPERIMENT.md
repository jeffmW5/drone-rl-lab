# Experiment 068 -- Extended No-Clamp Training (7200s)

## What we changed
Single change from exp_067: budget_seconds 3600 → 7200. Everything else identical.

Config: `configs/exp_068_extended_no_clamp.yaml`

## What was held constant
All exp_067/exp_060 settings: ent_coef=0.01, no max_logstd clamp, body_frame_obs=true, bilateral_progress=true, progress_coef=50, soft_collision=true, survive_coef=0.05, num_envs=512, seed=42.

## Why (the RL concept)
Tests HYP-002 (undertraining). exp_067 was still climbing steeply at budget end (32.01 reward). Doubling budget should show whether the policy continues improving and whether higher training reward translates to benchmark gates.

## Results
| Metric | exp_067 (3600s) | This experiment (7200s) |
|--------|:---:|:---:|
| mean_reward | 29.99 | **42.84** |
| peak_reward | 32.01 | **44.53** |
| timesteps_trained | 1.54M | **3.24M** |
| benchmark gates (det.) | 0/15 | 0/15 |
| benchmark flight (det.) | 1.70s | 1.67s |
| benchmark gates (stoch.) | — | 1/15 |
| benchmark gates (T=0.3) | — | 2/15 |

## Observations
- Training reward climbed from 10 → 36 in first 3600s (matching exp_067 trajectory), then continued to 42.84 mean / 44.53 peak at 3.2M steps. Still climbing at budget end.
- Reward growth slowed in the second half: ~36 at 1.5M → ~43 at 3.2M (7 point gain in 1.7M steps vs 26 point gain in first 1.5M).
- v_loss increased substantially in later training (30-220 range), suggesting value function instability.
- Deterministic benchmark: 1.67s avg, 0 gates in 15 runs — essentially identical to exp_067 despite 43 vs 30 training reward.
- Stochastic: 1 gate in 15 runs (1.05s avg). T=0.3: 2 gates in 15 runs (1.19s avg).
- Gate passages remain sparse: 3 total in 45 runs across all modes.

## Inference
Training reward continues climbing with more steps (42.84 vs exp_067's 29.99), supporting the hypothesis that the policy is undertrained. However, the benchmark gain from doubling training is marginal: deterministic flight time is unchanged (1.67s vs 1.70s) and gate passages are still near-zero (3/45 vs 0/50 for exp_067).

This weakens the simple "just train longer" story. The stochastic training policy is getting better at navigating (that's where the 43 reward comes from), but additional training reward does not translate proportionally into deterministic benchmark improvement. The deployment gap persists.

**Confidence:** medium. The training improvement is clear. The benchmark stagnation at 0 gates despite 50% higher reward is a meaningful negative signal against HYP-002 as a sufficient explanation.

## What this does NOT prove
- Does not prove training is fully converged — reward was still climbing.
- Does not prove longer training cannot eventually produce gates — we only doubled the budget, not 10x.
- Does not prove the sparse gate passages (3/45) are meaningful — could be noise.

## Next falsification test
Train for 4x budget (14400s, ~6M steps) with checkpoint saving. If deterministic benchmark still shows 0 gates with 50+ reward, then undertraining alone is not the solution.

## Suggested next experiment
The persistent 0-gate deterministic outcome despite high training reward points to a structural deployment problem. Consider: (1) periodic deterministic evaluation during training to see if the mean ever navigates, (2) reward introspection to understand what behavior produces 43 reward, or (3) much larger networks (Swift uses 2×128, we use 2×64).
