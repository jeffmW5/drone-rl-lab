# Experiment 064 -- Entropy Annealing (High ent_coef, No Logstd Clamp, 2x Budget)

## What we changed
Three changes from exp_060 baseline:
1. Removed `max_logstd=-1.0` clamp — std fully learnable
2. Increased `ent_coef` from 0.01 to 0.03, with `ent_coef_final=0.001` (linear decay)
3. Doubled budget from 3600s to 7200s

Config: `configs/exp_064_entropy_annealing.yaml`

## What was held constant
- body_frame_obs=true, bilateral_progress=true, progress_coef=50, soft_collision=true
- Network architecture (2×64 MLP), num_envs=512, num_steps=8
- All other reward terms identical to exp_060
- Mid-air spawn parameters (random_gate_start, spawn_offset=0.75)

## Why (the RL concept)
Research (2405.20250, 2512.18336) suggests: start with high entropy for broad exploration, then anneal to let the mean converge. The max_logstd=-1.0 clamp in exp_060 forced narrow distributions from the start, potentially preventing the mean from finding the navigation mode before the distribution was constrained. Removing the clamp plus higher initial entropy should allow fuller exploration early, with annealing converging the mean later.

## Results
| Metric | exp_060 (baseline) | This experiment |
|--------|:---:|:---:|
| mean_reward | 28.02 | 7.78 |
| timesteps_trained | ~1.5M | 2.5M |
| benchmark gates (det.) | 0 | 0 |
| benchmark flight (det.) | 0.66s | 0.52s |
| benchmark flight (stoch.) | 1.67s (exp_061) | 0.88s |

## Observations
- Training reward flat at ~8 from 200K steps to 2.5M steps. Never broke out.
- pg_loss approached zero by iter 500 (step 2M), suggesting near-zero policy gradient signal.
- Reward *declined* slightly from 8.3 to 7.5 in the second half of training.
- exp_060 reached 28.02 in fewer steps with the same reward definition, the only differences being the clamp and lower ent_coef.
- Deterministic benchmark: 0.52s uniform crash across all 5 runs (minimum possible flight time).
- Stochastic benchmark: 0.88s avg, 0 gates — also worse than exp_061 (1.67s with exp_060 model).

## Inference
The high ent_coef=0.03 without logstd clamp appears to have prevented learning entirely. The entropy bonus dominated the policy gradient signal, keeping the distribution too wide for the mean to converge on any useful behavior. The annealing from 0.03→0.001 over 9765 iterations meant that at budget-end (iter 614, 6.3% through), ent_coef was still ~0.028 — barely annealed. The 7200s budget only reached 6.3% of total_timesteps, so the annealing schedule was far too slow relative to the actual training duration.

**Confidence:** medium. The confound is that three things changed simultaneously (ent_coef, clamp removal, budget). The most likely explanation is that ent_coef=0.03 is too high for this reward scale, but we cannot cleanly separate the clamp removal effect.

## What this does NOT prove
- Does not prove entropy annealing is a bad idea in general — only that this specific schedule (0.03→0.001 over 40M steps, with only 2.5M actually trained) failed.
- Does not prove the logstd clamp removal is harmful — that change is confounded with the ent_coef increase.
- Does not prove the ent_coef=0.03 starting value is universally too high — with longer training or faster annealing, it might work.

## Next falsification test
Run with ent_coef=0.01 (same as exp_060), no logstd clamp, same 7200s budget. This isolates the clamp removal from the entropy increase. If reward still stays flat, then the clamp removal is the issue. If reward climbs like exp_060, then ent_coef=0.03 was the problem.

## Suggested next experiment
Clean ablation: exp_060 config with only `max_logstd` removed. Keep ent_coef=0.01, keep 3600s budget. Single-variable change to isolate the clamp effect.
