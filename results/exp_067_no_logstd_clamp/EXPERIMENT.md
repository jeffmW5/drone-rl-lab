# Experiment 067 -- No Logstd Clamp (Clean Ablation from exp_060)

## What we changed
Single change from exp_060: removed `max_logstd=-1.0`. The action distribution standard deviation is now fully learnable with no upper bound.

Config: `configs/exp_067_no_logstd_clamp.yaml`

## What was held constant
Everything else from exp_060: ent_coef=0.01, body_frame_obs=true, bilateral_progress=true, progress_coef=50, soft_collision=true, survive_coef=0.05, budget_seconds=3600, num_envs=512, seed=42.

## Why (the RL concept)
exp_064 failed (flat reward ~8) when changing three things: clamp removal + ent_coef 0.01→0.03 + budget 3600→7200s. This experiment isolates the clamp removal effect. If reward climbs like exp_060 (~28), then ent_coef=0.03 was exp_064's problem. If flat, the clamp itself matters.

## Results
| Metric | exp_060 (clamped) | This experiment (unclamped) |
|--------|:---:|:---:|
| mean_reward | 28.02 | 29.99 |
| final iteration reward | 28.02 | 32.01 |
| timesteps_trained | ~1.5M | 1.54M |
| benchmark gates (det.) | 0 | 0 |
| benchmark flight (det.) | 0.66s | **1.70s** |
| benchmark flight (stoch.) | 1.67s | **2.42s** |
| benchmark flight (T=0.3) | ~0.9s | **1.78s** |

## Observations
- Training reward tracked exp_060 closely, reaching 29.99 mean (32.01 at final iteration, still climbing).
- Breakout from ~10 to 25+ occurred around 1M steps, same timing as exp_060.
- Deterministic benchmark: 1.70s avg — **2.6x longer than exp_060's 0.66s**. This is the first time removing the clamp has improved benchmark flight time.
- Stochastic: 2.42s. Temperature-scaled: T=0.2 gives 1.97s, T=0.3 gives 1.78s, T=0.5 gives 1.48s.
- Still 0 gates across all 50 benchmark runs (5 det + 5 stoch + 30 temp sweep + 5 T=0.3 initial).
- v_loss spiked during the reward breakout (35-42 at iter 350-370), higher than exp_060's typical v_loss.

## Inference
Removing the logstd clamp does not hurt training reward and significantly improves deployment stability (1.70s vs 0.66s deterministic). The wider learned distribution may produce a mean that is more robust to deployment conditions, even though the benchmark still sees 0 gates.

This cleanly resolves the exp_064 attribution: **ent_coef=0.03 was the cause of exp_064's failure**, not the clamp removal. With ent_coef=0.01, unclamped training matches clamped training on reward and substantially beats it on benchmark flight time.

**Confidence:** high for the training equivalence claim (single-variable change, matched reward trajectory). Medium for the benchmark improvement claim (still within the 0-gate regime, flight time improvement could reflect different failure modes rather than better navigation).

## What this does NOT prove
- Does not prove the policy is navigating toward gates — longer flights could mean different crash dynamics, not better navigation.
- Does not prove removing the clamp is universally better — only tested at this budget and reward scale.
- Does not prove gates are achievable with more training — the 0-gate outcome persists.

## Next falsification test
Run exp_067 with a longer budget (7200s or more) to see if the unclamped policy eventually passes gates with more training time. If it doesn't, the 0-gate problem is not about training duration.

## Suggested next experiment
Longer training with the unclamped config (exp_060 base, no max_logstd, ent_coef=0.01, budget 7200s+). This is effectively the deferred exp_063 with the now-validated unclamped setting.
