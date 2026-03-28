# Experiment 069 -- Larger Network (2×128)

## What we changed
Single change from exp_068: hidden_size 64 → 128 (2×128 MLP for both actor and critic). This increases parameters from 16K to 48K, matching Swift's architecture.

Config: `configs/exp_069_larger_network.yaml`

## What was held constant
All exp_068 settings: ent_coef=0.01, no max_logstd clamp, body_frame_obs=true, bilateral_progress=true, progress_coef=50, soft_collision=true, survive_coef=0.05, num_envs=512, seed=42, budget_seconds=7200.

## Why (the RL concept)
Tests HYP-006 (network capacity). The deterministic mean policy consistently fails to navigate gates despite high training reward (42+ in exp_068). Swift (Nature 2023) uses 2×128 MLPs; our 2×64 may lack capacity for the mean to represent precise gate navigation in the 55D→4D mapping.

## Results
| Metric | exp_068 (2×64) | This experiment (2×128) |
|--------|:---:|:---:|
| mean_reward | 42.84 | **42.29** |
| peak_reward | 44.53 | **52.39** |
| timesteps_trained | 3.24M | **3.47M** |
| params | 15,817 | **48,009** |
| benchmark gates (det.) | 0/15 | **2/15** |
| benchmark flight (det.) | 1.67s | 0.86s |
| benchmark gates (stoch.) | 1/15 | 0/15 |
| benchmark gates (T=0.3) | 2/15 | **5/15** |
| total gates (all modes) | 3/45 | **7/45** |

## Observations
- Training reward was slower to grow (reached ~9 at 700K steps vs exp_068's ~11) but eventually matched the final mean (42.29 vs 42.84).
- Peak reward reached 52.39 — substantially higher than exp_068's 44.53. The larger network can achieve higher stochastic performance.
- v_loss was much more unstable (spikes to 500+) vs exp_068 (spikes to ~220). The value function struggles more with the larger architecture.
- **First deterministic gate passages in this family:** 2/15 runs passed a gate (runs 5 and 15 with 1.80s and 1.44s flight). Previous experiments (067, 068) had 0 deterministic gates.
- T=0.3 mode: 5/15 gates (33% passage rate) vs exp_068's 2/15 (13%). Most consistent gate passage mode.
- Stochastic mode: 0/15 — worse than exp_068's 1/15. Full noise disrupts the larger network's mean.
- Average deterministic flight time shorter (0.86s vs 1.67s) — the larger network is more aggressive but crashes faster when it doesn't navigate.

## Inference
The larger network produces the first deterministic gate passages in this experiment family and the highest gate passage rate at T=0.3. This supports HYP-006: network capacity was a limiting factor for the mean policy's ability to represent gate navigation.

However, the improvement is modest (2/15 det, 7/45 total vs 0/15 det, 3/45 total for exp_068). The deterministic mean still fails 87% of the time. The larger network helps but does not solve the deployment gap.

**Confidence:** medium. The gate passage improvement is directionally positive and consistent with the capacity hypothesis. The sample sizes are small (15 runs per mode), so the exact rates may be noisy, but the pattern of improvement across modes is meaningful.

## What this does NOT prove
- Does not prove 128 is the optimal size — larger (256, 512) or wider (3-layer) architectures might do better.
- Does not prove the improvement is solely from capacity — the different training dynamics (slower start, higher peak, more v_loss instability) could cause subtle changes in the final policy.
- Does not prove the 7/45 gate rate is reliably different from exp_068's 3/45 — a binomial test would need ~100+ runs per mode.
- Does not prove this approach will scale to full-course navigation.

## Next falsification test
1. Run with even larger network (2×256 or 3×128) to see if the capacity trend continues.
2. Run exp_069's config with longer budget (14400s) to give the slower-learning 128-unit network more time to converge.
3. Run 50+ deterministic benchmark runs to get statistically significant gate passage rates.

## Suggested next experiment
The 2×128 network needs more training steps (it was still improving rapidly at budget end, with peak reward at iter 650 of 848). A longer training budget (14400s or 28800s) with the 128-unit network could push the mean policy further. Alternatively, try 2×256 to see if capacity continues to be a factor.
