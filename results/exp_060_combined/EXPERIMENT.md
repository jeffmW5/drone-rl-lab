# exp_060 — Combined (Body-Frame Obs + Soft Collision + Strong Progress)

## Hypothesis
Combines three structural changes that each showed partial promise:
- body_frame_obs (exp_057): gate positions in drone body frame
- soft_collision (exp_058): crashes don't terminate in phase 1, -5.0 penalty + respawn
- progress_coef=50 (exp_056): strong bilateral progress reward

Body-frame obs makes "gate is forward" consistent across spawns. Strong progress breaks hover.
Soft collision allows learning through crashes.

## Config
- `body_frame_obs: true`, `soft_collision: true`, `progress_coef: 50.0`
- `bilateral_progress: true`, `survive_coef: 0.05`, `max_logstd: -1.0`
- File: `configs/exp_060_combined.yaml`

## Training Results
- **Mean reward: 28.02** (peak ~29 at iter 370, still climbing at budget)
- Steps: 1,564,672 / 20,000,000 (budget-limited at 3603s)
- Wall time: 3603s on RTX 3090 (secure cloud, ~$0.39/hr)

### Reward curve
- iter 1-40: 10.4 → 11.7 (initial)
- iter 50-60: 5.4 → 4.8 (crash dip — soft collision exploration)
- iter 70-130: 9.0 → 9.5 (recovery)
- iter 130-250: 9.5 → 12.7 (slow climb)
- iter 250-370: 12.7 → 28.7 (breakout — strong upward trend, not plateaued)

## Benchmark Results (level2_midair)
| Run | Flight time (s) | Gates | Notes |
|-----|----------------|-------|-------|
| 1 | 0.52 | 0 | Crash |
| 2 | 0.74 | 0 | Crash |
| 3 | 0.78 | 0 | Crash |
| 4 | 0.76 | 0 | Crash |
| 5 | 0.52 | 0 | Crash |

**Average: 0.66s, 0 gates, 0 finishes**

## Diagnosis
1. **Training reward matches exp_056** — 28.02 vs 28.92, similar trajectory. The combined
   structural changes didn't hurt or significantly help training reward vs bilateral progress alone.
2. **Benchmark identical to exp_056** — 0.66s vs 0.64s crash. Body-frame obs + soft collision
   made zero difference at deployment.
3. **The stochastic-to-deterministic gap is THE bottleneck** — Training reward of 28+ means
   the stochastic policy navigates well. But the deterministic mean crashes every time.
   This has been consistent across exp_056, 057, 058, and 060.
4. **Reward was still climbing** — Budget cut training at 1.56M steps while reward was rising
   fast (28.7 at iter 370). Longer training might push reward higher but likely won't fix
   the deployment gap based on exp_056's 3.7M steps showing the same problem.

## Key Takeaway
The combined approach (body_frame_obs + soft_collision + strong progress) doesn't fix the
fundamental problem: the deterministic mean policy crashes. All recent experiments (056-060)
produce 25-38 training reward with 0 benchmark gates. The stochastic exploration finds good
trajectories but the mean never converges to a stable navigating mode.

**Next direction: fix the deterministic deployment**, not reward/obs/curriculum. Options:
- Deploy stochastic policy (sample from learned distribution instead of using mean)
- Reduce action variance at deployment (temperature scaling)
- Train with deterministic evaluation periods (so the mean is directly optimized for)
- Architecture change (e.g., mode-seeking loss, mixture policy)
