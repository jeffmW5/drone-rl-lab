# Status -- Last Updated 2026-03-28

## Latest Results

**exp_061 -- Stochastic Deployment** completed 2026-03-28.
- Benchmark: **0 gates, 1.67s avg** — 2.5x longer flight than deterministic (0.66s)
- Stochastic sampling stabilizes but too imprecise for gates

**exp_062 -- Temperature-Scaled Deployment** completed 2026-03-28.
- Swept T=0.1 to 1.0, 70 total runs
- Benchmark: **2 gate passages in 70 runs** — essentially noise
- No sweet spot found; deployment-time fixes are insufficient

## The Pattern (exp_056-062)

| Exp | Train Reward | Benchmark Gates | Flight Time | Change |
|-----|:---:|:---:|:---:|------|
| 056 | 28.92 | 0 | 0.64s | bilateral progress |
| 057 | 9.78 | 0.2 | 0.63s | body_frame_obs (weak progress) |
| 058 | 37.84 | 0 | 1.22s | soft_collision |
| 060 | 28.02 | 0 | 0.66s | all three combined |
| 061 | — | 0 | 1.67s | stochastic deployment |
| 062 | — | 0.03 | 0.7-0.9s | temperature sweep |

## Current Bottleneck

**Policy mean hasn't learned gate navigation.** The stochastic training policy
gets reward by randomly sampling actions that happen to move toward gates, but
the mean action doesn't converge to a useful navigation strategy. This is a
training problem, not a deployment problem.

Deployment fixes (stochastic, temperature) confirmed insufficient.

## Next Steps

Training-time fixes needed:
1. **exp_064 — Entropy annealing** (high→low ent_coef, no logstd clamp, 10M+ steps) — READY, needs GPU
2. **exp_059 — Asymmetric actor-critic** (privileged critic) — READY, needs GPU
3. **exp_063 — Extended training** (10M+ steps, no logstd clamp) — DEFERRED until 064 done
