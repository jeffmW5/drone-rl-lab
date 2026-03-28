# Status -- Last Updated 2026-03-28

## Latest Results

**exp_059 -- Asymmetric Actor-Critic** completed 2026-03-28.
- Training: **32.50 mean reward** in the 1-hour RunPod window
- Matched `level2_midair` benchmark: **0 gates, 0.79s avg**
- Better training signal than `exp_056`, but no deployment win

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
| 059 | 32.50 | 0 | 0.79s | asymmetric critic |
| 060 | 28.02 | 0 | 0.66s | all three combined |
| 061 | — | 0 | 1.67s | stochastic deployment |
| 062 | — | 0.03 | 0.7-0.9s | temperature sweep |

## Current Bottleneck

**Training-side improvements are still not translating into matched benchmark success.**
`exp_059` weakens any claim that the direct-racing line is "fully exhausted" on
training reward, because it did improve 1-hour training efficiency. But the
matched mid-air benchmark still stayed at 0 gates, so deployment quality remains
the controlling metric.

Deployment fixes (stochastic, temperature) also remain insufficient.

## Next Steps

Training-time fixes needed:
1. **exp_064 — Entropy annealing** (high→low ent_coef, no logstd clamp, 10M+ steps) — READY, needs GPU
2. **exp_063 — Extended training** (10M+ steps, no logstd clamp) — DEFERRED until 064 done

## Operational Notes

- Fresh RunPod setup and the new startup logging worked on a real fresh RTX 3090 pod.
- Asymmetric checkpoints were **not benchmarkable** through the old generic controller; actor-only loading support was required in `lsy_drone_racing/control/attitude_rl_generic.py`.
- For `random_gate_start` experiments, the matched evaluator is `scripts/sim_midair.py`, not the race-start benchmark.
