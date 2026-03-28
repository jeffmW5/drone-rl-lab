# Status -- Last Updated 2026-03-28

## Latest Result

**exp_064 -- Entropy Annealing** completed 2026-03-28.
- Training: **7.78 mean reward** (flat at ~8 throughout 2.5M steps)
- Benchmark: **0 gates, 0.52s deterministic / 0.88s stochastic**
- ent_coef=0.03 + no logstd clamp prevented learning entirely
- Annealing schedule too slow — only 6.3% through at budget end

## The Pattern (exp_056-064)

| Exp | Train Reward | Benchmark Gates | Flight Time | Key Change |
|-----|:---:|:---:|:---:|------|
| 056 | 28.92 | 0 | 0.64s | bilateral progress |
| 057 | 9.78 | 0.2 | 0.63s | body_frame_obs (weak progress) |
| 058 | 37.84 | 0 | 1.22s | soft_collision |
| 059 | 32.50 | 0 | 0.79s | asymmetric critic |
| 060 | 28.02 | 0 | 0.66s | combined structural |
| 061 | — | 0 | 1.67s | stochastic deployment |
| 062 | — | 0.03 | 0.7-0.9s | temperature sweep |
| 064 | 7.78 | 0 | 0.52s | high entropy + no clamp |

## Current Bottleneck

Training-side improvements and deployment-side fixes both remain insufficient.
exp_064 shows ent_coef=0.03 with no logstd clamp prevents learning at this
reward scale. The confound between clamp removal and ent_coef increase is not
yet resolved.

## Next Steps

1. **exp_065** — Periodic deterministic eval + best checkpoint (claimed by another agent)
2. **exp_066** — Asymmetric + entropy annealing (BLOCKED: depends on exp_065 infrastructure, and uses ent_coef=0.03 which failed in exp_064)
3. **Clean ablation needed:** exp_060 with only max_logstd removed (keep ent_coef=0.01) to isolate clamp effect

## Operational Notes

- Pod `32eutll6uvb5um` stopped after exp_064 training.
- `ent_coef_final` and `DRONE_RL_NOISE_SCALE` features committed to lsy_drone_racing.
