# Status -- Last Updated 2026-03-28

## Latest Result

**exp_067 -- No Logstd Clamp** completed 2026-03-28.
- Training: **29.99 mean reward** (matched exp_060's 28.02, still climbing at 32.01)
- Benchmark: **0 gates, 1.70s deterministic** (2.6x longer than exp_060's 0.66s)
- Stochastic: 2.42s, T=0.3: 1.78s — all improved over exp_060
- **Key finding:** removing the logstd clamp improves deployment stability without hurting training

**exp_064 -- Entropy Annealing** completed 2026-03-28.
- FAILURE: 7.78 mean reward (flat). ent_coef=0.03 was too high — confirmed by exp_067 ablation.

## The Pattern (exp_056-067)

| Exp | Train Reward | Benchmark Gates | Det. Flight | Key Change |
|-----|:---:|:---:|:---:|------|
| 060 | 28.02 | 0 | 0.66s | combined structural (clamped) |
| 064 | 7.78 | 0 | 0.52s | high entropy + no clamp (FAILED) |
| 067 | 29.99 | 0 | **1.70s** | no clamp only (BEST det. flight) |

## Current Observations

- Removing the logstd clamp does not hurt training and improves deterministic flight time 2.6x.
- ent_coef=0.03 prevents learning entirely at this reward scale.
- All experiments in the exp_056-067 range produce 0 benchmark gates despite 25-32 training reward.
- Longer flights (1.7-3s) in exp_067 suggest the unclamped mean is more stable but still not navigating to gates.

## Next Steps

1. **exp_065** — Periodic deterministic eval + best checkpoint (claimed by another agent)
2. **exp_066** — Asymmetric + entropy annealing (BLOCKED: uses ent_coef=0.03 which failed; periodic eval infra not yet available)
3. **Longer training with exp_067 config** — the unclamped policy is still climbing at budget end (32.01). A 7200s+ run may reveal whether gates are achievable with more steps.
4. **Investigate why 0 gates persist** — the policy flies for 1.7-3s without crashing but doesn't navigate to gates. May need reward introspection or trajectory visualization.
