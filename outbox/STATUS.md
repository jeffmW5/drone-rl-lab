# Status -- Last Updated 2026-03-28

## Latest Result

**exp_068 -- Extended No-Clamp Training** completed 2026-03-28.
- Training: **42.84 mean reward** (all-time high, peak 44.53, still climbing at 3.2M steps)
- Deterministic benchmark: **0 gates, 1.67s avg** — unchanged from exp_067
- Stochastic/temperature: 3 gate passages in 45 runs — marginal improvement
- **Key finding:** more training reward does NOT proportionally improve deployment

## The Pattern (exp_067-068)

| Exp | Train Reward | Budget | Det. Gates | Det. Flight | Stoch/Temp Gates |
|-----|:---:|:---:|:---:|:---:|:---:|
| 067 | 29.99 | 3600s | 0/15 | 1.70s | 0/30 |
| 068 | 42.84 | 7200s | 0/15 | 1.67s | 3/30 |

Despite 43% higher training reward, deterministic benchmark is flat. The deployment gap is not a simple undertraining problem.

## Current Observations

- Removing logstd clamp improves benchmark stability (1.7s vs 0.66s) without hurting training.
- Training reward continues climbing with budget, but benchmark does not follow.
- The stochastic training policy navigates (42+ reward) but the deterministic mean does not.
- Sparse gate passages (3/45) suggest the policy is *near* navigating but not reliably.

## Next Steps

1. **Investigate what behavior produces 42+ reward** — the policy may be hovering/surviving rather than navigating in ways that translate to gates
2. **Larger network** — Swift uses 2×128, we use 2×64. May need more capacity for the mean to learn precise gate navigation
3. **Periodic deterministic eval during training** — needed to see if the mean ever navigates during training (exp_065 infrastructure, stalled)
4. **exp_066** — blocked on ent_coef fix (should use 0.01 not 0.03) and periodic eval infra
