# Status -- Last Updated 2026-03-28

## Latest Result

**exp_069 -- Larger Network (2×128)** completed 2026-03-28.
- Training: **42.29 mean reward** (peak 52.39 — new all-time high)
- Deterministic benchmark: **2/15 gates** (0.86s avg) — FIRST DETERMINISTIC GATES EVER
- T=0.3: **5/15 gates** (33% passage rate) — best gate rate ever
- **Key finding:** network capacity helps — 2×128 outperforms 2×64 on benchmark gates

## The Progression (exp_067-069)

| Exp | Hidden | Train Reward | Peak | Det. Gates | T=0.3 Gates | Total Gates |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| 067 | 2×64 | 29.99 | 32.01 | 0/15 | — | 0/50 |
| 068 | 2×64 | 42.84 | 44.53 | 0/15 | 2/15 | 3/45 |
| 069 | 2×128 | 42.29 | 52.39 | 2/15 | 5/15 | 7/45 |

Network capacity matters: same training reward level but better benchmark gates.

## Current Observations

- Removing logstd clamp improves benchmark stability without hurting training (exp_067).
- Doubling training budget improves reward but not benchmark (exp_068 vs 067).
- Larger network (128 vs 64) produces first deterministic gate passages and higher peak reward.
- T=0.3 temperature is the most consistent gate passage mode.
- The deployment gap persists but is narrowing with capacity.

## Research: Deployment Gap Literature Review (2026-03-28)

Literature review of 10+ papers targeting the stochastic-to-deterministic deployment gap.
See `research/deployment_gap_mean_policy.md` for full details.

**Key finding:** Two critical issues identified in our setup:
1. **No observation normalization** — the "What Matters" study (250K agents, ICLR 2021) identifies this as critical. Our Agent has NONE. Swift normalizes. SimpleFlight normalizes.
2. **No action smoothness penalty** — CAPS (ICRA 2021) and SimpleFlight both show action-difference penalties force the mean to be physically coherent. We have d_act_th/d_act_xy in the reward code but disabled (0.0) since exp_069.

**Theoretical explanation:** Montenegro et al. (ICML 2024) formally prove that PPO optimizes stochastic expected return — the mean is a side effect, not the objective. Without mechanisms to force mean quality (entropy annealing, smoothness penalties, obs normalization), the gap is expected.

## Next Steps

1. **exp_070 — Longer training for 2×128** (IN PROGRESS) — 14400s budget to see if capacity + time closes gap.
2. **exp_071 — Observation normalization** (READY) — single highest-priority intervention from literature.
3. **exp_072 — Action smoothness penalty** (READY) — re-enable d_act penalties; directly forces mean coherence.
4. **exp_073 — Entropy annealing** (READY) — anneal ent_coef 0.01→0.001 in last 30% to sharpen mean.
5. **exp_074 — Combined obs norm + smoothness** (READY) — if both help independently, combine.
6. **exp_065 — Periodic deterministic eval** (IN PROGRESS) — instrumentation for when gates emerge during training.
