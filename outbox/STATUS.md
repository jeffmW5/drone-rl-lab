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

## Next Steps

1. **Longer training for 2×128** — the larger network was still rapidly improving at budget end (peak 52.39 at iter 650/848). More steps could push further.
2. **Even larger network** — try 2×256 or 3×128 to see if capacity trend continues.
3. **Statistical significance** — run 50+ benchmark runs to confirm gate passage rates.
4. **Periodic deterministic eval** — still needed to understand when gates emerge during training.
