# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_029** -- Balanced Speed Reward (speed_coef=0.4, proximity_coef=1.0, survive_coef=0.5)
- **Training:** Mean reward 16.52 ± 2.18, 8M steps, 3031s. Very stable (v_loss < 0.05 typical).
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 29.98s** (max episode = perfect hover)
- **Finding:** speed_coef=0.4 is still in the "hover-only" regime. Model learned to hover even better than exp_026 (29.98s vs 28.8s) but never moved toward gates. The lower LR (0.0003) and higher survive_coef (0.5) reinforced stability at the expense of exploration.
- **Key insight: phase transition.** The behavior jumps sharply between hover (speed_coef ≤ 0.4) and navigation-crash (speed_coef = 1.0). Next experiment should try speed_coef=0.7 to probe the midpoint.

## Experiment Summary (exp_022-029)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |
| 027 | 11.67 | 0 | 1.1s | 100% random gate — no takeoff |
| 027b | 10.79 | 0 | 1.18s | 50/50 mix from scratch — no takeoff |
| 027c | 6.34 | 0 | 3.2s | Fine-tune + random gates — hover destroyed |
| 028 | 16.95 | 0.2 | 0.94s | First gate pass! But hover destroyed |
| **029** | **16.52** | **0** | **29.98s** | **Perfect hover, no navigation** |

## Speed Coefficient Map
| speed_coef | Behavior | Flight Time | Gates |
|-----------|----------|-------------|-------|
| 0.1 (exp_026) | Hover only | 28.8s | 0 |
| 0.4 (exp_029) | Hover only | 29.98s | 0 |
| 1.0 (exp_028) | Navigation + crash | 0.94s | 0.2 |
| **0.7 (next?)** | **Unknown — probe midpoint** | **?** | **?** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_029 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-029
- Next: Probe speed_coef=0.7 (midpoint of phase transition)
