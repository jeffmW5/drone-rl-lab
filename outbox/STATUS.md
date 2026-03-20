# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_027c** -- Fine-tune exp_026 with 50% random gate starts + LR=0.0005
- **Training:** Mean reward 6.34 ± 0.85, 6.99M steps (budget: 5400s).
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 3.2s flight time**
- **Partial improvement:** Takeoff partially preserved (3.2s vs 1.1s scratch), but hover destroyed (vs 28.8s exp_026).
- **Conclusion:** Random gate starts are a dead end. All 3 variants failed (027/027b/027c). The mid-air reward signal fundamentally conflicts with ground-start hover skills.

## In Progress
- **exp_028** -- Fine-tune exp_026 with high speed reward (speed_coef=1.0, proximity_coef=0.5)
- **Key insight:** exp_026 hovers stably but speed_coef=0.1 gives negligible lateral incentive. 10x speed reward should teach horizontal navigation without disrupting spawn logic.
- **No random gates** — preserves hover + adds lateral movement

## Experiment Summary (exp_022-027c)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |
| 027 | 11.67 | 0 | 1.1s | 100% random gate — no takeoff |
| 027b | 10.79 | 0 | 1.18s | 50/50 mix from scratch — no takeoff |
| 027c | 6.34 | 0 | 3.2s | Fine-tune + random gates — hover destroyed |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_026 -- 0/5 finishes, 0 gates, **28.8s stable hover**

## Queue Status
- Completed: exp_022-027c (random gate approach exhausted)
- In progress: exp_028 (high speed reward, no random gates)
