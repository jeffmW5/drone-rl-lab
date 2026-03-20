# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_027b** -- RaceCoreEnv with 50/50 random gate mix (from scratch)
- **Training:** Mean reward 10.79 ± 1.44, 4.87M/8M steps (budget: 5400s).
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 1.18s flight time** — REGRESSION
- **Root cause:** Same as exp_027. Mid-air random gate envs dominate reward signal even at 50%.
  Model learns max-thrust behavior from mid-air spawns and never learns ground hover/takeoff.
- **Lesson:** Hard rule #22 — random gate starts from scratch don't work. Must fine-tune.

## In Progress
- **exp_027c** -- Fine-tune exp_026 (stable hover) with 50% random gate starts
- **Key difference:** Loads exp_026 checkpoint + lower LR (0.0005 vs 0.0015)
- **Hypothesis:** Preserves hover skill while learning gate approach on top

## Experiment Summary (exp_022-027b)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 024 | ~0.3 | - | - | Plateaued, killed |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |
| 027 | 11.67 | 0 | 1.1s | 100% random gate — no takeoff |
| 027b | 10.79 | 0 | 1.18s | 50/50 mix from scratch — still no takeoff |

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv flight):** exp_026 -- 0/5 finishes, 0 gates, **28.8s stable hover**

## Queue Status
- Completed: exp_022-027b
- In progress: exp_027c (fine-tune exp_026 with random gates)
