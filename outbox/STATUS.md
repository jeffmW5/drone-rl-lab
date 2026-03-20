# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_027** -- RaceCoreEnv with 100% random gate starts (Swift-style)
- **Training:** Mean reward 11.67 ± 0.84, 4.575M/8M steps (budget: 5400s).
  Highest training reward yet — random gate spawns near gates are "easier" than ground starts.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 1.1s flight time** — REGRESSION
- **Root cause:** 100% random gate starts always spawn mid-air. Model never learns ground takeoff.
  At benchmark time, drone starts at z=0.01 and crashes immediately (like exp_025b).
- **Lesson:** Hard rule #21 — never use 100% random gate spawns. Must include ground starts.

## Experiment Summary (exp_022-027)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 024 | ~0.3 | - | - | Plateaued, killed |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |
| 027 | 11.67 | 0 | 1.1s | 100% random gate starts — no takeoff learned |

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv flight):** exp_026 -- 0/5 finishes, 0 gates, **28.8s stable hover**

## Queue Status
- Completed: exp_022-027 (exp_027 failed benchmark — no takeoff learned)
- Next: exp_027b (50/50 ground + random gate mix)
