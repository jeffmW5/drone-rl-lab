# Status -- Last Updated 2026-03-19

## Last Completed
- **exp_026** -- RaceCoreEnv with vertical velocity penalty + tighter ceiling
- **Training:** Mean reward 9.77 ± 1.07, 8M steps (budget: 5400s, completed in 3893s).
  Very stable convergence — v_loss dropped to 0.002 by 7M steps.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 28.8s flight time** (4/5 survive full 30s episode!)
- **BREAKTHROUGH:** First model to achieve stable hovering at gate altitude (z≈0.72, gate 0 at z=0.70).
  Model ascends smoothly, brakes at z~1.2, oscillates briefly, then settles to hover at z=0.72-0.77.
  Thrust modulation: +0.89 → -0.52 (braking) → -0.12 (hover). Flight time 1.16s → 28.8s.
- **Remaining gap:** Model hovers at correct altitude but doesn't navigate horizontally to gates.
  It stays near the start position. Needs horizontal navigation incentive or random gate starts.

## Experiment Summary (exp_022-026)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 024 | ~0.3 | - | - | Plateaued, killed |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv flight):** exp_026 -- 0/5 finishes, 0 gates, **28.8s stable hover**

## Queue Status
- Completed: exp_022-026 (altitude control SOLVED, horizontal navigation needed)
- Next: exp_027 (random gate starts — spawn near gates to learn approach/passage)
- Pod d54yx9n4s9i9k4 ready for exp_027
