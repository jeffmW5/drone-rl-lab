# Status -- Last Updated 2026-03-19

## Last Completed
- **exp_025b** -- RaceCoreEnv with altitude reward + 25-step grace period
- **Training:** Mean reward 10.79 ± 2.64, 7.45M steps (budget: 5400s). Peak reward 15.84 at 6.6M steps.
- **Benchmark L2:** 0/5 finishes, 0 gates, avg 1.16s crash time (1.10-1.28s range)
- **Progress vs prior:** Model now modulates thrust (reduces from 0.98→0.32 near z=1.1), but momentum
  carries it past z=1.5 to z=2.3+ where env terminates. Previous models applied max thrust throughout.
- **Key issue:** Training wrapper hard-terminates at z=1.5, but benchmark env only terminates at z=2.5
  (pos_limit_high). Model learns to fear z=1.5 but approaches too fast to stop. The training/benchmark
  gap is now about momentum, not about altitude awareness.
- **Grace period:** Fixed to 25 steps (0.5s) — enough for takeoff, much less than buggy 2.0s.

## Experiment Summary (exp_022-025b)
| Exp | Reward | Gates | Crash Time | Issue |
|-----|--------|-------|------------|-------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty, 2.0s grace bug |
| 023 | 10.14 | 0 | ~2.0s | Soft OOB penalty too weak |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust throughout |
| 024 | ~0.3 | - | - | proximity_coef=1 too weak, plateaued |
| 025 | 3.41 | 0 | 0.22s | 10-step grace kills on ground contact |
| 025b | 10.79 | 0 | 1.16s | Altitude reward works (thrust modulation!) but overshoots |

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv):** exp_025b -- 0/5 finishes, 0 gates (best altitude control so far)

## Queue Status
- Completed: exp_022-025b (altitude control improving but not yet passing gates)
- Pod d54yx9n4s9i9k4 idle, ready for next experiment
- **Next ideas:**
  - Lower z_high to 1.3 (just above highest gate at 1.2) to force earlier braking
  - Add vertical velocity penalty: -coef * max(vz, 0) when z > 0.5
  - Increase gamma (0.94→0.97) so model values future survival more
  - Curriculum: train on level0 first (fixed gates, easier geometry)
