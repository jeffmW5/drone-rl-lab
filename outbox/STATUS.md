# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_028** -- Fine-tune exp_026 with high speed reward (speed_coef=1.0, proximity_coef=0.5)
- **Training:** Mean reward 16.95 ± 5.33, 8M steps, 4642s wall time. Highest training reward ever.
- **Benchmark L2:** 0/5 finishes, **0.2 avg gates** (1 gate in run 4!), avg 0.94s flight time
- **Key finding:** speed_coef=1.0 taught navigation (first-ever gate passage!) but destroyed hover stability. Model zooms toward gates but crashes in <1s. The sweet spot is between 0.1 (no movement) and 1.0 (no hover).
- **Recommendation:** Try speed_coef=0.3-0.5 to balance hover stability with lateral incentive. Also consider reducing learning rate further to preserve more hover skill from exp_026.

## Experiment Summary (exp_022-028)
| Exp | Reward | Gates | Flight Time | Key Result |
|-----|--------|-------|-------------|------------|
| 022 | 10.46 | 0.5 avg | 2.02s | No altitude penalty |
| 023b | 6.42 | 0 | 0.79s | Hard OOB, max thrust |
| 025b | 10.79 | 0 | 1.16s | Thrust modulation but overshoots |
| **026** | **9.77** | **0** | **28.8s** | **Stable hover at gate height!** |
| 027 | 11.67 | 0 | 1.1s | 100% random gate — no takeoff |
| 027b | 10.79 | 0 | 1.18s | 50/50 mix from scratch — no takeoff |
| 027c | 6.34 | 0 | 3.2s | Fine-tune + random gates — hover destroyed |
| **028** | **16.95** | **0.2** | **0.94s** | **First gate pass! But hover destroyed** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_026 -- 0/5 finishes, 0 gates, **28.8s stable hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s (first gate pass!)

## Queue Status
- Completed: exp_022-028
- Next: Need balanced speed_coef (0.3-0.5 range) — exp_028 proved navigation is learnable
