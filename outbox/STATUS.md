# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_036** -- survive_coef=0.15 (binary search), fine-tune from exp_034
- **Training:** Mean reward 28.61 ± 0.22 (HIGHEST EVER), 8M steps, 4795s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.93s** (crashes instantly)
- **Finding:** survive_coef=0.15 crashes like 0.0, DESPITE highest training reward. Training reward inversely correlated with benchmark: stochastic policy navigates (28.61) but deterministic mean crashes. This is policy mode collapse, not a reward problem.

## survive_coef Bracket
| survive_coef | Flight Time | Gates | Training Reward | Behavior |
|-------------|-------------|-------|-----------------|----------|
| 0.5 (exp_034) | 29.98s | 0 | 17.26 | Hover trap |
| **0.3 (exp_037)** | **?** | **?** | **?** | **Testing...** |
| 0.15 (exp_036) | 0.93s | 0 | 28.61 | Crash (highest reward!) |
| 0.0 (exp_035) | 0.96s | 0 | 11.32 | Crash |

## In Progress
- **exp_037** -- survive_coef=0.3 (final bracket step), fine-tune from exp_034
- LAST pure reward-tuning experiment. If binary → pivot to ent_coef or curriculum.

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-036
- In progress: exp_037 (survive_coef=0.3, final bracket step)
