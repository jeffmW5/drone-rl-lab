# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_035** -- survive_coef=0 (remove hover anchor), fine-tune from exp_034
- **Training:** Mean reward 11.32 ± 5.52 (bimodal: peaks 24, dips 2-5), 8M steps, 4451s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.96s** (crashes instantly)
- **Finding:** survive_coef=0 destroys hover — same 0.96s crash as exp_028. alt_coef=1.5 alone insufficient for stability. survive_coef brackets: 0.5=hover, 0.0=crash. Training bimodality (peaks 24) shows the policy CAN navigate during exploration but the mean converges to crash mode.

## survive_coef Bracket
| survive_coef | Flight Time | Gates | Behavior |
|-------------|-------------|-------|----------|
| 0.5 (exp_034) | 29.98s | 0 | Hover trap |
| **0.15 (exp_036)** | **?** | **?** | **Testing...** |
| 0.0 (exp_035) | 0.96s | 0 | Crash |

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 028 | 16.95 | 0.2 | 0.94s | speed=1.0, old proximity (first gate!) |
| 033 | 14.37 | 0 | 24.58s | PBRS + truncation fix |
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover) |
| **035** | **11.32** | **0** | **0.96s** | **survive=0 (crash)** |

## In Progress
- **exp_036** -- survive_coef=0.15 (midpoint binary search), fine-tune from exp_034
- Tests whether there's a sweet spot between hover (0.5) and crash (0.0)
- If binary again → pivot to curriculum (shorter episodes) or architecture changes

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-035
- In progress: exp_036 (survive_coef=0.15 binary search)
