# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_035** -- survive_coef=0 (remove hover anchor), fine-tune from exp_034
- **Training:** Mean reward 11.32 ± 5.52, 8M steps, 4451s. Strongly bimodal (peaks 24, dips 2-5).
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.96s** (crashes instantly)
- **Finding:** Without survive reward, model crashes in <1s — same as exp_028. survive_coef is necessary for stability but at 0.5 it anchors the hover trap. We've now bracketed survive_coef: 0.5=hover, 0.0=crash.

## Survive Coefficient Map
| survive_coef | PBRS | speed_coef | Flight Time | Gates | Behavior |
|-------------|------|-----------|-------------|-------|----------|
| 0.5 (exp_034) | yes | 0.7 | 29.98s | 0 | Hover trap |
| **0.0 (exp_035)** | **yes** | **0.7** | **0.96s** | **0** | **Crash** |
| 0.1-0.2? | yes | 0.7 | ? | ? | **Sweet spot?** |

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 028 | 16.95 | 0.2 | 0.94s | speed=1.0, old proximity (first gate!) |
| 033 | 14.37 | 0 | 24.58s | PBRS + truncation fix |
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover) |
| **035** | **11.32** | **0** | **0.96s** | **survive=0 (crash)** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-035
- Awaiting next orchestrator instructions
