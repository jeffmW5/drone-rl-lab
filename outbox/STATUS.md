# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_030** -- Speed sweep: speed_coef=0.55, LR=0.0001, fine-tune exp_026
- **Training:** Mean reward 15.64 ± 4.36, 8M steps, 3115s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 25.98s** (4/5 hover @ 29.98s, 1/5 crash @ 9.98s)
- **Finding:** Edge of phase transition! Hover is starting to destabilize (1 crash in 5 runs) but navigation hasn't emerged yet. Narrows transition zone to 0.55-1.0.

## Speed Coefficient Map (updated)
| speed_coef | LR | Behavior | Flight Time | Gates |
|-----------|------|----------|-------------|-------|
| 0.1 (exp_026) | 0.0015 | Hover only | 28.8s | 0 |
| 0.4 (exp_029) | 0.0003 | Hover only | 29.98s | 0 |
| 0.55 (exp_030) | 0.0001 | **Edge: 4/5 hover, 1/5 crash** | 25.98s | 0 |
| 1.0 (exp_028) | 0.0005 | Navigate + crash | 0.94s | 0.2 |
| **0.70 (exp_031)** | **0.0001** | **In progress** | **?** | **?** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_029 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-030
- In progress: exp_031 (speed_coef=0.70)
- Queued: exp_032 (PBRS progress reward), exp_033 (truncation fix)
