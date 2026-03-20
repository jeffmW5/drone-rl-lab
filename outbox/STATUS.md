# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_033** -- Truncation fix (Pardo et al. 2018) + PBRS, fine-tuned from exp_032
- **Training:** Mean reward 14.37 ± 4.11, 8M steps, 3143s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 24.58s** (4/5 hover @ 29.98s, 1/5 crash @ 2.98s)
- **Finding:** Truncation fix dramatically restored hover stability (exp_032 was 2.92s crash-heavy). The improved value estimates made the model more conservative. But still 0 gates — back in hover trap.
- **Key insight:** The problem is binary: either the model crashes trying to navigate (exp_028/031/032) or hovers safely but never moves (exp_026/029/030/033). We need a reward structure or curriculum that teaches *stable* lateral movement, not just "move or die."

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 026 | 9.77 | 0 | 28.8s | Stable hover baseline |
| 028 | 16.95 | 0.2 | 0.94s | speed_coef=1.0 (first gate!) |
| 029 | 16.52 | 0 | 29.98s | speed_coef=0.4 (hover) |
| 030 | 15.64 | 0 | 25.98s | speed_coef=0.55 (edge) |
| 031 | 10.44 | 0 | 2.02s | speed_coef=0.70 (crash) |
| 032 | 11.87 | 0 | 2.92s | PBRS progress (crash) |
| **033** | **14.37** | **0** | **24.58s** | **Truncation fix (hover restored)** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_029 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-033
- Awaiting next orchestrator instructions
