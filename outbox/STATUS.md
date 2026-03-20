# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_034** -- PBRS + speed_coef=0.7, fine-tune from exp_033
- **Training:** Mean reward 17.26 ± 5.50, 8M steps, 3299s. Peaks to 24.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 29.98s** (perfect hover, all 5 runs)
- **Finding:** PBRS successfully eliminated the crash at speed_coef=0.7 (exp_031 was 2.02s with old proximity). But the deterministic policy still hovers — high training reward came from stochastic exploration finding gates, not from the learned mean policy navigating. The hover optimum persists because alt_coef(1.5)+survive_coef(0.5)=2.0/step is a guaranteed baseline, while progress/speed reward requires risky lateral movement.

## Key Insight: The Hover Trap Is a Policy Optimization Problem
The reward signal is correct (PBRS rewards approach, not proximity), but the **policy mean** never shifts to lateral movement because:
1. Hover gives guaranteed 2.0/step (alt+survive)
2. Progress/speed only fire when stochastic noise pushes the drone toward a gate
3. The policy gradient from occasional exploration events isn't strong enough to shift the mean

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 026 | 9.77 | 0 | 28.8s | Stable hover baseline |
| 028 | 16.95 | 0.2 | 0.94s | speed_coef=1.0, old proximity (first gate!) |
| 031 | 10.44 | 0 | 2.02s | speed_coef=0.7, old proximity (crash) |
| 032 | 11.87 | 0 | 2.92s | PBRS progress, speed=0.3 (crash) |
| 033 | 14.37 | 0 | 24.58s | + truncation fix (hover restored) |
| **034** | **17.26** | **0** | **29.98s** | **PBRS + speed=0.7 (hover, no crash!)** |

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-034
- Awaiting next orchestrator instructions
