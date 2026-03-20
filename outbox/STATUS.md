# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_036** -- survive_coef=0.15 (binary search midpoint), fine-tune from exp_034
- **Training:** Mean reward 28.61 ± 0.22 (HIGHEST EVER, very stable final 100 iters), 8M steps, 4795s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.93s** (crashes instantly)
- **Finding:** survive_coef=0.15 crashes like 0.0 — bracket narrows to [0.15, 0.5]. Training reward (28.61) was misleading: stochastic exploration achieved high rewards but the deterministic mean policy crashes. The training was bimodal early (7-29 range) but converged to ~28-29 late, suggesting the stochastic policy found a high-reward trajectory that the mean policy can't reproduce. This is a **policy mode collapse** problem, not a reward shaping problem.

## survive_coef Bracket (UPDATED)
| survive_coef | Flight Time | Gates | Training Reward | Behavior |
|-------------|-------------|-------|-----------------|----------|
| 0.5 (exp_034) | 29.98s | 0 | 17.26 | Hover trap |
| **0.15 (exp_036)** | **0.93s** | **0** | **28.61** | **Crash (highest reward!)** |
| 0.0 (exp_035) | 0.96s | 0 | 11.32 | Crash |

## Key Insight
Training reward is **inversely correlated** with benchmark performance for survive_coef < 0.5.
The stochastic policy exploits speed/progress rewards during exploration but the deterministic
mean policy can't maintain flight. This suggests the problem needs:
1. Higher entropy (ent_coef) to keep the mean policy closer to the exploration distribution
2. Curriculum learning (shorter episodes, easier goals)
3. Architecture changes (e.g., separate hover/navigate heads)
4. Or try survive_coef=0.3 to narrow the bracket further

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 028 | 16.95 | 0.2 | 0.94s | speed=1.0, old proximity (first gate!) |
| 033 | 14.37 | 0 | 24.58s | PBRS + truncation fix |
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover) |
| 035 | 11.32 | 0 | 0.96s | survive=0 (crash) |
| **036** | **28.61** | **0** | **0.93s** | **survive=0.15 (crash, highest reward!)** |

## In Progress
- None — awaiting orchestrator direction

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-036
- In progress: none
