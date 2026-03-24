# Status -- Last Updated 2026-03-24

## Last Completed
- **exp_041/042/043** -- Three reward variants without stability reward, random mid-air spawns
- **exp_041** (progress only): 7.742 reward, 0 gates, 0.52s crash
- **exp_042** (view=0.1): 7.737 reward, 0 gates, 0.52s crash
- **exp_043** (view*progress): 7.748 reward, 0 gates, 0.52s crash
- **Finding:** All three identical failures. Without survive/altitude reward, drone crashes in 0.5s from spawn drift. Progress/view reward alone cannot bootstrap flight — needs stability component.

## Approaches Attempted
1. **Reward tuning** (exp_026-037): survive, speed, PBRS sweeps — all firmly hover-or-crash
2. **Entropy regularization** (exp_038): destroys hover → crash, no navigation
3. **Short episodes** (exp_039): bistable (hover+crash), edge of transition, still 0 gates
4. **View+progress from mid-air** (exp_040): falling exploit (view=1.0 rewards facing gate while falling)
5. **Progress/view variants** (exp_041-043): all crash in 0.5s without survival incentive

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 039 | 20.50 | 0 | 18.35s | 300-step episodes (BISTABLE) |
| 040 | 7.75 | 0 | 0.52s | view+progress, falling exploit |
| 041 | 7.74 | 0 | 0.52s | progress only (no view) |
| 042 | 7.74 | 0 | 0.52s | view=0.1 + progress |
| 043 | 7.75 | 0 | 0.52s | view*progress multiplicative |

## In Progress
- Designing exp_044+ — need stability + progress without hover trap

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-043
- In progress: designing next batch
