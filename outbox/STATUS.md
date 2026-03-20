# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_030** (speed_coef=0.55): Reward 15.64, 0 gates, avg 25.98s (4/5 hover, 1/5 crash — edge of transition)
- **exp_031** (speed_coef=0.70): Reward 10.44, 0 gates, avg 2.02s (hover destroyed, no navigation)
- **Conclusion:** Sharp phase transition between 0.55→0.70 with NO navigation sweet spot. exp(-k*dist) proximity reward fundamentally flawed — rewards being near gate, not moving toward it.

## Speed Coefficient Map (complete)
| speed_coef | LR | Behavior | Flight Time | Gates |
|-----------|------|----------|-------------|-------|
| 0.1 (exp_026) | 0.0015 | Hover only | 28.8s | 0 |
| 0.4 (exp_029) | 0.0003 | Hover only | 29.98s | 0 |
| 0.55 (exp_030) | 0.0001 | Edge: 4/5 hover, 1/5 crash | 25.98s | 0 |
| **0.70 (exp_031)** | **0.0001** | **Crash, no navigation** | **2.02s** | **0** |
| 1.0 (exp_028) | 0.0005 | Navigate + crash | 0.94s | 0.2 |

## In Progress
- **exp_032** -- PBRS delta-progress reward (progress_coef=50, speed_coef=0.3). Fine-tune from exp_026.
- Code change: `max(prev_dist - curr_dist, 0) * progress_coef` replaces `exp(-k*dist)`
- This rewards movement toward gate (not proximity) — hover gets ~0 progress per step

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_029 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-031
- In progress: exp_032 (PBRS progress reward)
- Queued: exp_033 (truncation fix)
