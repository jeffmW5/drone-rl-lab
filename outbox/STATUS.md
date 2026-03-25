# Status -- Last Updated 2026-03-25

## Last Completed
- **exp_052** -- Action smoothness + tight logstd
- **Training reward**: 45.219 ± 2.867 (ALL-TIME HIGH, peak 54.95)
- **Benchmark L2**: 0 gates, 1.19s avg flight (comparable to exp_046's 1.3s)
- **Finding:** Action smoothness inflates training reward (reduced penalties) but doesn't
  improve benchmark navigation. The 1.3s crash is NOT caused by action instability.
  The bottleneck is likely the 8-step rollout window.

## In Progress
- **exp_051** -- Longer rollouts (num_steps=64) + gate_bonus=50
  - Hypothesis: 64-step rollouts let GAE observe full flight trajectory + gate bonus
  - Training on RunPod, PID 1754832, started 09:20 UTC

## Tight Logstd Series (exp_045-052)
| Exp | Key Change | Train Reward | Flight Time | Result |
|-----|-----------|-------------|-------------|--------|
| 045 | max_logstd=0.5 | 26.50 | 0.7-2.2s | std=1.65 still too wide |
| **046** | **max_logstd=-1.0, survive=0.05** | **29.20 (peak 39)** | **1.2-1.6s** | **BEST BENCHMARK — flies toward gate** |
| 047 | survive=0.15 | 10.01 | 0.76s | Hover trap (survive×1500=225) |
| 048 | short episodes (200 steps) | 18.88 | 0.54s | Too aggressive |
| 049 | survive=0.08 | 21.02 (peak 38.2) | 0.80s | Unstable — v_loss collapse |
| 050 | gate_bonus=100, survive=0.06 | — | — | QUEUED |
| 051 | num_steps=64, gate_bonus=50 | training... | ? | IN PROGRESS |
| 052 | action smoothness penalties | 45.22 (peak 55) | 1.19s | High reward, same crash |

## Key Insights
1. **survive=0.05 is optimal** — binary search complete (0.05 stable, 0.08 unstable, 0.15 hover trap)
2. **Action smoothness is NOT the fix** — inflates training reward but benchmark unchanged
3. **8-step rollout window is the bottleneck hypothesis** — GAE can only observe 0.16s of a 1.3s flight. exp_051 (64 steps = 1.28s) should capture full trajectory

## Current Best
- **Racing L2 (benchmark):** exp_046 -- 0 gates, **1.3s consistent flights toward gate**
- **Racing L2 (training reward):** exp_052 -- 45.22 mean (ALL-TIME HIGH)
- **Racing L2 (lap time, legacy):** exp_016 -- 13.49s, 2/10 finishes

## Queue Status
- Completed: exp_022-049, exp_052
- In progress: exp_051 (training on RunPod)
- Queued: exp_050 (big gate bonus)
