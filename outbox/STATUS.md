# Status -- Last Updated 2026-03-25

## Last Completed
- **exp_049** -- survive=0.08 + tight logstd (binary search)
- **Training reward**: 21.017 ± 1.098 (FINAL) — peaked at 38.20 (iter 690) then collapsed
- **Benchmark L2**: 0 gates, 0.80s avg flight (WORSE than exp_046's 1.3s)
- **Finding:** survive=0.08 causes post-breakout instability. Training peaks then collapses
  (v_loss spikes to 608). Survive binary search complete: 0.05 is optimal.

## In Progress
- **exp_052** -- Action smoothness penalties + tight logstd (max_logstd=-1.0)
  - Hypothesis: exp_046's 1.3s crash is unstable control near gate. d_act_th=0.4, d_act_xy=0.5, rpy=0.06 stabilize flight.
  - Training on RunPod, PID 1666050, started ~08:17 UTC

## Tight Logstd Series (exp_045-052)
| Exp | Key Change | Train Reward | Flight Time | Result |
|-----|-----------|-------------|-------------|--------|
| 045 | max_logstd=0.5 | 26.50 | 0.7-2.2s | std=1.65 still too wide |
| **046** | **max_logstd=-1.0, survive=0.05** | **29.20 (peak 39)** | **1.2-1.6s** | **BEST — flies toward gate** |
| 047 | survive=0.15 | 10.01 | 0.76s | Hover trap (survive×1500=225) |
| 048 | short episodes (200 steps) | 18.88 | 0.54s | Too aggressive |
| 049 | survive=0.08 | 21.02 (peak 38.2) | 0.80s | Unstable — v_loss collapse |
| 050 | gate_bonus=100, survive=0.06 | — | — | QUEUED |
| 051 | num_steps=64, gate_bonus=50 | — | — | QUEUED |
| 052 | action smoothness penalties | training... | ? | IN PROGRESS |

## Key Insight
**survive_coef binary search COMPLETE:** 0.05 is optimal. Higher values either hover-trap (0.15)
or cause post-breakout instability (0.08). The path forward is stabilizing exp_046's 1.3s
flight to reach the gate, not increasing survive. Action smoothness (exp_052) and longer
rollouts (exp_051) are the two most promising approaches.

## Current Best
- **Racing L2 (benchmark):** exp_046 -- 0 gates, **1.3s consistent flights toward gate**
- **Racing L2 (training reward):** exp_049 -- peak 38.20 (but collapsed); exp_046 -- 29.20 stable
- **Racing L2 (lap time, legacy):** exp_016 -- 13.49s, 2/10 finishes

## Queue Status
- Completed: exp_022-049
- In progress: exp_052 (training on RunPod)
- Queued: exp_050 (big gate bonus), exp_051 (longer rollouts)
