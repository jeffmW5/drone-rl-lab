# Status -- Last Updated 2026-03-25

## BREAKTHROUGH: Root Cause Identified
**ALL experiments (046-052) crash at ~1.3s due to DOMAIN MISMATCH, not training issues.**

| Condition | Training | Benchmark |
|-----------|----------|-----------|
| Start position | 0.75m from gate, mid-air | [-1.5, 0.75, 0.01] (ground) |
| Start altitude | Gate altitude (~0.7m) | Ground level (0.01m) |
| Distance to gate 0 | 0.75m | 2.06m (XY) + 0.69m (climb) |

Three different approaches (survive tuning, action smoothness, longer rollouts) ALL produce
the same ~1.2-1.3s benchmark crash. Training rewards vary wildly (29-176) but benchmark is
stuck. The policy was never trained on the benchmark starting condition.

## In Progress
- **exp_053** -- spawn_offset=1.5 (2x current), spawn_pos_noise=0.3
  - Trains at 1.5m from gate (vs 0.75m), closer to benchmark's 2.06m distance
  - Training on RunPod, PID 1855701, started 10:26 UTC

## Queued
- **exp_054** -- random_gate_ratio=0.0 (NO random gate starts)
  - Trains from actual race start position, eliminates domain gap entirely
  - More aggressive test: ground takeoff + 2.06m navigation from scratch

## Results Summary (exp_046-052)
| Exp | Key Change | Train Reward | Benchmark Time | Gates |
|-----|-----------|-------------|----------------|-------|
| **046** | baseline tight logstd | 29.20 | **1.3s** | 0 |
| 047 | survive=0.15 | 10.01 | 0.76s | 0 |
| 048 | short episodes | 18.88 | 0.54s | 0 |
| 049 | survive=0.08 | 21.02 (peak 38.2) | 0.80s | 0 |
| 050 | gate_bonus=100 | — | — | — |
| 051 | num_steps=64 | 175.60 | **1.22s** | 0 |
| 052 | action smoothness | 45.22 (peak 55) | **1.19s** | 0 |
| 053 | spawn_offset=1.5 | training... | ? | ? |

## Current Best
- **Racing L2 (benchmark):** exp_046 -- 0 gates, 1.3s consistent flights toward gate
- **Racing L2 (training reward):** exp_051 -- 175.60 mean (per-step 2.74)
- **Racing L2 (lap time, legacy):** exp_016 -- 13.49s, 2/10 finishes

## Queue Status
- Completed: exp_022-052
- In progress: exp_053 (training on RunPod)
- Queued: exp_054 (race start), exp_050 (big gate bonus)
