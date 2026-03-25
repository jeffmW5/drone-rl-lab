# exp_051 — Longer Rollouts (num_steps=64)

## Hypothesis
With 8-step rollouts, GAE can only observe 0.16s of a 1.3s flight. Increasing to 64 steps
(1.28s) lets GAE directly observe the full navigation trajectory including gate bonus.
Also increased gate_bonus from 10 to 50 since longer rollouts capture gate-passage signal better.

## Config
- num_steps: 64 (8x increase from exp_046)
- gate_bonus: 50.0 (5x increase from exp_046)
- survive_coef: 0.05, max_logstd: -1.0 (same as exp_046)
- All other params same as exp_046

## Training
- **Mean reward:** 175.604 ± 10.433 (per-step: 2.74)
- **Peak reward:** 189.27 (iter 80, per-step: 2.96)
- **Steps:** 3,899,392 / 20M (time-budget limited at 3601s)
- **Iterations:** 119 of 610 total
- **GPU:** RTX 4090 on RunPod

### Notes
- Only 610 total iterations (vs 4882 for 8-step) due to 8x larger batch per iter
- Each iteration takes ~30s (vs ~5s for 8-step)
- Reward climbed quickly then plateaued at per-step ~2.5-2.9
- v_loss consistently high (400-600) — value function struggles with longer horizon

## Benchmark (Level 2)
| Run | Time (s) | Gates | Finished |
|-----|----------|-------|----------|
| 1 | 1.32 | 0 | No |
| 2 | 1.16 | 0 | No |
| 3 | 1.24 | 0 | No |
| 4 | 1.26 | 0 | No |
| 5 | 1.12 | 0 | No |
| **Avg** | **1.22** | **0** | **0%** |

## Result
**FAILURE** — 0 gates, 1.22s avg flight. Same as exp_046 (1.3s) and exp_052 (1.19s).

## Analysis: THE REAL BOTTLENECK IS DOMAIN MISMATCH

Three fundamentally different training modifications (survive tuning, action smoothness,
longer rollouts) ALL produce the SAME ~1.2-1.3s benchmark flight. The training reward
varies massively (29→45→176) but benchmark is stuck. This is NOT a training problem.

**Root cause: training-benchmark domain gap**

| Condition | Training | Benchmark |
|-----------|----------|-----------|
| Start position | 0.75m from gate, mid-air | [-1.5, 0.75, 0.01] (ground) |
| Start altitude | Gate altitude (~0.7m) | Ground level (0.01m) |
| Distance to gate 0 | 0.75m | 2.06m (XY) + 0.69m (climb) |
| Ground takeoff | Never trained | Required |

The policy was NEVER trained to:
1. Take off from ground level (z=0.01)
2. Fly >0.75m to a gate
3. Climb while navigating

The 1.3s flight is the policy's best attempt to generalize from a completely
different starting condition. No amount of reward tuning or training hyperparameter
changes will fix this because the fundamental domain gap remains.

**Fix requires:** Either train from the actual race start position
(random_gate_ratio=0.0) or increase spawn_offset to cover the benchmark distance.
