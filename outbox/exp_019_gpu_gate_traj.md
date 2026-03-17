# exp_019_gpu_gate_traj — Results

**Backend:** racing (lsy_drone_racing)
**GPU:** RTX 3090 (24GB VRAM)

## Training
| Metric | Value |
|--------|-------|
| mean_reward | 7.546 ± 0.115 |
| timesteps | 2,998,272 |
| wall_time | 448s |

## Level 0 Benchmark (5 runs)
Gate 1: **5/5 (100%)**. Crashes at ~5.77s (gate 1→2 transition).

## Level 2 Benchmark (10 runs)
Gate 1: **5/10 (50%)**. No finishes. Avg crash time ~5.5s.

## Key Finding
Gate-aware trajectories reliably pass gate 1, but the gate 1→2 spline segment is too aggressive (z=0.7→1.2 climb + direction change). More training (exp_020, 10M) doesn't fix this — it's a trajectory shape issue.
