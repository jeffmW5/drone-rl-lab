# exp_020_gpu_gate_traj_long — Results

**Backend:** racing (lsy_drone_racing)
**GPU:** RTX 3090 (24GB VRAM)

## Training
| Metric | Value |
|--------|-------|
| mean_reward | **7.793 ± 0.051** (best ever) |
| peak_reward | 7.84 (iter 1200) |
| timesteps | 9,994,240 |
| wall_time | 1473s |

## Level 0 Benchmark (5 runs)
Gate 1: **5/5 (100%)**. Crashes at ~5.77s (gate 1→2 transition). Identical to exp_019.

## Level 2 Benchmark (10 runs)
Gate 1: **4/10 (40%)**. No finishes. Avg crash time ~5.2s.

## Key Finding
**More training does NOT fix the gate 1→2 crash.** Both 3M and 10M models crash at the same point. The bottleneck is trajectory shape, not policy quality. Reward 7.79 is our best ever, but the midpoint-based spline creates too-aggressive paths between gates at different altitudes.

## Recommendation
**Improve trajectory shape**: Use gate yaw-based approach/departure vectors (3 waypoints per gate instead of 2). This should create smoother through-gate paths that avoid the aggressive altitude transitions.

Alternatively, investigate training directly on MuJoCo (RaceCoreEnv) to eliminate the crazyflow→MuJoCo physics gap.
