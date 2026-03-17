# Experiment 019 — GPU Gate-Aware Trajectory Training (3M steps)

## What we changed
Scaled exp_018's gate-aware trajectory approach to GPU: 1024 envs, 3M steps on RTX 3090.
Config: `configs/exp_019_gpu_gate_traj.yaml` with `gate_aware: true`, `cuda: true`.

## Why (the RL concept)
exp_018 (CPU, 213k steps) proved gate-aware trajectories work conceptually (100% gate 1 on L0). But 213k steps was severely undertrained (reward 5.48). This experiment provides adequate compute to test whether gate-aware trajectory following leads to reliable gate passage on Level 2.

## Results
| Metric | exp_016 (10M, random traj) | exp_018 (213k CPU, gate traj) | exp_019 (3M GPU, gate traj) |
|--------|---------------------------|-------------------------------|------------------------------|
| mean_reward | 7.71 | 5.48 | **7.55** |
| timesteps | 10,000,000 | 213,504 | 2,998,272 |
| wall_time | ~600s GPU | 601s CPU | 448s GPU |

### Level 0 Benchmark (5 runs)
| Gates passed | Time | Status |
|:---:|:---:|:---:|
| 1/4 | 5.80s | crash after gate 1 |
| 1/4 | 5.76s | crash after gate 1 |
| 1/4 | 5.76s | crash after gate 1 |
| 1/4 | 5.78s | crash after gate 1 |
| 1/4 | 5.76s | crash after gate 1 |
**Gate 1: 5/5 (100%). Crashes consistently at ~5.77s transitioning to gate 2.**

### Level 2 Benchmark (10 runs)
| Gates passed | Time |
|:---:|:---:|
| 0/4 | 4.54s |
| 1/4 | 6.14s |
| 0/4 | 5.36s |
| 1/4 | 5.36s |
| 1/4 | 5.96s |
| 1/4 | 6.72s |
| 0/4 | 4.78s |
| 0/4 | 5.78s |
| 1/4 | 6.68s |
| 0/4 | 6.02s |
**Gate 1: 5/10 (50%) on L2. No finishes.**

## What this tells us
1. **Gate-aware training reaches competitive reward quickly**: 7.55 at 3M steps vs 7.71 for exp_016 at 10M random-traj steps.
2. **Gate 1 passage works**: 100% on L0, 50% on L2. The trajectories correctly route through gates.
3. **Gate 1→2 transition is the bottleneck**: The drone crashes consistently at ~5.7s after passing gate 1. The climb from z=0.7 (gate 1) to z=1.2 (gate 2) combined with a direction change may be too aggressive for the spline path.
4. **Sim-to-training transfer gap**: Training reward is high (7.55) but sim performance crashes. The DroneEnv (crazyflow) → DroneRaceEnv (MuJoCo) domain gap may be significant.

## Questions this opens up
- Is the gate 1→2 spline segment too aggressive? Would adding more intermediate waypoints help?
- Could using gate yaw to create proper approach/departure vectors smooth the path?
- Is the sim-to-training physics gap the limiting factor?

## Suggested next experiment
exp_020 (10M steps with gate-aware) is already run alongside this. If results are similar, the bottleneck is trajectory shape, not training compute.
