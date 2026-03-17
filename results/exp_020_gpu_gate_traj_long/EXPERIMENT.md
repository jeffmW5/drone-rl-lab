# Experiment 020 — GPU Gate-Aware Trajectory Training (10M steps)

## What we changed
Extended gate-aware training to 10M steps on GPU (RTX 3090). Same architecture as exp_019, 3.3x more compute.
Config: `configs/exp_020_gpu_gate_traj_long.yaml`.

## Why (the RL concept)
exp_019 (3M steps) reached 7.55 reward and 50% gate 1 pass on L2. Hard rule #7 says reward plateaus ~7.7 for random trajectories. Does gate-aware training break through this ceiling with more compute?

## Results
| Metric | exp_016 (10M, random) | exp_019 (3M, gate) | exp_020 (10M, gate) |
|--------|----------------------|---------------------|---------------------|
| mean_reward | 7.71 | 7.55 | **7.79** |
| timesteps | 10,000,000 | 2,998,272 | 9,994,240 |
| wall_time | ~600s | 448s | 1473s |
| GPU | RTX 3090 | RTX 3090 | RTX 3090 |

Peak reward during training: **7.84** at iter 1200 (9.8M steps).

### Level 0 Benchmark (5 runs)
| Gates passed | Time | Status |
|:---:|:---:|:---:|
| 1/4 | 5.80s | crash after gate 1 |
| 1/4 | 5.78s | crash after gate 1 |
| 1/4 | 5.76s | crash after gate 1 |
| 1/4 | 5.78s | crash after gate 1 |
| 1/4 | 5.72s | crash after gate 1 |
**Gate 1: 5/5 (100%). Identical crash pattern to exp_019.**

### Level 2 Benchmark (10 runs)
| Gates passed | Time |
|:---:|:---:|
| 0/4 | 4.28s |
| 1/4 | 5.90s |
| 1/4 | 5.54s |
| 0/4 | 4.18s |
| 1/4 | 5.10s |
| 1/4 | 6.60s |
| 0/4 | 7.00s |
| 0/4 | 4.58s |
| 0/4 | 4.28s |
| 0/4 | 4.26s |
**Gate 1: 4/10 (40%) on L2. No finishes. Marginally worse than exp_019 (50%).**

## What this tells us
1. **Reward still climbing slightly**: 7.79 at 10M vs 7.55 at 3M, but the improvement is marginal (consistent with hard rule #7).
2. **More training does NOT fix the gate 1→2 crash**: Both 3M and 10M models crash at the same point (~5.7s on L0). The bottleneck is trajectory shape, not policy quality.
3. **The gate 1→2 segment requires trajectory redesign**: The straight midpoint-based spline from gate 1 (z=0.7) to gate 2 (z=1.2) creates an aggressive climb+turn that the agent can't execute in the sim environment.
4. **Sim-to-training physics gap is significant**: Reward 7.79 means excellent trajectory following in training (crazyflow), but the MuJoCo-based eval env has different dynamics.

## New hard rule
**Gate-aware midpoint trajectories crash at gate 1→2 transition** regardless of training steps (exp_018/019/020 all show same pattern). The climb from z=0.7 to z=1.2 combined with lateral direction change is too aggressive. Need approach/departure vectors based on gate yaw, or more intermediate waypoints.

## Questions this opens up
- Would gate yaw-based approach vectors create smoother trajectories?
- Would adding 2-3 waypoints per gate (approach, gate, departure) instead of 1 midpoint smooth the path?
- Is the fundamental issue the crazyflow→MuJoCo physics gap rather than trajectory shape?
- Would training directly on MuJoCo (RaceCoreEnv) eliminate the physics gap?

## Suggested next experiment
**Improve gate 1→2 trajectory**: Add approach/departure vectors based on gate yaw. Each gate gets 3 waypoints (approach at -0.3m along gate normal, gate center, departure at +0.3m) instead of 2 (midpoint, gate). This should create smoother through-gate paths.
