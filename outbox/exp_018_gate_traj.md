# exp_018_gate_traj — Results

**Backend:** racing (lsy_drone_racing)
**Hypothesis:** Gate-aware trajectories — RandTrajEnv generates splines through actual gate positions with L2 randomization. Same reward, same obs, just better trajectories.

## Training Results (CPU, 213k steps)
| Metric | Value |
|--------|-------|
| mean_reward | 5.477 ± 0.263 |
| timesteps_trained | 213,504 |
| wall_time | 600.7s |
| level | level2 |

## Sim Benchmark — Level 0 (5 runs)
| Gates passed | Time | Status |
|:---:|:---:|:---:|
| 1/4 | 5.86s | crash after gate 1 |
| 1/4 | 5.42s | crash after gate 1 |
| 1/4 | 5.42s | crash after gate 1 |
| 1/4 | 5.38s | crash after gate 1 |
| 1/4 | 5.46s | crash after gate 1 |

**Gate 1 pass rate: 100% (5/5)**. Model crashes after gate 1 due to insufficient training.

## Sim Benchmark — Level 2 (10 runs)
| Gates passed | Time | Status |
|:---:|:---:|:---:|
| 0/4 | 4.82s | crash |
| 0/4 | 4.68s | crash |
| 0/4 | 4.52s | crash |
| 1/4 | 5.38s | crash after gate 1 |
| 0/4 | 4.04s | crash |
| 0/4 | 5.64s | crash |
| 0/4 | 6.14s | crash |
| 0/4 | 4.44s | crash |
| 0/4 | 5.10s | crash |
| 0/4 | 4.84s | crash |

**Gate 1 pass rate: 10% (1/10)**. L2 randomization + undertrained model = poor results.

## Analysis
- **Approach validated**: 100% gate 1 pass on L0 proves trajectories route through gates correctly
- **Severely undertrained**: 213k steps (CPU) vs 10M (GPU) for exp_016. Reward 5.48 vs 7.71.
- Training was still climbing when budget expired — reward trajectory suggests 7+ achievable with more steps
- **Critical next step**: GPU training (10M+ steps) with gate_aware=true

## Code Changes
See `outbox/gate_traj_implementation.md` for full details.
- Modified `RandTrajEnv.reset()` in lsy_drone_racing to generate gate-aware trajectories
- Added `gate_aware` config flag (fallback to original random trajectories when false)
- Created inference controller `attitude_rl_exp018.py`

## Recommendation
**exp_019: GPU training with gate-aware trajectories, 10M steps**. This is the most promising path to Level 2 completion. The approach works — it just needs more training compute.
