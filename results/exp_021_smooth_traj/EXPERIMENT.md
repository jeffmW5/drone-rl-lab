# Experiment 021 — Smooth Yaw-Aware Trajectory (No Retraining)

## What we changed
Replaced midpoint-based gate trajectory generation with yaw-aware approach/departure waypoints. Each gate gets 3 waypoints: pre_gate (0.5m before along gate yaw), gate center, post_gate (0.5m after). Uses exp_020 model checkpoint (no retraining). See `attitude_rl_exp021.py`.

## Why (the RL concept)
The exp_019/020 models crashed at the gate 1→2 transition because the midpoint-based spline created aggressive altitude changes (z=0.7→1.2). By aligning approach/departure vectors with gate orientation (extracted from quaternion yaw), the spline generates smoother fly-through paths. Since the policy is trained to follow arbitrary trajectories, improving trajectory quality should directly improve gate passage without retraining.

## Results
| Metric | exp_020 (midpoint) | exp_021 (yaw-aware) |
|--------|:------------------:|:-------------------:|
| L0 avg gates | 1.0/4 | **2.4/4** |
| L2 avg gates | 0.4/4 | **1.1/4** |
| L2 best gates | 1 | **3** |
| L2 finishes | 0/10 | 0/10 |
| L2 gate 1 rate | 40% | **60%** |
| L2 gate 2+ rate | 0% | **30%** |

Fallback variants tested:
- d=1.0m: worse (avg 0.4 gates L2)
- d=0.75m: worse (avg 0.2 gates L2)
- d=0.5m + altitude interpolation: similar (avg 1.2, max 2 gates L2)

## What this tells us
1. Yaw-aware trajectories are a clear improvement — 2.4x more gates on L0, 2.75x on L2
2. The gate 1→2 crash is fixed; new bottleneck is gate 3→4
3. 40% of L2 runs still crash before gate 1 (extreme randomization)
4. The trajectory-following approach has hit its ceiling — smoother paths help but can't achieve finishes
5. Competition winners are 3x faster with ~100% finish rate, suggesting a fundamentally different approach

## Questions this opens up
- Can training on RaceCoreEnv (directly in MuJoCo) eliminate the physics sim gap?
- Would a reactive gate-seeking policy outperform pre-computed trajectory following?
- Is the 40% early crash rate due to physics randomization or trajectory shape?

## Suggested next experiment
**Build training pipeline on RaceCoreEnv** with dense gate-proximity reward and speed incentive. The trajectory-following paradigm (RandTrajEnv) has been exhausted. The next breakthrough requires end-to-end gate racing training.
