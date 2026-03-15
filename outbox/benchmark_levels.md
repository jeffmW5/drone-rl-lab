# Full Benchmark — Levels 0, 1, 2

All controllers tested with 5 runs each. State controller uses `control_mode=state`; attitude-based controllers use `control_mode=attitude`.

**Competition target: sub-5s on Level 2 (Kaggle top 3: 3.39s, 4.89s, 5.02s)**

## Level 0 (perfect knowledge)

| Controller | Type | Times (s) | Avg (s) | Finished | Gates |
|-----------|------|-----------|:-------:|:--------:|:-----:|
| state_controller | Trajectory | 13.86, 13.86, 13.86, 13.86, 13.86 | 13.86 | 5/5 | 4/4 |
| attitude_controller | PID | 13.36, 13.40, 13.36, 13.38, 13.36 | 13.37 | 5/5 | 4/4 |
| attitude_rl (theirs) | RL (pre-trained) | 13.34, 13.34, 13.34, 13.32, 13.34 | 13.34 | 5/5 | 4/4 |
| **attitude_rl_exp013 (ours)** | **RL (exp_013)** | 3.38, 6.30, 2.84, 3.84, 3.82 | **4.04** | **0/5** | **0-1/4** |

## Level 1 (randomized physics)

| Controller | Type | Times (s) | Avg (s) | Finished | Gates |
|-----------|------|-----------|:-------:|:--------:|:-----:|
| state_controller | Trajectory | 13.84, 13.86, 13.84, 13.88, 13.86 | 13.86 | 5/5 | 4/4 |
| attitude_controller | PID | 13.38, 13.38, 13.40, 13.40, 13.38 | 13.39 | 5/5 | 4/4 |
| attitude_rl (theirs) | RL (pre-trained) | 13.34, 13.34, 13.34, 13.34, 13.34 | 13.34 | 5/5 | 4/4 |
| **attitude_rl_exp013 (ours)** | **RL (exp_013)** | 6.30, 6.52, 6.68, 2.84, 3.18 | **5.10** | **0/5** | **0-1/4** |

## Level 2 (randomized physics + gates) — COMPETITION LEVEL

| Controller | Type | Times (s) | Avg (s) | Finished | Gates |
|-----------|------|-----------|:-------:|:--------:|:-----:|
| state_controller | Trajectory | 4.02, **13.92**, 3.78, 4.10, 3.98 | 5.96 | **1/5** | 0-4/4 |
| attitude_controller | PID | 3.52, 6.28, **13.42**, 6.38, **13.34** | 8.59 | **2/5** | 0-4/4 |
| attitude_rl (theirs) | RL (pre-trained) | 13.08, 3.56, 6.28, 3.92, 9.40 | 7.25 | **0/5** | 0-3/4 |
| **attitude_rl_exp013 (ours)** | **RL (exp_013)** | 3.24, 3.76, 3.20, 8.52, 2.78 | **4.30** | **0/5** | **0-2/4** |

## Analysis

### exp_013 (n_obs=2) vs exp_010 (n_obs=0)
The n_obs fix made things **worse**, not better. exp_013 can't even finish Level 0 (0/5 runs), while the previous exp_010 completed 5/5. The root cause: n_obs=2 increased observation space from 47 to 73 dims (+55%), slowing training so much that the agent only completed 297k of 500k steps within the 600s budget. It's severely undertrained.

### Level difficulty progression
- **Level 0**: All 3 established controllers finish 5/5. Track is "solved."
- **Level 1**: Same — randomized physics has negligible impact on these controllers. The trajectory waypoints are hardcoded and the track layout is the same.
- **Level 2**: Catastrophic for everyone. Gate randomization breaks all controllers since they follow fixed trajectories. State controller: 1/5. PID: 2/5. Their RL: 0/5. Nobody reliably handles randomized gate positions.

### Where we stand vs Kaggle target (sub-5s on Level 2)
- **We're not close.** Our exp_013 averages 4.30s on Level 2 but with 0/5 finishes and 0-2 gates — it crashes fast, not flies fast.
- **Nobody finishes Level 2 reliably** — even the reference controllers. The hardcoded waypoints don't adapt to randomized gate positions.
- **The Kaggle winners likely use dynamic path planning** that adapts waypoints to observed gate positions, not static trajectories.

### The real bottleneck
It's NOT the RL policy quality — it's the **trajectory generation**. All controllers (including RL) follow a fixed spline trajectory computed from hardcoded waypoints. On Level 2, gates move but the trajectory doesn't. The RL agent learned to follow trajectories well, but following the wrong trajectory through empty space doesn't help.

### What's needed to be competitive
1. **Dynamic waypoint generation** — use observed gate positions from `obs["gates_pos"]` to compute trajectory on-the-fly
2. **More training compute for n_obs=2** — if we want observation stacking, need 1200s+ budget or 128+ envs
3. **Alternatively, stick with n_obs=0** (exp_010 worked) and focus on the trajectory problem
4. **Level 2 training** — train on Level 2 directly so the agent sees randomized gates during training
