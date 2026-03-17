# Reward Investigation — Full Analysis

**Date:** 2026-03-16
**Task:** INBOX exp_017 reward shaping investigation

---

## Critical Finding: Training Env Has NO Gate Awareness

The RL training pipeline does **NOT** use the gate-racing environment (`RaceCoreEnv`). Instead, it uses `RandTrajEnv` — a trajectory-following environment that has **zero gate knowledge**.

| Aspect | Training (`RandTrajEnv`) | Evaluation (`RaceCoreEnv`) |
|--------|-------------------------|---------------------------|
| Base class | `DroneEnv` (crazyflow) | MuJoCo-based gate racing |
| Task | Follow random spline | Fly through gates |
| Gates visible? | **NO** | Yes |
| Gate reward? | **NO** | Sparse: -1 at episode end |
| Trajectory | Random 10-waypoint cubic spline | Hardcoded or from gate positions |

This is the root cause of the 20% finish rate and 13.49s lap time on Level 2.

---

## Current Reward Function (Complete)

### Base Reward — `RandTrajEnv.reward()` (train_rl.py:270-283)

```python
# Distance to current trajectory point
norm_distance = ||pos - trajectory_point||
reward = exp(-2.0 * norm_distance)    # Range: (0, 1], peaks at 1.0 when on trajectory
reward = -1.0 if crashed              # Penalize ground/boundary collision
```

### Wrapper 1 — `AngleReward` (train_rl.py:379-398)

```python
rpy_norm = ||euler_angles(quat)||     # Roll-pitch-yaw magnitude
reward -= rpy_coef * rpy_norm         # Default rpy_coef = 0.06
# Also: yaw action is zeroed (action[..., 2] = 0.0)
```

### Wrapper 2 — `ActionPenalty` (train_rl.py:401-434)

```python
reward -= act_coef * thrust²          # Energy penalty (default 0.02)
reward -= d_act_th_coef * Δthrust²    # Thrust smoothness (default 0.4)
reward -= d_act_xy_coef * ||Δrp||²    # Roll/pitch smoothness (default 1.0)
```

### Summary: Total Reward Per Step

```
R = exp(-2 * dist_to_traj)           # trajectory following [0, 1]
  - 0.06 * ||rpy||                   # attitude penalty
  - 0.02 * thrust²                   # energy penalty
  - 0.4  * Δthrust²                  # thrust smoothness
  - 1.0  * (Δroll² + Δpitch² + Δyaw²)  # action smoothness
  or -1.0 if crashed
```

**There is NO gate-passage bonus. No gate proximity term. No progress term.**

---

## Observation Space (73 dims, flattened)

| Component | Source | Dims | Description |
|-----------|--------|------|-------------|
| pos | DroneEnv | 3 | Drone position [x, y, z] |
| quat | DroneEnv | 4 | Orientation quaternion |
| vel | DroneEnv | 3 | Linear velocity |
| ang_vel | DroneEnv | 3 | Angular velocity |
| local_samples | RandTrajEnv | 30 | Relative pos error to next 10 trajectory points (10 x 3) |
| prev_obs | StackObs (n_obs=2) | 26 | 2 frames of past state (2 x 13) |
| last_action | ActionPenalty | 4 | Previous [roll, pitch, yaw, thrust] |
| **Total** | | **73** | |

### What the agent DOES see:
- Its own full state (pos, quat, vel, ang_vel)
- Where the trajectory goes next (10 lookahead points, relative to drone)
- History of its own state (2 past frames)
- Its last action

### What the agent does NOT see:
- Gate positions or orientations
- Target gate index
- Obstacles
- Whether gates have been visited
- Any gate-related information whatsoever

---

## Inference-Time Pipeline

At evaluation, the RL policy is wrapped in an `attitude_rl*.py` controller that:

1. **Generates a trajectory** from hardcoded waypoints (or from `obs["gates_pos"]` in the dynamic variant)
2. **Constructs the same 73-dim observation** (pos, quat, vel, ang_vel, local_samples, prev_obs, last_action)
3. **Feeds it to the trained agent** for attitude commands
4. The agent follows this trajectory — it has no concept of gates

### Why Level 2 fails:
- Gates are randomized (up to ±0.15m position, ±0.2 rad yaw)
- The hardcoded trajectory passes through nominal gate positions, not actual ones
- The dynamic trajectory variant (attitude_rl_dynamic.py) failed because the policy is coupled to training trajectory shapes (Hard Rule #6)

---

## The `RaceCoreEnv.reward()` Function (NOT used in training)

For reference, the actual gate-racing env has this reward:
```python
def reward(self):
    return -1.0 * (self.data.target_gate == -1)  # -1 when finished, 0 otherwise
```
This is explicitly marked in the code as "will most likely not work directly for training."

The `gate_passed()` function exists in `envs/utils.py` and correctly detects gate passage via plane-crossing geometry. But it only runs in `RaceCoreEnv._step_env()`, never during training.

---

## Can We Add Gate Reward via Config?

**No.** The only configurable reward parameters are:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `rpy_coef` | 0.06 | Attitude penalty magnitude |
| `d_act_th_coef` | 0.4 | Thrust change penalty |
| `d_act_xy_coef` | 1.0 | Roll/pitch change penalty |
| `act_coef` | 0.02 | Thrust magnitude penalty |

All four are **penalties** on the existing trajectory-following reward. None add gate awareness. Tuning them can only change how smoothly the agent follows trajectories, not whether it targets gates.

---

## What Could Actually Help

### Option A: Modify `RandTrajEnv` to Generate Gate-Aware Trajectories
- Instead of random waypoints, generate splines that pass through the actual gate positions from the level config
- On Level 2, randomize gate positions during training (matching the level2.toml randomization)
- The agent would still learn trajectory-following, but the trajectories would always pass through gates
- **Requires modifying train_rl.py** (specifically `RandTrajEnv.reset()`)

### Option B: Train on `RaceCoreEnv` Directly
- Use the actual gate-racing environment for training
- Design a dense reward function with gate proximity bonus
- The agent would see gates in its observations and learn to fly through them
- **Requires a new training pipeline** — the current `train_racing.py` only supports `RandTrajEnv`

### Option C: Reduce Penalties for More Aggressive Flight (Config-Only)
- Lower `d_act_xy_coef` and `d_act_th_coef` to allow sharper turns
- This won't fix the gate problem but might improve lap time on Level 0
- **Testable on CPU** as a quick signal

### Option D: Increase Trajectory Diversity During Training
- The current `RandTrajEnv` already uses random waypoints, but the first 3 are hardcoded
- Making all waypoints random could improve generalization
- **Requires modifying train_rl.py** (`RandTrajEnv.reset()` lines 219-221)

---

## Recommendation for exp_017

**Option C** is the only one achievable without modifying `train_rl.py`. It tests whether the penalty coefficients are too conservative, causing slow flight. This won't solve the gate problem but provides useful signal.

**Option A** is the most promising path forward. It keeps the existing architecture but makes the training trajectories gate-aware. This should be proposed to the orchestrator as the next major step.

---

## Config: exp_017_reward_tuning.yaml

Created as a quick CPU test of reduced action penalties. Hypothesis: if the agent is less penalized for aggressive maneuvers, it may follow trajectories faster.

Changes from exp_016:
- `d_act_xy_coef`: 1.0 → 0.3 (less penalty for sharp turns)
- `d_act_th_coef`: 0.4 → 0.15 (less penalty for thrust changes)
- `act_coef`: 0.02 → 0.005 (less energy penalty)
- CPU/64 envs/500k steps (quick signal check)
