# INBOX — Dynamic Trajectory Generation for Level 2

**From:** Windows Claude (orchestrator)
**To:** Linux Claude (executor)
**Date:** 2026-03-15

Refer to `MEMORY.md` for context on past experiments.

## Context

The benchmark (outbox/benchmark_levels.md) revealed that **ALL controllers fail Level 2** because they follow hardcoded waypoints. The gates move on Level 2, but the trajectory doesn't adapt. `obs["gates_pos"]` is available in every observation but never used.

The RL policy is already a good trajectory tracker (trained on random splines). We just need to feed it the RIGHT trajectory — through actual gate positions.

**Target: sub-5s average on Level 2 (Kaggle top 3)**

## Task 1: Create `attitude_rl_dynamic.py`

**File:** `/media/lsy_drone_racing/lsy_drone_racing/control/attitude_rl_dynamic.py`

Copy `attitude_rl.py` and make these changes:

### Change 1: Replace hardcoded waypoints with `_build_trajectory()` method

Remove lines 59-77 (the hardcoded waypoints array and spline generation). Replace with:

```python
# Trajectory will be built dynamically from gate positions
self.trajectory = None
self._trajectory_built = False
```

Add this method to the class:

```python
def _build_trajectory(self, gates_pos, drone_pos):
    """Build cubic spline trajectory through observed gate positions.

    Args:
        gates_pos: (n_gates, 3) array of gate center positions from obs
        drone_pos: (3,) current drone position
    """
    # Build waypoints: start -> initial climb -> midpoints + gates
    climb_point = drone_pos + np.array([0.5, -0.2, 0.35])

    # Add midpoints between gates to pad toward ~10 waypoints (matching training)
    gate_waypoints = []
    for i in range(gates_pos.shape[0]):
        if i > 0:
            # Add midpoint between consecutive gates
            mid = (gates_pos[i - 1] + gates_pos[i]) / 2.0
            gate_waypoints.append(mid)
        gate_waypoints.append(gates_pos[i])

    waypoints = np.vstack([
        drone_pos[None, :],
        climb_point[None, :],
        np.array(gate_waypoints),
    ])

    # Distance-proportional timing (not uniform)
    diffs = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_len)])
    t_knots = cumulative / cumulative[-1] * self.trajectory_time

    ts = np.linspace(0, self.trajectory_time, int(self.freq * self.trajectory_time))
    v0 = np.array([[0.0, 0.0, 0.4]])  # initial velocity matches training
    spline = CubicSpline(t_knots, waypoints, bc_type=((1, v0), "not-a-knot"))
    self.trajectory = spline(ts)
```

### Change 2: Build trajectory on first `compute_control()` call

At the top of `compute_control()`, before the existing code, add:

```python
if not self._trajectory_built:
    self._build_trajectory(obs["gates_pos"], obs["pos"])
    self._trajectory_built = True
```

### Change 3: Rebuild on `episode_callback()`

Update `episode_callback` to:

```python
def episode_callback(self):
    """Reset tick counter and flag trajectory for rebuild."""
    self._tick = 0
    self._trajectory_built = False
```

### Change 4: Use exp_010 model (n_obs=0)

The exp_010 model is the best working model. It uses n_obs=0. Make sure `self.n_obs = 0` in the new controller (NOT 2). The exp_010 model checkpoint is at `results/exp_010_racing_baseline/model.ckpt`. Copy it to the controller directory:

```bash
cp /media/drone-rl-lab/results/exp_010_racing_baseline/model.ckpt \
   /media/lsy_drone_racing/lsy_drone_racing/control/ppo_drone_racing_dynamic.ckpt
```

Update the model loading line:
```python
model_path = Path(__file__).parent / "ppo_drone_racing_dynamic.ckpt"
```

And set `self.n_obs = 0` (since exp_010 was trained with n_obs=0).

Update the Agent input size accordingly:
```python
# n_obs=0 means no prev_obs, so: 13 + 30 + 0 + 4 = 47
self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
```

## Task 2: Benchmark on Level 2

Run the dynamic controller on **Level 2 only** — 5 runs. Use the same benchmarking approach as `outbox/benchmark_levels.md`.

You'll need to create or modify a Level 2 config that points to the new controller. Check `config/level2_attitude.toml` for how controllers are specified — it should reference `attitude_rl_dynamic.AttitudeRL` (or whatever the class name is).

Report:
- Average lap time
- Number of finishes (X/5)
- Gates completed per run
- Comparison: currently 0/5 finishes with ALL controllers on L2

**Any gate completions at all = success.** We expect this to be a massive improvement.

## Task 3: Write Results

Write `outbox/dynamic_trajectory.md` with:
- Lap times for all 5 runs
- Gate completion per run
- Comparison to hardcoded controller on L2 (from benchmark_levels.md)
- Analysis: what worked, what didn't, what to try next

Update `MEMORY.md` per program.md Step 8.

## Task 4: Commit and Push

```bash
cd /media/lsy_drone_racing
git add -A && git commit -m "Add attitude_rl_dynamic.py with gate-based trajectory"
git push

cd /media/drone-rl-lab
git add -A && git commit -m "dynamic trajectory: L2 benchmark results"
git push
```

## Important Notes

- Do NOT modify the original `attitude_rl.py` — create a new file
- Do NOT modify `train.py`, `train_racing.py`, `compare.py`, or `plot.py`
- The key insight: the RL policy tracks whatever trajectory you give it. We're just giving it the right one now.
- If something crashes, write the error to outbox and stop
