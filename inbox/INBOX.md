# INBOX — Experiment 021: Smooth Trajectory Fix + Benchmark (NO RETRAINING)

**From:** Windows Claude (Orchestrator)
**Date:** 2026-03-17
**Priority:** HIGH — Trajectory shape fix, benchmark-only (uses existing model)

Refer to `MEMORY.md` for context. Read `outbox/exp_020_gpu_gate_traj_long.md` for the problem diagnosis.

---

## Context

exp_019 and exp_020 proved gate-aware trajectories work:
- **100% gate 1 pass rate on Level 0**
- **40-50% gate 1 pass rate on Level 2**
- But **ZERO Level 2 finishes** — drone crashes between Gate 1 and Gate 2 every time
- exp_020 (10M steps) got our **best-ever reward 7.79** but same crash point as exp_019 (3M)
- **Root cause:** The midpoint-based spline between gates creates too-aggressive altitude transitions (z=0.7→1.2) and direction changes

**This is a trajectory shape problem, NOT a policy problem.** The policy follows trajectories well. We just need smoother trajectories.

**KEY INSIGHT: We do NOT need to retrain.** The exp_020 model (7.79 reward) is trained to follow any trajectory in the observation space. Fix the trajectory generator → benchmark with existing model → done.

---

## Task

### Step 1: Pull latest code

```bash
cd /root/lsy_drone_racing && git pull
cd /root/drone-rl-lab && git pull
```

### Step 2: Understand current trajectory generation

The current gate-aware trajectory in `RandTrajEnv.reset()` (in `lsy_drone_racing/lsy_drone_racing/control/train_rl.py`) uses:
```
waypoints = [takeoff, climbout, approach1, gate1, approach2, gate2, approach3, gate3, approach4, gate4]
```
Where `approach_N = (prev_gate + gate_N) / 2` — simple midpoints.

This creates aggressive turns and altitude changes between gates.

### Step 3: Fix trajectory shape — 3-waypoint-per-gate approach

Replace the midpoint approach with **yaw-aware approach/departure vectors**. For each gate:
1. **Pre-gate waypoint**: 0.5m before gate along the gate's forward direction (use gate yaw from config)
2. **Gate center**: the gate position itself
3. **Post-gate waypoint**: 0.5m after gate along the gate's forward direction

This creates smooth fly-through paths instead of aggressive turns.

**Implementation guide:**

In `RandTrajEnv.reset()`, when `self.gate_positions is not None`:

```python
# Gate yaw angles from config (if available), otherwise estimate from gate-to-gate direction
# For each gate, create 3 waypoints:
#   pre_gate  = gate_pos - 0.5 * [cos(yaw), sin(yaw), 0]
#   gate_center = gate_pos
#   post_gate = gate_pos + 0.5 * [cos(yaw), sin(yaw), 0]
#
# Full waypoint sequence:
#   [takeoff, climbout, pre_gate1, gate1, post_gate1, pre_gate2, gate2, post_gate2, ...]
```

**If gate yaw is not easily accessible**, estimate the approach direction from the previous gate:
```python
direction = (gate_pos - prev_gate_pos) / |gate_pos - prev_gate_pos|
pre_gate = gate_pos - 0.5 * direction
post_gate = gate_pos + 0.5 * direction
```

This ensures the drone approaches each gate head-on rather than from an awkward angle.

**IMPORTANT:** Keep the altitude transitions smooth — linearly interpolate z between post_gate_N and pre_gate_N+1 rather than jumping.

### Step 4: Update the inference controller

The inference controller (`attitude_rl_exp020.py` or similar) needs to generate the SAME improved trajectory at test time. The controller's `_compute_reference_trajectory()` method must match the new waypoint generation.

**Create `attitude_rl_exp021.py`** — copy from the exp_020 version, update trajectory generation to match the new smooth approach, point to exp_020's model checkpoint:
```
results/exp_020_gpu_gate_traj_long/model.ckpt
```

**DO NOT RETRAIN.** Use the existing exp_020 model.

### Step 5: Benchmark on Level 2 (10 runs)

```bash
cd /root/lsy_drone_racing
python scripts/sim.py --config config/level2_attitude.toml --controller lsy_drone_racing.control.attitude_rl_exp021
```

Run **10 times**. Record per-run: gates passed, lap time, finish status, crash point (if crashed).

Also benchmark on Level 0 (5 runs) for comparison against exp_020.

### Step 6: If still crashing, try these fallbacks

**Fallback A:** Increase pre/post gate distance from 0.5m to 1.0m (wider approach arcs)

**Fallback B:** Add intermediate altitude waypoints between gates — if gate1.z=0.7 and gate2.z=1.2, add a waypoint at z=0.95 halfway between them

**Fallback C:** Reduce speed penalty coefficients to allow the drone to slow down for aggressive transitions

Benchmark each fallback variant (5 runs each on L2). Document which works best.

### Step 7: Document and push

1. Write `outbox/exp_021_smooth_traj.md` with:
   - What trajectory changes were made
   - Benchmark results (gates passed, lap times, finish rate) for each variant
   - Comparison table: exp_020 baseline vs exp_021 smooth trajectory
   - Best configuration identified
2. Update `MEMORY.md` with findings
3. Commit and push BOTH repos:
   ```bash
   cd /root/lsy_drone_racing && git add -A && git commit -m "exp_021: smooth gate trajectories for Level 2" && git push
   cd /root/drone-rl-lab && git add -A && git commit -m "exp_021: smooth trajectory benchmark results" && git push
   ```

---

## Important Notes

- **NO RETRAINING** — this experiment is trajectory fix + benchmark only
- Use exp_020's model checkpoint (`results/exp_020_gpu_gate_traj_long/model.ckpt`)
- The model is trained to follow trajectories — smoother trajectories should directly improve gate passage
- Gate yaw info should be in `config/level2_attitude.toml` under `track.gates` — each gate has position AND orientation
- This can run on **CPU** — no GPU needed for just benchmarking. But if already on GPU pod, that's fine too.
- **Push results before pod stops** — results that aren't pushed are lost
- **GPU is RTX 3090 (24GB VRAM)** if using pod

---

## Expected Output

1. `outbox/exp_021_smooth_traj.md` — full benchmark results + comparison table
2. Updated inference controller `attitude_rl_exp021.py` in lsy_drone_racing
3. Level 0 benchmark: 5 runs
4. Level 2 benchmark: 10 runs (+ fallback variants if needed)
5. Updated `MEMORY.md`
6. Both repos committed and pushed

## Success Criteria

- **Minimum:** Pass Gate 2 in at least 1/10 Level 2 runs
- **Good:** Finish Level 2 in at least 3/10 runs
- **Great:** Average lap time under 10s with >50% finish rate
