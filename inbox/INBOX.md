# INBOX — Experiment 018: Gate-Aware Trajectory Training

**From:** Windows Claude (Orchestrator)
**Date:** 2026-03-16
**Priority:** CRITICAL — This is the key unlock for Level 2

Refer to `MEMORY.md` for context on past experiments.
Read `outbox/reward_investigation.md` for the full reward analysis from the previous task.

---

## Context

The reward investigation revealed the **root cause** of our Level 2 failure:

> The training environment (`RandTrajEnv`) generates random spline trajectories with NO gate awareness. The agent learns to follow trajectories perfectly (reward 7.71) but has never seen a gate. At evaluation, the hardcoded trajectory doesn't pass through randomized gates → 20% finish rate, 13.49s.

**The fix: make training trajectories pass through gates.**

The agent already knows how to follow trajectories. If we make those trajectories go through the actual gates, it should fly through gates at test time too.

---

## Task: Modify RandTrajEnv to Generate Gate-Aware Trajectories

### Step 1: Understand the current trajectory generation

Read `RandTrajEnv.reset()` in `/media/lsy_drone_racing/lsy_drone_racing/control/train_rl.py`.

The current code generates random 10-waypoint cubic splines. The first 3 waypoints are hardcoded (climb out), the rest are random within bounds. None reference gate positions.

### Step 2: Find where gate positions are available

The level config files (`level0.toml`, `level2.toml`, etc.) define gate positions. Find:
- Where gate positions are loaded during training
- Whether `RandTrajEnv` has access to gate info (or could be given it)
- The gate position format (x, y, z, yaw)

### Step 3: Modify trajectory generation

**In the lsy_drone_racing fork** (`/media/lsy_drone_racing/`), modify `RandTrajEnv.reset()` to:

1. Load gate positions from the level config
2. For Level 2: apply the same randomization as `level2.toml` (±0.15m pos, ±0.2 rad yaw)
3. Generate a cubic spline that passes through the gate positions (in order)
4. Keep the climb-out waypoints at the start
5. Keep the existing reward function (trajectory following) — no reward changes needed

The key insight: we don't need to change the reward or observation space. We just need the trajectories to go through gates. The agent learns "follow this trajectory" and the trajectory happens to go through gates.

### Step 4: Verify the modification

Run a quick sanity check on CPU:
```bash
python train.py configs/exp_018_gate_traj.yaml
```

Create `configs/exp_018_gate_traj.yaml`:
```yaml
name: exp_018_gate_traj
backend: racing
hypothesis: "Gate-aware trajectories — RandTrajEnv generates splines through actual gate positions with L2 randomization. Same reward, same obs, just better trajectories."
budget_seconds: 600
racing:
  level: level2
  total_timesteps: 500000
  num_envs: 64
  num_steps: 8
  learning_rate: 0.0015
  gamma: 0.94
  gae_lambda: 0.97
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  n_obs: 2
  cuda: false
  seed: 42
```

Check:
- Does training converge? (reward should climb to ~7+)
- Does the agent crash more or less than before?
- Are the generated trajectories reasonable? (print first few waypoints)

### Step 5: Run sim benchmark on Level 2

After training, benchmark on Level 2 (5 runs) using the sim comparison method from before. The critical question: **does the agent pass through more gates?**

Even if the CPU model isn't fast enough (500k steps is limited), we should see >0 gates on Level 2 runs if the approach works.

### Step 6: Document and push

1. Document exact code changes to lsy_drone_racing in `outbox/gate_traj_implementation.md`
2. Write `results/exp_018/EXPERIMENT.md`
3. Write `outbox/exp_018_gate_traj.md`
4. Update `MEMORY.md`
5. Commit changes to BOTH repos:
   - `cd /media/lsy_drone_racing && git add -A && git commit -m "gate-aware trajectory generation in RandTrajEnv" && git push`
   - `cd /media/drone-rl-lab && git add -A && git commit -m "exp_018: gate-aware trajectory training" && git push`

---

## Important notes

- **You ARE allowed to modify lsy_drone_racing source code** for this task — specifically `RandTrajEnv.reset()` in `control/train_rl.py`
- **Do NOT change the reward function** — keep trajectory following as-is
- **Do NOT change the observation space** — keep the same 73 dims
- **Do NOT change train_racing.py** in drone-rl-lab unless absolutely necessary
- Keep the original random trajectory generation as a fallback (e.g., a config flag `gate_aware: true/false`)
- If gate positions aren't easily accessible in `RandTrajEnv`, document what's blocking and propose a workaround
- This is the most important experiment so far — take time to get it right

---

## Expected output

1. Modified `RandTrajEnv.reset()` in lsy_drone_racing fork
2. `configs/exp_018_gate_traj.yaml`
3. `outbox/gate_traj_implementation.md` — exact code changes documented
4. `outbox/exp_018_gate_traj.md` — results summary
5. `results/exp_018_gate_traj/EXPERIMENT.md` + `metrics.json`
6. Updated `MEMORY.md`
7. Level 2 sim benchmark results (even partial — gate count matters more than lap time here)
