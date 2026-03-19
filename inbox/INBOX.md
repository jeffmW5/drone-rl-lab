# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_021 -- Smooth Trajectory Fix + Benchmark
**Completed:** 2026-03-18
**Results:** outbox/exp_021_smooth_traj.md

Yaw-aware approach/departure vectors (d=0.5m) using gate quaternion.
L0: 2.4 avg gates (up from 1.0). L2: 1.1 avg gates, best 3 (up from 0.4).
Still 0 finishes -- trajectory-following approach has hit its ceiling.

---

### [NEXT] exp_022 -- RaceCoreEnv Baseline (Direct Gate-Racing RL)
**Config:** `configs/exp_022_racecore.yaml`
**Depends on:** None

**Goal:** Train directly on VecDroneRaceEnv with dense gate-proximity reward.
This is the paradigm shift from trajectory-following to direct gate-racing.

**What's new:**
- `env_type: race` in config triggers the new VecDroneRaceEnv pipeline
- Dense reward: gate proximity + passage bonus + speed toward gate
- Observation: drone state + relative gate positions (57 dims with n_obs=2)
- MuJoCo physics = same as benchmark (no physics gap)
- Attitude control mode via level2_attitude.toml

**Training:** 1024 envs, 3M steps on GPU (RTX 3090)

**Success criteria:**
- Agent learns to fly toward gates (reward should increase over training)
- Passes at least 1 gate on Level 2 benchmark
- Any gate passage would exceed the trajectory-following ceiling for trained-on-RaceCoreEnv

**If it doesn't work:**
- May need reward tuning (gate_bonus, proximity_coef)
- May need larger network (64→128 or 256)
- May need curriculum (start L0, transfer to L2)
