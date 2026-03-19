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

### [DONE] exp_022 -- RaceCoreEnv Baseline (Direct Gate-Racing RL)
**Completed:** 2026-03-19
**Results:** outbox/exp_022_racecore.md

Mean reward 6.34 ± 2.60, 1.93M steps (hit 1800s budget at iter 473/732).
Peak reward 10.04 at 1.23M steps. Agent consistently passes 1-2 gates.
Model checkpoint on RunPod pod at /root/drone-rl-lab/results/exp_022_racecore/model.ckpt.
Needs benchmark on L2 sim to measure actual gate count and lap time.

---

### [NEXT] exp_022 -- Benchmark on Level 2 sim
**Config:** Use attitude_rl_race.py controller with exp_022 checkpoint
**Depends on:** exp_022 training (DONE)

Benchmark the RaceCoreEnv-trained model on the actual L2 competition sim.
Need to get the model.ckpt from the pod and run benchmark.py.

---

### [QUEUED] exp_023 -- Extended RaceCoreEnv Training (5M+ steps)
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

### [QUEUED] exp_023 -- Extended RaceCoreEnv Training (5M+ steps)
**Config:** TBD (based on exp_022 benchmark results)
**Depends on:** exp_022 benchmark

exp_022 hit budget at 1.93M/3M steps and was still improving (reward 8.26 at iter 470).
Extend training to 5M+ steps with budget_seconds: 3600. Also consider:
- Larger network (128 or 256 hidden units)
- Reward tuning based on benchmark results
- Possibly reduce to 256 envs if compilation is an issue on fresh pod
