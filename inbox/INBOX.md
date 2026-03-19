# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [NEXT] exp_022 -- Benchmark on Level 2 sim
**Config:** Use attitude_rl_race.py controller with exp_022 checkpoint
**Depends on:** exp_022 training (DONE)

Benchmark the RaceCoreEnv-trained model on the actual L2 competition sim.
Need to get the model.ckpt from the pod and run benchmark.py.

---

### [QUEUED] exp_023 -- Extended RaceCoreEnv Training (5M+ steps)
**Config:** TBD (based on exp_022 benchmark results)
**Depends on:** exp_022 benchmark

**Goal:** Extend RaceCoreEnv training to 5M+ steps with a longer time budget. exp_022 hit the 1800s wall at 1.93M/3M steps while reward was still climbing (peak 10.04 at 1.23M, ~8.26 at iter 470). More compute should push the agent from 1-2 gates toward full lap completion.

**What's new:**
- `budget_seconds: 3600` (2x longer than exp_022)
- Possibly larger network (128 or 256 hidden units) based on benchmark diagnostics
- Reward tuning based on benchmark results (gate_bonus, proximity_coef)
- Possibly reduce to 256 envs if compilation is slow on a fresh pod

**Training:** 512-1024 envs, 5M+ steps on GPU (RTX 3090)

**Success criteria:**
- Passes more gates than exp_022 (agent learns beyond gate 2)
- At least 1 lap completion on Level 2 benchmark
- Training reward continues to climb past exp_022 peak of 10.04

**If it doesn't work:**
- Try larger network (hidden_dim: 64 → 128 or 256)
- Reduce gate_bonus, increase proximity_coef for smoother learning signal
- Add curriculum: train first on level0, then transfer to level2
