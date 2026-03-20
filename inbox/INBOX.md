# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_027 -- Random Gate Starts (Swift-style initialization)
**Result:** Reward 11.67, but benchmark FAILED: 0/5 finishes, 0 gates, avg 1.1s flight time. 100% random gate spawns meant model never learned ground takeoff. Regressed to exp_025b levels.

---

### [DONE] exp_027b -- 50/50 Random Gate Mix
**Result:** Reward 10.79, benchmark 0/5 finishes, 0 gates, avg 1.18s. Model still applies max thrust
from step 1, never brakes. Mid-air envs dominated reward — model never learned ground hover despite
50% ground starts. Same failure mode as exp_027.

---

### [DONE] exp_027c -- Fine-tune exp_026 with Random Gate Starts
**Result:** Reward 6.34, benchmark 0/5 finishes, 0 gates, avg 3.16s. Partial takeoff preserved
but hover destroyed (3.2s vs 28.8s). Random gate starts fundamentally conflict with hover objective.

---

### [IN PROGRESS] exp_028 -- High Speed Reward (fine-tune exp_026, no random gates)
**Config:** `configs/exp_028_racecore_highspeed.yaml`
**Depends on:** exp_026 checkpoint

**Goal:** Teach horizontal navigation without random gate starts. exp_026 hovers at z=0.72 but
speed_coef=0.1 gives only 0.1 reward/step vs 1.8 for hovering — no incentive to move laterally.
Fix: speed_coef=1.0 (10x) + proximity_coef=0.5 (gentler falloff, signal from 2m away).
Fine-tune exp_026 with LR=0.0005 to preserve hover while adding lateral navigation.

**Training:** 512 envs, 8M steps, 5400s budget. Pretrained from exp_026 checkpoint.
