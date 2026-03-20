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

### [IN PROGRESS] exp_027c -- Fine-tune exp_026 with Random Gate Starts
**Config:** `configs/exp_027c_racecore_finetune_randomgate.yaml`
**Depends on:** exp_026 checkpoint (pretrained hover model)

**Goal:** Fine-tune exp_026 (stable hover at z=0.72, 28.8s flight) with 50% random gate starts.
Unlike exp_027/027b (trained from scratch), this preserves hover skill via pretrained checkpoint.
Lower LR (0.0005 vs 0.0015) to prevent catastrophic forgetting of hover behavior.

**Training:** 512 envs, 8M steps, 5400s budget. Pretrained from exp_026 checkpoint.
