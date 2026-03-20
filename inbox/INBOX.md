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

### [DONE] exp_028 -- High Speed Reward (fine-tune exp_026, no random gates)
**Result:** Reward 16.95 (highest ever), benchmark 0/5 finishes, 0.2 avg gates, avg 0.94s. speed_coef=1.0
overwhelmed hover rewards — model zooms and crashes. BUT: first-ever gate passage (1 gate in run 4).
Model learned navigation but lost stability. Need speed_coef between 0.1 (too low) and 1.0 (too high).
