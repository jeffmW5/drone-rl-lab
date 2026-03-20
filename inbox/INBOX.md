# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_028 -- High Speed Reward (fine-tune exp_026, no random gates)
**Result:** Reward 16.95 (highest ever), benchmark 0/5 finishes, **0.2 avg gates** (FIRST GATE PASSAGE in run 4!), avg 0.94s.
speed_coef=1.0 taught navigation but destroyed hover stability. Sweet spot is 0.3-0.5.

---

### [DONE] exp_029 -- Balanced Speed Reward (hover + navigation)
**Result:** Reward 16.52 ± 2.18 (very stable training), benchmark 0/5 finishes, 0 gates, avg 29.98s.
Hover perfectly preserved (29.98s = max episode!) but speed_coef=0.4 wasn't enough to break out of
hover local optimum. The phase transition between hover-only and navigation is between 0.4 and 1.0.
