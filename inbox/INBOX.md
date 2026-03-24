# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [NEXT] exp_041 -- Progress Only (no view reward)
- **Type:** training
- **Config:** `configs/exp_041_progress_only.yaml`
- **cuda:** true
- **Hypothesis:** Remove view reward entirely. exp_040 showed it saturates at 1.0/step (just face
  gate), dominating progress. Drone has gate-relative obs — doesn't need a reward to look at gates.

### [QUEUED] exp_042 -- View Scaled Down (0.1)
- **Type:** training
- **Config:** `configs/exp_042_view_scaled_down.yaml`
- **cuda:** true
- **Hypothesis:** View=0.1 so progress (0.5/step) is 5x stronger. Small directional hint, can't be
  gamed by sitting still.

### [QUEUED] exp_043 -- View * Progress (multiplicative)
- **Type:** training
- **Config:** `configs/exp_043_view_times_progress.yaml`
- **cuda:** true
- **Hypothesis:** view * progress — zero unless both facing AND moving toward gate. No free reward
  for orientation alone.

---

## Completed

- exp_028-039: reward tuning, PBRS, entropy, curriculum — all hover-or-crash, 0 gates
- exp_040: view+progress, falling exploit (0.95s crash), then XY-only fix still saturates at 7.75 (facing-only exploit)
