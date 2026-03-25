# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [NEXT] exp_045 -- Logstd Clamp (max_logstd=0.5)
- **Type:** training
- **Config:** `configs/exp_045_logstd_clamp.yaml`
- **cuda:** true
- **Code change:** Added `max_logstd` param to Agent in `train_rl.py` — clamps actor_logstd
- **Hypothesis:** Same as exp_044 but with max_logstd=0.5 (std ≤ 1.65). exp_044's pitch std
  exploded to 90, making the mean meaningless. Clamping forces the policy to learn a meaningful
  mean action. If the mean can learn, the deterministic policy should navigate at benchmark.

### [DONE 2026-03-24] exp_044 -- Min Stability + Max Progress (20M steps)
- **Result:** Best training ever (26.38 mean, 37 peak) but FAILURE at benchmark — 0 gates, 0.9s crash
- **Key finding:** Reward design solved. Problem is deterministic mean policy crashes (exploration-exploitation gap)
- See `results/exp_044_min_stability_max_progress/EXPERIMENT.md`

### [DONE 2026-03-24] exp_041 -- Progress Only (no view reward)
- **Result:** FAILURE — 0 gates, 0.52s crash, reward 7.74
- See `results/exp_041_progress_only/EXPERIMENT.md`

### [DONE 2026-03-24] exp_042 -- View Scaled Down (0.1)
- **Result:** FAILURE — 0 gates, 0.52s crash, reward 7.74
- See `results/exp_042_view_scaled_down/EXPERIMENT.md`

### [DONE 2026-03-24] exp_043 -- View * Progress (multiplicative)
- **Result:** FAILURE — 0 gates, 0.52s crash, reward 7.75
- See `results/exp_043_view_times_progress/EXPERIMENT.md`

---

## Completed

- exp_028-039: reward tuning, PBRS, entropy, curriculum — all hover-or-crash, 0 gates
- exp_040: view+progress, falling exploit (0.95s crash), then XY-only fix still saturates at 7.75 (facing-only exploit)
- exp_041-043: progress/view variants without stability — all 0.52s crash, 0 gates (need survival incentive)
- exp_044: best training ever (peak 37) but 0 benchmark gates — deterministic mean crashes (exploration-exploitation gap confirmed)
