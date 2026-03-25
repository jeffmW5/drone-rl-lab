# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [IN PROGRESS] exp_051 -- Longer Rollouts (num_steps=64)
- **Hypothesis:** 8-step rollouts can't observe full 65-step flight to gate. 64-step rollouts let GAE directly see navigation + gate bonus. gate_bonus=50 (5x increase).
- **Training on RunPod** — started 2026-03-25 09:20 UTC, budget 3600s, PID 1754832
- Config: `configs/exp_051_longer_rollout.yaml`

### [DONE 2026-03-25] exp_052 -- Action Smoothness + Tight Logstd
- **Result:** FAILURE — 0 gates, 1.19s flight. Training reward ALL-TIME HIGH (45.2) but benchmark same as exp_046. Action instability is NOT the crash cause.
- See `results/exp_052_smooth_actions/EXPERIMENT.md`

### [DONE 2026-03-25] exp_049 -- Survive=0.08 + Tight Logstd
- **Result:** FAILURE — 0 gates, 0.8s crash. Training peaked at 38.2 (ALL-TIME HIGH) but v_loss collapsed, final reward 21.0. survive=0.08 causes post-breakout instability.
- See `results/exp_049_survive_008/EXPERIMENT.md`

### [DONE 2026-03-25] exp_048 -- Short Episodes + Tight Logstd + Survive=0.15
- **Result:** FAILURE — 0 gates, 0.54s crash, reward 18.88. Short episodes → too aggressive.
- See `results/exp_048_short_ep_tight_logstd/EXPERIMENT.md`

### [DONE 2026-03-25] exp_047 -- Tight Logstd + Survive=0.15
- **Result:** FAILURE — hover trap, reward 10.0 flat. survive×1500=225 dominates gate bonus.
- See `results/exp_047_tight_logstd_more_survive/EXPERIMENT.md`

### [DONE 2026-03-24] exp_046 -- Tight Logstd (max_logstd=-1.0)
- **Result:** BEST BENCHMARK — 1.2-1.6s consistent flights toward gate, 29.2 train reward (peak 39.3)
- See `results/exp_046_tight_logstd/EXPERIMENT.md`

### [DONE 2026-03-24] exp_045 -- Logstd Clamp (max_logstd=0.5)
- **Result:** FAILURE — 0 gates, 0.7-2.2s. Pitch std clamped 90→1.65, still too wide.
- See `results/exp_045_logstd_clamp/EXPERIMENT.md`

### [DONE 2026-03-24] exp_044 -- Min Stability + Max Progress (20M steps)
- **Result:** Best training ever (26.38 mean, 37 peak) but FAILURE at benchmark — 0 gates, 0.9s crash
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
