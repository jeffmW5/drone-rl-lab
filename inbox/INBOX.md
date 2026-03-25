# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [IN PROGRESS] exp_055 -- Race Start + Takeoff Incentive
- **Hypothesis:** exp_054 stuck at ground (z_low=-0.05 gives full alt_reward at z=0.01). Fix: z_low=0.5, alt_coef=1.0 penalizes ground sitting (+6.17/rollout incentive to climb).
- **Training on RunPod** — started 2026-03-25 ~12:30 UTC, budget 3600s
- Config: `configs/exp_055_race_start_takeoff.yaml`

### [DONE 2026-03-25] exp_054 -- Race Start (no random gates)
- **Result:** FAILURE — stuck at reward 8.89 for 14.5M steps (ground hover trap). z_low=-0.05 gives full alt reward at z=0.01.
- See `results/exp_054_race_start/EXPERIMENT.md`

### [DONE 2026-03-25] exp_053 -- Farther Spawn (offset=1.5)
- **Result:** FAILURE — 0 gates, 0.70s (WORSE than exp_046). Farther horizontal spawn doesn't fix vertical domain gap.
- See `results/exp_053_farther_spawn/EXPERIMENT.md`

### [DONE 2026-03-25] exp_051 -- Longer Rollouts (num_steps=64)
- **Result:** FAILURE — 0 gates, 1.22s flight. Same crash pattern as exp_046/052 despite 64-step rollouts. Domain gap is the real bottleneck.
- See `results/exp_051_longer_rollout/EXPERIMENT.md`

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
