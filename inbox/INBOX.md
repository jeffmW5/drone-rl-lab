# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [DONE 2026-03-27] exp_057 -- Gate Observations in Body Frame
- **Result:** FAILURE — training reward 9.78 (flat), benchmark 0.2 gates, 0.63s avg crash
- **Diagnosis:** progress_coef=20 too weak; body-frame obs alone doesn't compensate
- See `results/exp_057_body_frame_obs/EXPERIMENT.md`

### [DONE 2026-03-27] exp_058 -- Soft-Collision Curriculum (2-phase)
- **Result:** FAILURE — training reward 37.84, benchmark 0 gates, 1.22s avg crash
- **Diagnosis:** Soft collision boosted training reward (multi-life episodes) but domain gap (mid-air spawn vs ground benchmark) is the bottleneck, not crash termination
- See `results/exp_058_soft_collision/EXPERIMENT.md`

### [READY] exp_059 -- Asymmetric Actor-Critic
- **Status:** Code implemented, config created. Ready to train.
- **What was done:** `asymmetric_critic=true` flag. `AsymmetricAgent` class in train_rl.py. `AppendPrivilegedObs` wrapper adds all gate pos/quat (28D) to obs. Actor uses 57D, critic uses 85D. Inference auto-detects and loads only actor weights.
- **Config:** `configs/exp_059_asymmetric_critic.yaml`

### [DONE 2026-03-27] exp_060 -- Combined (Body-Frame + Soft Collision + Strong Progress)
- **Result:** FAILURE — training reward 28.02, benchmark 0 gates, 0.66s avg crash
- **Diagnosis:** Same stochastic-to-deterministic gap as exp_056. Combined structural changes don't fix deployment.
- See `results/exp_060_combined/EXPERIMENT.md`

### [NOTE] All three structural changes (057/058/059) are done except exp_059 (asymmetric critic, reclaimed).
- **Key finding:** exp_056-060 all show 25-38 training reward with 0 benchmark gates.
- **Bottleneck:** deterministic mean policy crashes; stochastic training policy navigates fine.

### [DONE 2026-03-28] exp_061 -- Stochastic Deployment of exp_060 Model
- **Result:** PARTIAL — avg 1.67s flight (2.5x improvement) but still 0 gates
- **Diagnosis:** Stochastic sampling stabilizes flight but full noise too imprecise for gates
- See `results/exp_061_stochastic_deploy/EXPERIMENT.md`

### [DONE 2026-03-28] exp_062 -- Temperature-Scaled Deployment
- **Result:** FAILURE — 2 gates in 70 runs across T=0.1-1.0. No sweet spot found.
- **Diagnosis:** Policy mean hasn't learned gate navigation; deployment-time fixes insufficient
- See `results/exp_062_temperature_scaled/EXPERIMENT.md`

### [CLAIMED:jeff-VirtualBox-6047-1774638579] exp_064 -- Entropy Annealing Schedule
- **Hypothesis:** Start with high entropy (ent_coef=0.05, no logstd clamp) for exploration, then anneal both to low values. Lets mean converge naturally.
- **What to change:** Implement ent_coef annealing (0.05 → 0.001). Remove max_logstd clamp. Train 10M+ steps on GPU.
- **Expected outcome:** Smoother convergence, mean finds the navigation mode.
- **Paper basis:** Entropy Annealing (2405.20250)
- **Config:** `configs/exp_064_entropy_annealing.yaml` (to create)

### [DEFERRED] exp_063 -- Extended Training (10M+ steps, no logstd clamp)
- **Depends on:** exp_061, 062, 064 results
- **Hypothesis:** Swift trains 100M steps; we train 1.5M. Remove max_logstd clamp and train much longer.
- **Paper basis:** Swift (Nature 2023)

---

### [DONE 2026-03-27] exp_057 -- Body-Frame Gate Observations (re-run)
- **Result:** See exp_057 above — same experiment, this was the re-run entry

### [DONE 2026-03-25] exp_056 -- Bilateral Progress Reward
- **Result:** Training reward 28.92 (peak 40.96) but benchmark 0 gates, 0.64s avg dive crash
- **Diagnosis:** Bilateral progress works (correct gate direction) but progress_coef=50 too aggressive
- See `results/exp_056_bilateral_progress/EXPERIMENT.md`

### [DONE 2026-03-25] exp_055 -- Race Start + Takeoff Incentive
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
