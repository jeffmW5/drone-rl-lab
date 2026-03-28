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

### [DONE 2026-03-28] exp_059 -- Asymmetric Actor-Critic
- **Result:** MIXED — training reward improved to 32.50, but matched mid-air benchmark stayed at 0 gates with 0.79s avg flight
- **Diagnosis:** asymmetric critic helped 1-hour training efficiency but did not fix deployment
- **Operational note:** benchmarking required actor-only loading support for asymmetric checkpoints in `attitude_rl_generic.py`
- See `results/exp_059_asymmetric_critic/EXPERIMENT.md`

### [DONE 2026-03-27] exp_060 -- Combined (Body-Frame + Soft Collision + Strong Progress)
- **Result:** FAILURE — training reward 28.02, benchmark 0 gates, 0.66s avg crash
- **Diagnosis:** Same stochastic-to-deterministic gap as exp_056. Combined structural changes don't fix deployment.
- See `results/exp_060_combined/EXPERIMENT.md`

### [NOTE] Structural change sweep 057/058/059 is now complete.
- **Key finding:** body-frame obs, soft collision, and asymmetric critic all changed training behavior, but none produced a successful matched benchmark.
- **Scope note:** `exp_059` improved training reward, so "no training-side gain" would be too strong; the stronger conclusion is that benchmark success still did not follow.

### [DONE 2026-03-28] exp_061 -- Stochastic Deployment of exp_060 Model
- **Result:** PARTIAL — avg 1.67s flight (2.5x improvement) but still 0 gates
- **Diagnosis:** Stochastic sampling stabilizes flight but full noise too imprecise for gates
- See `results/exp_061_stochastic_deploy/EXPERIMENT.md`

### [DONE 2026-03-28] exp_062 -- Temperature-Scaled Deployment
- **Result:** FAILURE — 2 gates in 70 runs across T=0.1-1.0. No sweet spot found.
- **Diagnosis:** Policy mean hasn't learned gate navigation; deployment-time fixes insufficient
- See `results/exp_062_temperature_scaled/EXPERIMENT.md`

### [DONE 2026-03-28] exp_064 -- Entropy Annealing Schedule
- **Result:** FAILURE — 7.78 mean reward (flat at ~8 for 2.5M steps), 0 gates, 0.52s deterministic crash
- **Diagnosis:** ent_coef=0.03 too high, entropy dominated policy gradient. Annealing schedule too slow (only 6.3% through at budget end). Three-way confound with clamp removal and budget change.
- See `results/exp_064_entropy_annealing/EXPERIMENT.md`

### [CLAIMED:jeff-VirtualBox-15078-1774669522] exp_065 -- Periodic Deterministic Eval + Best Checkpoint
- **Hypothesis:** We are currently blind to whether the deployable deterministic mean improves during training. If periodic deterministic eval rises late, or `best_det.ckpt` beats the final checkpoint, that supports undertraining and checkpoint-selection effects. If training reward rises while deterministic eval stays flat, that weakens the simple "just train longer" story.
- **What to change:** Use the new `periodic_deterministic_eval` trainer hook. Every 50 iterations, run 8 deterministic eval episodes on a matched training env, save `best_det.ckpt`, and write `deterministic_evaluations.npz`. After training, benchmark both `best_det.ckpt` and the final `model.ckpt`.
- **Expected outcome:** Cleaner visibility into whether the mean policy is maturing, plus protection against saving a late-collapsed final checkpoint.
- **Scope note:** This is primarily an instrumentation experiment attached to the exp_064-style long-budget line, not a claim that eval logic alone will improve the policy.
- **Config:** `configs/exp_065_periodic_det_eval.yaml`

### [READY] exp_066 -- Asymmetric Critic + Entropy Annealing
- **Hypothesis:** If the remaining bottleneck is a combination of weak value targets and insufficient mean-policy convergence time, then exp_059's asymmetric critic plus exp_064's unclamped variance / higher entropy / longer budget should outperform either intervention alone.
- **What to change:** Start from the exp_064 long-budget line, add `asymmetric_critic: true`, and keep the periodic deterministic eval / best-checkpoint instrumentation from exp_065.
- **Expected outcome:** Better deterministic-eval trajectory during training and a better selected checkpoint than exp_064 alone. Benchmark improvement would support the "critic quality + convergence" story.
- **Scope note:** This is a highest-upside combination experiment, not a clean attribution study. A positive result would justify follow-up ablations; a negative result would weaken multiple training-side hypotheses at once.
- **Config:** `configs/exp_066_asym_entropy_annealing.yaml`

### [DONE 2026-03-28] exp_067 -- No Logstd Clamp (Clean Ablation from exp_060)
- **Result:** MIXED — 29.99 reward (matched exp_060), 0 gates, but 1.70s deterministic (2.6x better than exp_060's 0.66s)
- **Diagnosis:** Clamp removal does not hurt training; ent_coef=0.03 was exp_064's failure cause. Unclamped model has better deployment stability.
- See `results/exp_067_no_logstd_clamp/EXPERIMENT.md`

### [DONE 2026-03-28] exp_068 -- Extended No-Clamp Training (7200s)
- **Result:** MIXED — 42.84 reward (all-time high) but 0/15 det. gates (1.67s avg). 3 gate passages in 45 total runs across modes.
- **Diagnosis:** More training increases reward but doesn't proportionally improve deployment. Weakens simple undertraining hypothesis.
- See `results/exp_068_extended_no_clamp/EXPERIMENT.md`

### [DONE 2026-03-28] exp_069 -- Larger Network (2x128)
- **Result:** MIXED-POSITIVE — First deterministic gates (2/15), 5/15 T=0.3 gates (33%), 7/45 total. Peak reward 52.39 (new high).
- **Diagnosis:** Network capacity helps — 2×128 outperforms 2×64 on benchmark gates. But 87% deterministic failure rate persists.
- See `results/exp_069_larger_network/EXPERIMENT.md`

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
