# INBOX -- Experiment Queue

> Mark each task [DONE] when complete.

---

## Queue

### [NEXT] Research — Drone Racing RL Literature Review
- **Type:** research
- **Query:** drone racing reinforcement learning, gate-relative observations, reward shaping
- **Output:** research/drone_racing_rl_overview.md
- **Completed:** 2026-03-23
- **Result:** Key finding from Swift (Nature 2023) + Song et al. (IROS 2021): obs space must include
  gate-relative info (distance, bearing, normal). Our 73-dim obs has ZERO gate info — #1 bottleneck.
  PPO works for drone racing when given correct observations. See research/drone_racing_rl_overview.md.

---

### [NEXT] exp_040 -- Gate-Relative Observations + View Reward (paper-informed)
- **Type:** training
- **Config:** `configs/exp_040_view_progress.yaml`
- **cuda:** true
- **Hypothesis:** Literature review (Swift, Song et al.) shows gate-relative observations are critical.
  Adding gate_in_view reward (dot product of drone forward axis with gate direction) gives the policy
  directional information it was missing. Combined with progress-only reward (no survive, no altitude,
  no proximity), the hover trap is eliminated — zero reward for staying still. Train from scratch
  with mid-air random gate spawns.
- **Code change:** Added `gate_in_view_coef` to `RaceRewardAndObs` in `train_race.py`
  (dot product of drone forward axis with direction to gate, clamped to [0,1])
- **Paper basis:** Swift (Kaufmann et al. Nature 2023), Song et al. (IROS 2021)
- **Research ref:** research/drone_racing_rl_overview.md
- **Success criteria:** Drone actively flies toward gates. Any gate passage = breakthrough.

---

## Completed

- exp_028-039: reward tuning, PBRS, entropy, curriculum — all hover-or-crash, 0 gates (see EXPERIMENT_LOG.md)
