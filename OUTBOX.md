# OUTBOX — Results from Linux Claude

> This file is written by Linux Claude after each experiment.
> Windows Claude reads this to plan the next one.

---

## Full Experiment Comparison (5 experiments)

| # | Name | mean_reward | std | timesteps | collapses | notes |
|---|------|-------------|-----|-----------|-----------|-------|
| 001 | quartic (baseline) | **474.171** | 0.000 | 63,489 | 1 | best overall |
| 002 | quartic + 6min | 474.206 | 0.000 | 223,070 | 1 | proves reward is ceiling |
| 003 | quadratic | 465.792 | 0.000 | 114,001 | many | stronger gradient = more crashes |
| 004 | quartic + vel penalty | 470.394 | 0.288 | 108,001 | 2 | penalty hurts, not helps |
| 005 | conservative PPO | 437.347 | 0.083 | 94,753 | **0** | stable training, lower ceiling |

---

## exp_005 Summary

The conservative PPO settings achieved the stated goal: **zero policy collapses**.
Training was smooth and monotonically climbing for 65k steps. But:

- Final reward (437.347) is **−36.824 below the baseline** (474.171)
- A new failure mode appeared at 78k: slow degradation + episode length dropping
  to 201 (drone drifting out of bounds without crashing sharply)
- The default PPO's collapses appear to be beneficial exploration, not bugs

---

## Insight: we may have hit the environment ceiling

After 5 experiments, the quartic baseline (474.171, exp_001) remains the best.
Nothing we've tried has beaten it:
- More training time: +0.035 (negligible)
- Quadratic reward: −8.379
- Velocity penalty: −3.777
- Conservative PPO: −36.824

**474 may be the practical maximum for this environment/action space.**
The drone hovers near [0,0,1] but never locks on — episode always times out at
242 steps. With `ONE_D_RPM` action space (single-axis thrust), 3D precision
hovering may be fundamentally limited.

---

## Suggested next experiment

**Option A — Randomize starting position (generalization test)**
Change `initial_xyzs` to sample randomly within a cube around the target. Tests
whether the policy can generalize rather than memorizing one trajectory. More
interesting scientifically and practically useful.

**Option B — Middle-ground learning rate (2e-4)**
Single targeted change to see if there's a sweet spot between aggressive (3e-4,
collapses but high ceiling) and conservative (1e-4, stable but low ceiling).

**My recommendation:** Option A. We've exhausted incremental reward/hyperparameter
tuning. The 474 ceiling appears real. The next interesting question is
generalization.

---

*Full analysis in: `results/exp_005_ppo_tuning/EXPERIMENT.md`*

---

## exp_010 — Racing Baseline (NEW BACKEND)

**Backend:** racing (lsy_drone_racing + CleanRL PPO + JAX environments)

| Metric | Value |
|--------|-------|
| mean_reward | **7.359 +/- 0.014** |
| timesteps_trained | 499,712 |
| wall_time | 588.8s |
| level | level0 |

Racing pipeline is fully operational. Agent learned random trajectory following from scratch (-44.93 -> 7.35 reward). Three-phase learning: crash recovery (0-25k), rapid improvement (25k-250k), plateau (250k-500k).

### Setup notes for Windows Claude
- **Python 3.11 required** — JAX >= 0.7 doesn't support Python 3.10
- Venv is at `/home/jeff/drones-venv` (not `/media/`) because VirtualBox shared folder can't create symlinks
- `/media/drones-venv/bin/activate` is a redirect script that sources the real venv
- Pre-trained sim.py test passed: 4/4 gates, 13.86s

### Suggested next
- 1M timesteps to push past the 7.35 plateau
- Level 1 (gate randomization) for generalization
- Increase num_envs to 128 or 256

*Full analysis in: `results/exp_010_racing_baseline/EXPERIMENT.md`*
