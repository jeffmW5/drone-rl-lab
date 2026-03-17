# Memory — Lessons Learned

Read this file BEFORE starting any experiment. Update it AFTER every experiment.

---

## Hard Rules (never violate these)

1. **n_obs=2 with 64 envs + 600s is insufficient** — exp_013 only completed 297k/500k steps, reward regressed from 7.36 to 5.02. Need 1024+ envs on GPU. exp_014 proved this: 1024 envs + GPU → 7.29 reward in 206s.
2. **Level 2 requires adaptive trajectories** — all controllers (state, PID, RL) use fixed polynomial trajectories from hardcoded waypoints. When gates are randomized on Level 2, the trajectory flies through empty space. 0/5 finishes for every controller.
3. **ONE_D_RPM action space (hover backend) caps reward at ~474** — tested quartic, quadratic, velocity penalty, conservative PPO. The bottleneck is 1 action dimension, not the reward function.
4. **Don't confuse fast crashes with fast laps** — exp_013 averaged 4.30s on Level 2 with 0/5 finishes. The low time = early crash, not speed.
5. **Kaggle winners likely use dynamic path planning** that adapts waypoints to observed gate positions, not the static spline trajectory all current controllers use.
6. **Inference-time trajectory swapping doesn't work** — attitude_rl_dynamic.py built splines from obs["gates_pos"] using the exp_010 policy. Result: 0/5 finishes, 0-1 gates on L2 (4.64s avg = crashes). The policy is tightly coupled to the training trajectory shape. Fix requires training on diverse trajectories, not just swapping at inference.
7. **Training reward plateaus ~7.7 regardless of steps** — exp_016 (10M steps) only reached 7.71 vs exp_015's 7.53 at 3M. Diminishing returns after ~6M steps. Need architectural changes, not more compute.
8. **GPU training works but RunPod SSH proxy blocks SCP/file transfer** — use base64-over-TTY or Jupyter Lab file browser to retrieve model checkpoints. Set up deploy keys BEFORE training to enable git push from pod.
9. **exp_016 is the first RL to finish Level 2** — 13.49s, 2/10 finishes (20%). Better than reference RL (0/5) but 3-4x slower than Kaggle winners (~3.4-5.0s).
10. **Training env (RandTrajEnv) has ZERO gate awareness** — the RL agent trains on random trajectory following, not gate racing. It never sees gate positions, never gets gate-passage rewards. The only configurable reward params are penalties (rpy, action smoothness, energy). Adding a gate reward requires modifying train_rl.py's RandTrajEnv or building a new training pipeline on RaceCoreEnv.
11. **Obs space is 73 dims: drone state (13) + trajectory lookahead (30) + history (26) + last action (4)** — no gate info is included. The agent can only follow trajectories, not navigate to gates.

---

## Experiment Log

| Exp | Backend | Level | Key Change | Reward | Lap (s) | Gates | Outcome |
|-----|---------|-------|-----------|--------|---------|-------|---------|
| 001 | hover | — | Baseline quartic reward | 474 | — | — | ✅ ceiling for 1D RPM |
| 002 | hover | — | 2x budget | 474 | — | — | ✅ same ceiling, converges faster |
| 003 | hover | — | Quadratic reward | 369 | — | — | ❌ worse than quartic |
| 004 | hover | — | Velocity penalty | 407 | — | — | ❌ penalty hurts |
| 005 | hover | — | Conservative PPO | 437 | — | — | ❌ stability vs performance tradeoff |
| 010 | racing | L0 | Baseline n_obs=0, 64 envs | 7.36 | 13.36 | 4/4 | ✅ beats PID, 0.024s off reference |
| 013 | racing | L0 | n_obs=2 fix, 64 envs | 5.02 | crash | 0-1/4 | ❌ undertrained, only 297k steps |
| dyn | racing | L2 | Dynamic traj from gates_pos | — | 4.64 | 0-1/4 | ❌ policy coupled to training traj shape |
| 014 | racing | L0 | GPU, n_obs=2, 1024 envs, 1.5M | 7.29 | TBD | TBD | ✅ validates n_obs=2 works with GPU |
| 015 | racing | L2 | GPU, 3M steps | 7.53 | TBD | TBD | ✅ first L2 training, still climbing |
| 016 | racing | L2 | GPU, 10M steps | 7.71 | 13.49 | 2/10 finish | ✅ first RL to finish L2, but 20% rate |

---

## Benchmark: All Controllers vs All Levels (5 runs each)

### Level 0 (perfect knowledge)
| Controller | Avg (s) | Finished | Gates |
|-----------|:-------:|:--------:|:-----:|
| Their RL (pre-trained) | 13.34 | 5/5 | 4/4 |
| Our exp_010 (n_obs=0) | 13.36 | 5/5 | 4/4 |
| PID attitude | 13.37 | 5/5 | 4/4 |
| State trajectory | 13.86 | 5/5 | 4/4 |

### Level 1 (randomized physics)
| Controller | Avg (s) | Finished | Gates |
|-----------|:-------:|:--------:|:-----:|
| Their RL | 13.34 | 5/5 | 4/4 |
| PID attitude | 13.39 | 5/5 | 4/4 |
| State trajectory | 13.86 | 5/5 | 4/4 |

### Level 2 (randomized physics + gates) — COMPETITION LEVEL
| Controller | Avg (s) | Finished | Gates |
|-----------|:-------:|:--------:|:-----:|
| State trajectory | 5.96 | 1/5 | 0-4/4 |
| Their RL | 7.25 | 0/5 | 0-3/4 |
| PID attitude | 8.59 | 2/5 | 0-4/4 |

**Nobody finishes Level 2 reliably. The problem is trajectory generation, not policy quality.**

---

## Kaggle Target

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |

**Our goal: sub-5.0s on Level 2 (top 3)**
**Current best: 13.49s, 2/10 finishes (exp_016, 10M GPU steps)**
**Gap: ~3-4x slower than winners, 20% finish rate vs ~100%**

---

## What to Try Next

1. ~~**GPU training with 1024 envs + n_obs=2** (exp_014)~~ — DONE. Reward 7.29, validates n_obs=2 works with GPU
2. ~~**Train directly on Level 2** (exp_015)~~ — DONE. Reward 7.53 at 3M steps
3. ~~**Dynamic trajectory generation**~~ — TESTED, doesn't work at inference time alone (see hard rule #6)
4. ~~**Extended training** (exp_016, 10M steps)~~ — DONE. Reward 7.71, converged, 13.49s lap, 2/10 finishes
5. ~~**Retrieve model checkpoints from RunPod**~~ — DONE. Used base64-over-TTY. Pod stopped.
6. ~~**Benchmark exp_016 on Level 2 sim**~~ — DONE. 13.49s, 2/10 finishes. First RL to finish L2 but far from competitive.
7. **Improve finish rate** — 20% is not competition-ready. Need >80% to be meaningful.
8. **Improve lap time** — 13.49s vs target 5.0s. Need fundamental approach change, not just more training.
9. **Investigate: train with domain randomization on trajectory shape** — the policy needs to see diverse trajectories during training, not just the default spline
10. ~~**Investigate: reward shaping for gate passage**~~ — INVESTIGATED. Gate reward is impossible via config; training env (RandTrajEnv) has no gate concept. Requires code changes. See outbox/reward_investigation.md.
11. **[HIGH PRIORITY] Modify RandTrajEnv.reset() to generate gate-aware trajectories** — make training splines pass through actual gate positions (from level config), with randomization matching level2.toml. This is the most promising path: keeps existing architecture, makes trajectories gate-relevant. Requires modifying train_rl.py.
12. **[ALTERNATIVE] Build new training pipeline on RaceCoreEnv** — train directly on the gate-racing env with dense gate-proximity reward. Higher effort but fundamentally correct approach.
13. **exp_017: test reduced action penalties on CPU** — quick signal check: do lower d_act_xy_coef (0.3) and d_act_th_coef (0.15) allow faster flight? Won't fix gate problem but tests if penalties cause slowness.
