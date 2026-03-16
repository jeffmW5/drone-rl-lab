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
| 016 | racing | L2 | GPU, 10M steps | 7.71 | TBD | TBD | ✅ L2 converged, needs sim benchmark |

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
**Current status: DNF (0/5 finishes on Level 2 with all controllers)**

---

## What to Try Next

1. ~~**GPU training with 1024 envs + n_obs=2** (exp_014)~~ — DONE. Reward 7.29, validates n_obs=2 works with GPU
2. ~~**Train directly on Level 2** (exp_015)~~ — DONE. Reward 7.53 at 3M steps
3. ~~**Dynamic trajectory generation**~~ — TESTED, doesn't work at inference time alone (see hard rule #6)
4. ~~**Extended training** (exp_016, 10M steps)~~ — DONE. Reward 7.71, converged
5. **Benchmark exp_015/016 models on Level 2 sim** — CRITICAL. Training reward ≠ lap completion. Need to download models from RunPod pod and run 5-run sim benchmark.
6. **Retrieve model checkpoints from RunPod** — SCP blocked by RunPod proxy. Need git credentials or alternative transfer method. Pod still running.
7. **Investigate periodic reward dips** — Every ~800k steps, reward drops ~0.3 then recovers. Likely v_loss instability from catastrophic trajectory resets. May benefit from gradient clipping or lower LR.
