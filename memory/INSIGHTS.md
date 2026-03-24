# Insights & Benchmarks

## Kaggle Target

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |

**Our goal: sub-5.0s on Level 2 (top 3)**
**Current best (lap): 13.49s, 2/10 finishes (exp_016, 10M GPU steps, trajectory-following)**
**Current best (RaceCoreEnv): exp_026 — stable hover at gate altitude for 28.8s, 0 gates (altitude SOLVED, horizontal navigation needed)**
**Gap: ~3-4x slower than winners, 20% finish rate vs ~100%**

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

### Level 2 (randomized physics + gates) -- COMPETITION LEVEL
| Controller | Avg (s) | Finished | Gates |
|-----------|:-------:|:--------:|:-----:|
| State trajectory | 5.96 | 1/5 | 0-4/4 |
| Their RL | 7.25 | 0/5 | 0-3/4 |
| PID attitude | 8.59 | 2/5 | 0-4/4 |

**Nobody finishes Level 2 reliably. The problem is trajectory generation, not policy quality.**

---

## Paper References

> Papers that informed experiment design. Added by `/research` command.

| Paper | ArXiv ID | Key Insight | Used In |
|-------|----------|-------------|---------|
| Swift (Kaufmann et al. Nature 2023) | PMC10468397 | 31D obs with 12D gate corners in body frame; progress reward; 2×128 MLP; 100 envs, 1e8 steps | exp_040+ |
| Competitive Racing (Pasumarti 2024) | 2512.11781 | Sparse reward beats dense; 42D obs with 24D gate corners; [512,512,256,128] network; IPPO | exp_044 |
| Dream to Fly (Romero 2025) | 2501.14377 | PPO fails from pixels, DreamerV3 works; progress+gate bonus reward; b1=1.0 b2=0.01 | reference |
| RSS 2024 Vision Racing (UZH) | rss20/p082 | Asymmetric actor-critic with privileged info; gate edge sensor abstraction; 40 km/h | exp_042 |
| Curriculum Quadrotor (2025) | 2501.18490 | 3-stage curriculum (hover→random start→random vel); weight continuation; 20M total steps | exp_041 |
| Diverse Cluttered Tracks (RA-L 2025) | 2512.09571 | Soft-collision→hard-collision 2-phase training; preserves exploration in phase 1 | exp_041 |
| Teacher-Student Racing (CoRL 2024) | CoRL24_Xing | Train teacher with privileged state → distill to student → fine-tune student with RL | exp_042 |

---

## Architecture Notes

- **Obs space:** 73 dims: drone state (13) + trajectory lookahead (30) + history (26) + last action (4). No gate info included.
- **Training env:** RandTrajEnv has zero gate awareness. Agent trains on random trajectory following, not gate racing.
- **Physics gap:** Training uses crazyflow, benchmarking uses MuJoCo sim. Different dynamics may be a limiting factor.
- **Action space (hover):** ONE_D_RPM = single-axis thrust. Fundamentally limits 3D precision.
- **Action space (racing):** attitude control (roll, pitch, yaw, thrust). 4 DOF.
