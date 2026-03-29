# Insights & Benchmarks

> Background context, benchmark tables, and paper notes.
> This file is not a source of hard rules. Current repo beliefs should be stored
> in `FACTS.md`, `HYPOTHESES.md`, `TENTATIVE_LESSONS.md`, or `BELIEF_AUDIT.md`.

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

**Historical benchmark context:** nobody in this table finishes Level 2 reliably.
This legacy table does not by itself diagnose the active direct-racing line.

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
| Dynamic Entropy Tuning Quadcopter (2024) | 2512.18336 | Stochastic policies generalize where deterministic crash; dynamic entropy target=-dim(A) | exp_061 |
| When MaxEnt Misleads (2025) | 2506.05615 | PPO beats SAC for precision control; entropy corrupts Q-values in narrow-corridor tasks | reference |
| Entropy Annealing (2024) | 2405.20250 | Decay entropy τ=1/√(s+1) for continuous; start high, anneal to deterministic | exp_064 |
| Optimal Det. Policies from Stochastic PG (ICML 2024) | 2405.02235 | Mean is side effect of stochastic training, not the objective; action-space noise degrades mean more than parameter-space | exp_071-073 |
| gSDE: Smooth Exploration (CoRL 2022) | 2005.05719 | State-dependent noise forces mean to encode recovery; temporally correlated; available in SB3 | exp_075 (deferred) |
| CAPS: Action Smoothness (ICRA 2021) | 2012.06644 | Temporal + spatial smoothness regularization; 80% power reduction on real quadrotor | exp_072 |
| SimpleFlight: Sim-to-Real Factors (2024) | 2412.11764 | 5 critical factors; action-difference penalty critical for transfer; obs normalization | exp_071-072 |
| What Matters in On-Policy RL (ICLR 2021) | 2006.05990 | 250K agents: obs normalization critical, tanh, small final-layer init, softplus std | exp_071 |
| Structured Control Nets (ICML 2018) | 1802.08311 | Linear + nonlinear parallel branches; linear provides stability guarantee | reference |
| Colored Noise PPO (AAAI 2024) | 2312.11091 | Temporally correlated noise improves exploration and final perf in continuous control | reference |
| L2 Regularization in Policy Opt (2019) | 1910.09191 | L2 on policy network improves harder tasks; constrains mean drift | reference |

---

## Architecture Notes

- **Obs space:** 73 dims: drone state (13) + trajectory lookahead (30) + history (26) + last action (4). No gate info included.
- **Training env:** RandTrajEnv has zero gate awareness. Agent trains on random trajectory following, not gate racing.
- **Physics gap:** Training uses crazyflow, benchmarking uses MuJoCo sim. Different dynamics may be a limiting factor.
- **Action space (hover):** ONE_D_RPM = single-axis thrust. Fundamentally limits 3D precision.
- **Action space (racing):** attitude control (roll, pitch, yaw, thrust). 4 DOF.

---

## Harness Engineering Plan (2026-03-28)

- Repo plan saved at `docs/agentic_harness_upgrade_plan.md`.
- Priority order: typed artifacts -> resumable job runner -> skills/handoffs -> semantic tools -> harness evals -> observability -> memory promotion -> selective multi-agent work.
- If a task concerns queueing, agent orchestration, state management, or harness architecture, read that plan before making structural changes.
