# exp_016 Level 2 Sim Benchmark

**Date:** 2026-03-15
**Model:** exp_016_gpu_level2_long (10M steps, reward 7.71, n_obs=2)
**Controller:** attitude_rl_exp016.py (uses hardcoded trajectory + exp_016 policy)

## Results — Level 2 (10 runs)

| Run | Time (s) | Finished | Gates |
|:---:|:--------:|:--------:|:-----:|
| 1 | 6.30 | No | 1/4 |
| 2 | 3.96 | No | 1/4 |
| 3 | **13.52** | **Yes** | **4/4** |
| 4 | **13.46** | **Yes** | **4/4** |
| 5 | 12.84 | No | 3/4 |
| 6 | 12.86 | No | 3/4 |
| 7 | 3.36 | No | 0/4 |
| 8 | 3.64 | No | 0/4 |
| 9 | 3.40 | No | 0/4 |
| 10 | 6.34 | No | 1/4 |
| **Avg** | **7.97** | **2/10** | **0-4/4** |

## Comparison to previous L2 results (from benchmark_levels.md)

| Controller | Avg (s) | Finished | Best gates |
|-----------|:-------:|:--------:|:----------:|
| state_controller (hardcoded) | 5.96 | 1/5 | 4/4 |
| attitude_controller (hardcoded) | 8.59 | 2/5 | 4/4 |
| attitude_rl theirs (hardcoded) | 7.25 | 0/5 | 3/4 |
| attitude_rl_dynamic (exp_010 + gate traj) | 4.64 | 0/5 | 1/4 |
| **attitude_rl_exp016 (this)** | **7.97** | **2/10** | **4/4** |

## Analysis

### The breakthrough
exp_016 is the **first RL model to finish Level 2** — 2 complete laps out of 10 runs with all 4 gates. Previous best was the PID attitude controller at 2/5 finishes but the reference RL model was 0/5.

### The problem
Still highly inconsistent: 2/10 finishes. The runs split into two modes:
1. **Success mode** (runs 3-6): Drone follows trajectory, hits 3-4 gates, lap times 12.8-13.5s
2. **Crash mode** (runs 1-2, 7-10): Drone crashes early, 0-1 gates, times 3.3-6.3s

The crash mode happens when gate randomization places gates far enough from the hardcoded trajectory that the policy can't recover. The policy is more robust than before (trained on L2 randomization) but still follows a fixed trajectory.

### Why training reward doesn't predict finishes
Training reward of 7.71 looked great but only translates to 20% finish rate. The training environment rewards trajectory-following, not gate-passing. The policy learned to follow the trajectory well even when gates have moved — but following the wrong trajectory still means missing gates.

### Lap times for completed runs
- 13.52s and 13.46s — comparable to L0 times (13.34-13.86s)
- These are NOT competitive with Kaggle (target: sub-5s)
- The policy follows a 15s trajectory at training speed, not racing speed

### Next steps
1. **Combine exp_016 policy with dynamic trajectory** — exp_016 was trained on diverse trajectories. Unlike exp_010 (which was coupled to one trajectory shape), exp_016 may generalize to gate-based trajectories. This could fix both the crash rate AND the speed.
2. **Train with faster trajectory** — Current trajectory_time=15s produces ~13.5s laps. Reducing to 5-8s would force the policy to fly faster.
3. **Train with gate-aware reward** — Add reward for passing through gates, not just following the trajectory.
