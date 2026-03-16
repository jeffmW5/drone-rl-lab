# Dynamic Trajectory Controller — Level 2 Benchmark

**Date:** 2026-03-15
**Controller:** `attitude_rl_dynamic.py` (exp_010 policy + gate-based trajectory)

## Results — Level 2 (5 runs)

| Run | Time (s) | Finished | Gates Passed |
|:---:|:--------:|:--------:|:------------:|
| 1 | 5.62 | No | 0 |
| 2 | 4.08 | No | 0 |
| 3 | 5.40 | No | 1 |
| 4 | 4.08 | No | 0 |
| 5 | 4.00 | No | 0 |
| **Avg** | **4.64** | **0/5** | **0-1/4** |

## Comparison to Hardcoded Controllers on Level 2

| Controller | Avg (s) | Finished | Gates |
|-----------|:-------:|:--------:|:-----:|
| state_controller (hardcoded) | 5.96 | 1/5 | 0-4/4 |
| attitude_controller (hardcoded) | 8.59 | 2/5 | 0-4/4 |
| attitude_rl theirs (hardcoded) | 7.25 | 0/5 | 0-3/4 |
| attitude_rl_exp013 (hardcoded) | 4.30 | 0/5 | 0-2/4 |
| **attitude_rl_dynamic (this)** | **4.64** | **0/5** | **0-1/4** |

## Analysis

### What happened
The dynamic trajectory controller did NOT improve over the hardcoded version. Despite pointing the trajectory at actual gate positions, the drone still crashes early with 0/5 finishes and only 1 gate passed across all 5 runs.

### Why it didn't work
1. **Trajectory shape mismatch** — The exp_010 policy was trained on a specific hardcoded spline shape (10 waypoints, uniform timing, specific curve characteristics). The dynamic trajectory has different curvature, spacing, and timing even though it passes through the right gates. The policy learned to follow *that specific trajectory shape*, not arbitrary trajectories.

2. **The policy is not a general trajectory tracker** — This disproves the INBOX hypothesis that "the policy already knows how to follow any trajectory." It learned to follow one trajectory well. A different spline through different waypoints produces different local curvatures and velocity profiles that the policy hasn't seen.

3. **The climb point heuristic may be wrong** — The hardcoded `drone_pos + [0.5, -0.2, 0.35]` climb point was designed for the fixed start position. With randomized starts on L2, this may send the drone in the wrong direction initially.

### What to try next
1. **Train on randomized trajectories** — Train the RL policy on Level 2 directly so it sees many different trajectory shapes during training. This is the fundamental fix.
2. **Train with trajectory augmentation** — Generate random splines during training (not just the one hardcoded trajectory) so the policy generalizes to arbitrary paths.
3. **Simpler trajectory** — Instead of a full spline, try straight-line segments between gates. Less smooth but more predictable for the policy.
4. **Refit the original trajectory** — Instead of building a completely new spline, take the original 10 hardcoded waypoints and shift them toward the observed gate positions. This preserves the trajectory shape the policy expects while adapting to gate movement.

### Key insight
The bottleneck is NOT just "point at the right gates." The policy and trajectory are tightly coupled — the policy learned control responses specific to the training trajectory's curvature and timing. Fixing Level 2 requires training on diverse trajectories, not just inference-time trajectory adaptation.
