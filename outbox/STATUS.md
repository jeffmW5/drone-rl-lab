# Status -- Last Updated 2026-03-27

## Latest Completed Results

**exp_057 -- Body-Frame Gate Observations** completed 2026-03-27.

- Training reward: **9.78 mean** (flat, never broke out) at 1.67M steps
- Benchmark: **0.2 avg gates, 0.63s avg flight** — worse than exp_056
- Takeaway: `progress_coef=20` too weak. Body-frame obs alone doesn't replace
  strong progress reward. Needs combining with `progress_coef=50`.

**exp_058 -- Soft-Collision Curriculum** completed 2026-03-27.

- Training reward: **37.84 mean** at 5.37M steps (high due to multi-life episodes)
- Benchmark: **0 gates, 1.22s avg flight** — same domain gap ceiling
- Takeaway: soft collision doesn't fix the mid-air→ground domain gap

## Combined Results Summary

| Experiment | Change | Train Reward | Benchmark Gates | Benchmark Time |
|------------|--------|:------------:|:---------------:|:--------------:|
| exp_056 | bilateral progress, coef=50 | 28.92 | 0 | 0.64s (dive) |
| exp_057 | body_frame_obs, coef=20 | 9.78 | 0.2 | 0.63s (crash) |
| exp_058 | soft_collision | 37.84 | 0 | 1.22s (crash) |

## Ready Next

- **exp_060 -- Combined** (body_frame_obs + progress_coef=50 + soft_collision)
  - Config ready: `configs/exp_060_combined.yaml`
  - Combines the structural changes that each showed partial promise
- **exp_059 -- Asymmetric Actor-Critic** (CLAIMED by another agent, status unknown)

## Key Bottleneck

The **domain gap** (Hard Rule #36) remains the primary issue. All three structural
experiments (057/058/060) use `random_gate_start=true` (mid-air spawns). Benchmark
starts from ground level. Until training includes ground starts, benchmark will
remain at ~0.6-1.3s regardless of reward or obs improvements.

## Reference Milestones

- **Legacy trajectory-following best lap:** `exp_016` -- 13.49s avg, 2/10 L2 finishes
- **Direct-racing benchmark reference:** `exp_046` -- 1.37s avg flight, 0 gates
- **Best gate passage:** `exp_028` -- 0.2 avg gates (1 gate in 1/5 runs)
