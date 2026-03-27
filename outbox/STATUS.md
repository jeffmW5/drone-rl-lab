# Status -- Last Updated 2026-03-27

## Latest Result

**exp_060 -- Combined** completed 2026-03-27.

- Training reward: **28.02 mean** (still climbing at budget, peak ~29)
- Benchmark: **0 gates, 0.66s avg** — same crash as exp_056
- Combined body_frame_obs + soft_collision + progress_coef=50
- Confirms: stochastic-to-deterministic deployment gap is THE bottleneck

## The Pattern (exp_056-060)

| Exp | Train Reward | Benchmark Gates | Flight Time | Change |
|-----|:---:|:---:|:---:|------|
| 056 | 28.92 | 0 | 0.64s | bilateral progress |
| 057 | 9.78 | 0.2 | 0.63s | body_frame_obs (weak progress) |
| 058 | 37.84 | 0 | 1.22s | soft_collision |
| 060 | 28.02 | 0 | 0.66s | all three combined |

All produce 0 benchmark gates. The stochastic training policy navigates
(that's where the 25-38 reward comes from) but the deterministic mean crashes.

## Current Bottleneck

**Deterministic mean policy crashes at deployment.** This is NOT a reward
design, observation, or curriculum problem. It's a policy optimization /
deployment problem. The mean action at each state does not produce stable flight.

## Next Steps

Focus on fixing the deterministic deployment gap:
1. Deploy stochastic policy (sample from distribution, not mean)
2. Temperature scaling at deployment
3. Deterministic evaluation during training
4. Architecture changes (mixture policy, mode-seeking loss)
