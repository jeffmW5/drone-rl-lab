# exp_044 — Min Stability + Max Progress (20M steps)

**Result**: FAILURE at benchmark — 0 gates, 0.7-1.4s flight (6 runs)
**Training reward**: 26.384 ± 0.724 (peaked at 37 — HIGHEST EVER)
**Timesteps**: 18.3M / 20M (time-budget stopped)

Best-ever training reward proves reward design works. Stochastic policy navigates
during training. Deterministic mean crashes at deployment. Problem is policy
optimization, not reward.
