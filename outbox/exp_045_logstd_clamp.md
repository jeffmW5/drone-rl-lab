# exp_045 — Logstd Clamp (max_logstd=0.5)

**Result**: FAILURE — 0 gates, 0.7-2.2s flight (deterministic), 0 gates all stochastic scales
**Training reward**: 26.518 ± 0.962 (peaked ~37, same as exp_044)
**Timesteps**: 4.7M (3600s budget)

Clamp reduced pitch std from 90→1.65 but still too wide. Mean didn't learn precise control.
