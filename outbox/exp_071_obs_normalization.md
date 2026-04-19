# exp_071 -- Observation Normalization

**Result: MIXED / INCOMPLETE** — training reward improved, but benchmark outcome is **not verified**.

- Single change from `exp_069`: enabled `obs_normalize: true`
- Training: **45.627 ± 0.801** mean reward in **4,628,480** timesteps over **7200.8s**
- Benchmark attempt ran, but the current generic controller path produced **zero parsed runs**
- Throughput note on the same pod/config: env stepping remains the bottleneck, not policy forward or PPO update
- Current evidence supports "better training reward under obs normalization"
- Current evidence does **not** support any benchmark/deployment claim yet
