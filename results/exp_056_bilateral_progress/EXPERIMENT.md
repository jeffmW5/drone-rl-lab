# exp_056 — Bilateral Progress Reward

## Hypothesis
One-sided progress reward `max(prev_dist - dist, 0)` only rewards getting closer but has ZERO
penalty for moving away. The stochastic policy collects reward from random movements that happen
to drift closer, while the deterministic mean has no directional learning signal. Fix:
`bilateral_progress=true` uses raw delta `(prev_dist - dist)` so moving AWAY from the gate is
penalized. This creates a proper gradient toward the gate for the deterministic mean.

## Config
- Base: exp_046 (tight logstd, survive=0.05, random gate starts)
- Change: `bilateral_progress: true`, `progress_coef: 50.0`
- File: `configs/exp_056_bilateral_progress.yaml`

## Training Results
- **Mean reward: 28.92** (peak 40.96 at iter 480)
- Steps: 3,706,880 / 20,000,000 (budget-limited at 3600s)
- Wall time: 3601s on RTX 3090 (community, $0.22/hr)
- GPU: RunPod RTX 3090, pod cw7hef3jd7kijr

### Reward curve
- iter 1-100: 11.79 → 14.99 (initial learning)
- iter 100-200: 14.99 → 13.84 (dip, then recovery)
- iter 200-350: 13.84 → 21.64 (breakout begins)
- iter 350-480: 21.64 → 40.96 (rapid climb, new all-time high)
- iter 480-700: 40.96 → 35-38 (settled with v_loss instability)
- iter 700-906: gradual decay to 28-30 range

### v_loss instability
Multiple v_loss spikes (339.7, 405.3, 333.7, 308.5, 215.6) suggest the value function
struggled to track rapid policy improvement. This may have contributed to the reward decay
in later iterations.

## Benchmark Results (level2_midair.toml)
| Run | Flight time (s) | Gates | Notes |
|-----|----------------|-------|-------|
| 1 | 0.64 | 0 | Dive crash |
| 2 | 0.80 | 0 | Dive crash |
| 3 | 0.62 | 0 | Dive crash |
| 4 | 0.64 | 0 | Dive crash |
| 5 | 0.52 | 0 | Dive crash |

**Average: 0.64s, 0 gates, 0 finishes**

### Trajectory analysis
The policy moves strongly toward gate 0 (correct direction in body frame) but:
- Dives aggressively: vz reaches -2.08 m/s, dropping from z=0.70 to z=0.01 in <0.8s
- Speed is very high (up to 4.0 m/s horizontal) but uncontrolled
- Crashes into ground before reaching gate (0.75m away at spawn)

This is PROGRESS vs exp_045/046: the policy now has a clear directional preference toward
the gate, whereas before it flew in arbitrary directions. But the bilateral progress signal
is too strong relative to the altitude/stability rewards, causing the deterministic mean to
prioritize horizontal approach over staying airborne.

## Diagnosis
1. **Bilateral progress works** — the policy learned gate direction
2. **progress_coef=50 is too aggressive** — dominates altitude reward (alt_coef=0.5)
3. **Training reward != benchmark performance** — 28.92 mean reward (peak 40.96) but 0 gates in benchmark
4. **The deterministic mean over-commits to horizontal approach** — stochastic training finds good trajectories but the mean is too aggressive

## Recommendations
- Lower `progress_coef` from 50.0 to 10-20 to balance with altitude
- Combine with body-frame obs (exp_057) for better directional control
- Combine with soft collision (exp_058) to learn from crash recoveries
- Asymmetric critic (exp_059) for better value estimates
