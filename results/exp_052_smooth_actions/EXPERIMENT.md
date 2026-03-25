# exp_052 — Action Smoothness + Tight Logstd

## Hypothesis
exp_046 flies 1.3s toward gate but crashes 0.5s short. Adding action smoothness penalties
(d_act_th_coef=0.4, d_act_xy_coef=0.5, rpy_coef=0.06) should stabilize flight without
creating hover trap (penalties apply equally to hover and navigation).

## Config
- survive_coef: 0.05 (same as exp_046)
- max_logstd: -1.0 (std ≤ 0.37)
- d_act_th_coef: 0.4, d_act_xy_coef: 0.5, rpy_coef: 0.06 (NEW)
- progress_coef: 50.0, gate_bonus: 10.0, speed_coef: 0.5
- alt_coef: 0.5, z_low: -0.05, z_high: 2.0
- num_envs: 512, num_steps: 8, gamma: 0.97
- max_episode_steps: 1500, random_gate_start: true

## Training
- **Mean reward:** 45.219 ± 2.867 (ALL-TIME HIGH)
- **Peak reward:** 54.95 (iter 850)
- **Steps:** 5,951,488 / 20M (time-budget limited at 3602s)
- **GPU:** RTX 4090 on RunPod

### Training Notes
- Started with NEGATIVE rewards (-35.66 at iter 1) due to action smoothness penalties
- Rapid recovery: -35 → +10 in ~100 iterations
- No hover plateau — broke out immediately (faster than exp_046)
- Very volatile: oscillated 32-55 throughout training
- v_loss spikes (up to 2348) but reward maintained high average
- No collapse like exp_049

## Benchmark (Level 2)
| Run | Time (s) | Gates | Finished |
|-----|----------|-------|----------|
| 1 | 1.34 | 0 | No |
| 2 | 1.14 | 0 | No |
| 3 | 1.34 | 0 | No |
| 4 | 1.12 | 0 | No |
| 5 | 1.02 | 0 | No |
| **Avg** | **1.19** | **0** | **0%** |

## Result
**FAILURE** — 0 gates, 1.19s avg flight. Comparable to exp_046 (1.3s), not better.

## Analysis
1. **Action smoothness inflates training reward but doesn't improve benchmark:**
   The 55% higher training reward (45.2 vs 29.2) comes from reduced penalty terms,
   not from better navigation. The benchmark shows the same ~1.2-1.3s crash pattern.

2. **The 1.3s crash is NOT caused by action instability:**
   If jerky actions were the crash cause, smoothing them should extend flight time.
   It didn't. The crash is caused by something else — likely the policy can't learn
   the precise deceleration needed to approach the gate without overshooting/crashing.

3. **8-step rollout window is the bottleneck hypothesis:**
   With num_steps=8 and gamma=0.97, GAE can only observe 8 steps (0.16s) of
   reward at a time. The flight to gate takes ~65 steps (1.3s). The value function
   must extrapolate the remaining 57 steps of navigation benefit from only 8 observed
   steps. This is extremely hard. Longer rollouts (exp_051, num_steps=64) should
   let GAE directly observe the full trajectory including gate bonus.

4. **No hover trap with smoothness penalties + survive=0.05:**
   The combination works well for training (no hover trap, fast breakout) but
   the benchmark benefit is zero. The penalties don't help the deterministic mean
   navigate better.

## Logstd
Raw: [clamped to -1.0 at runtime → std ≤ 0.37]
