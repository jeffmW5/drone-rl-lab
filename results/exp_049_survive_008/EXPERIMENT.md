# exp_049 — Survive=0.08 + Tight Logstd (Binary Search)

## Hypothesis
Binary search between exp_046 (survive=0.05, 1.3s flights) and exp_047 (survive=0.15, hover trap).
survive=0.08 should provide slightly more flight incentive without triggering hover trap.
With tight logstd (max_logstd=-1.0), the deterministic mean must learn precisely.

## Config
- survive_coef: 0.08 (midpoint between 0.05 and 0.15)
- max_logstd: -1.0 (std ≤ 0.37)
- progress_coef: 50.0, gate_bonus: 10.0, speed_coef: 0.5
- alt_coef: 0.5, z_low: -0.05, z_high: 2.0
- num_envs: 512, num_steps: 8, gamma: 0.97
- max_episode_steps: 1500, random_gate_start: true

## Training
- **Mean reward:** 21.017 ± 1.098 (FINAL — declined from peak)
- **Peak reward:** 38.20 (iter 690)
- **Steps:** 3,690,496 / 20M (time-budget limited at 3604s)
- **GPU:** RTX 4090 on RunPod

### Training Trajectory (REMARKABLE)
The training showed a textbook late-stage breakout from hover trap, followed by instability:

1. **Hover plateau (iter 1-370):** reward oscillating at ~10.7-11.0
2. **Breakout (iter 390-570):** gradual climb 11.56 → 25.38
3. **Peak (iter 580-700):** rapid acceleration 27 → 38.20 (ALL-TIME HIGH)
4. **Collapse (iter 700-830):** v_loss explosions (243, 608), reward crashed 38 → 21

The breakout confirmed survive=0.08 CAN escape hover trap, but takes ~400 iterations
(vs ~200 for survive=0.05 in exp_046). However, the subsequent instability destroyed
the learned policy. The checkpoint saved at budget end captured the COLLAPSED state,
not the peak.

### Key v_loss Spikes
| Iter | Reward | v_loss | Note |
|------|--------|--------|------|
| 640 | 32.67 | 157.9 | First spike |
| 700 | 38.02 | 170.7 | Near peak |
| 720 | 32.05 | 243.2 | Decline starts |
| 770 | 25.86 | 608.7 | Catastrophic |
| 810 | 24.79 | 9.6 | Recovery attempt |
| 820 | 25.07 | 69.5 | Still unstable |

## Benchmark (Level 2)
| Run | Time (s) | Gates | Finished |
|-----|----------|-------|----------|
| 1 | 0.86 | 0 | No |
| 2 | 0.90 | 0 | No |
| 3 | 0.64 | 0 | No |
| 4 | 0.84 | 0 | No |
| 5 | 0.78 | 0 | No |
| **Avg** | **0.80** | **0** | **0%** |

## Result
**FAILURE** — 0 gates, 0.80s avg flight (WORSE than exp_046's 1.3s).

## Analysis
1. **survive=0.08 causes training instability:** The extra survive reward (0.08 vs 0.05)
   provides enough reward-per-step that the policy oscillates between navigation and hover
   modes after breakout. This causes the value function to explode (v_loss > 600), which
   corrupts the policy.

2. **Checkpoint captures wrong phase:** The final checkpoint at budget-end has the
   collapsed policy (reward ~21-25), not the peak policy (reward ~38). A best-checkpoint
   saver would have captured the superior policy.

3. **Survive binary search complete:**
   - 0.05: stable training (29.2 mean), best benchmark (1.3s flights) ← BEST
   - 0.08: unstable training (peaked 38.2, collapsed to 21), 0.8s flights
   - 0.15: hover trap (10.0 flat), 0.76s flights

4. **Conclusion:** survive=0.05 is the optimal value. The breakout at 0.08 shows that
   slightly higher survive CAN work for training, but the subsequent instability negates
   the gains. The path forward is NOT more survive — it's making the 1.3s flight reach
   the gate through action smoothness (exp_052) or longer rollouts (exp_051).

## Logstd
Raw: [1.25, 0.59, 1.50, -0.47] (clamped to -1.0 at runtime → std ≤ 0.37)
