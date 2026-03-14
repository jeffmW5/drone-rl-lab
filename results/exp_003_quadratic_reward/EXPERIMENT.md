# Experiment 003 — Quadratic Reward (Precision Test)

## What we changed
```python
# Before (quartic)
return max(0, 2 - dist**4)

# After (quadratic)
return max(0, 2 - dist**2)
```
Only change: exponent from `**4` to `**2`. Budget back to 180s.

## Why (the RL concept)
The quartic reward has a near-flat gradient close to the target (`d(d^4)/dd = 4d^3`
→ near zero as d→0). Our hypothesis was that a quadratic gradient (`d(d^2)/dd = 2d`,
still non-zero near target) would give the agent a stronger pull toward the exact
target, potentially triggering the success termination (distance < 0.0001m).

## Results
| Metric | exp_001 (quartic) | exp_003 (quadratic) |
|--------|-------------------|---------------------|
| mean_reward | 474.171 ✅ | 465.792 ❌ |
| std_reward | 0.000 | 0.000 |
| timesteps_trained | 63,489 | 114,001 |
| episode length | 242 (always) | 242 at end (varied during training) |
| success termination triggered | No | No |

**Quadratic reward performed worse than quartic: -8.379 mean_reward.**

## Training curve — notable events

| Timestep range | Reward | Episode length | What happened |
|----------------|--------|----------------|---------------|
| 1k → 8k | 294 → 386 | 242 | Fast climb |
| 8k → 30k | 386 → 188 | 242 → **124** | **Drone crashing more** |
| 30k → 95k | 188 → 416 | 124 → 242 | Long recovery |
| 95k → 114k | ~465 | 242 | Plateau |

The episode length dropping to 124 (vs always 242 in quartic experiments) means
the drone was being **truncated early** — flying out of bounds or tilting too
much. This is crashing behavior, not success termination.

## What this tells us
**The quadratic reward's stronger gradient caused more instability, not more
precision.**

The reasoning was sound — quadratic does provide a stronger gradient near the
target. But the stronger gradient at *all distances* appears to be the problem.
When the drone is at medium distance (0.3–1.0m), quadratic's gradient is 2–4x
stronger than quartic's. This pushed the drone to overshoot, destabilize, and
crash during the learning phase. The policy had to learn to dampen its own
behavior before it could hover stably — costing ~80k extra timesteps to recover.

The final plateau (465 vs 474) is **lower** than quartic despite 2x more
timesteps (114k vs 63k). The quadratic is both less sample-efficient and achieves
worse final performance.

**Neither experiment triggered success termination.** The issue isn't just the
reward gradient — the success condition (distance < 0.0001m = 0.1mm) may be
unreachable with `ONE_D_RPM` action space and KIN observations alone, regardless
of reward shape.

## Questions this opens up
- The instability at 8k–30k was worse with quadratic. Is this because the
  gradient is too strong, or is the PPO policy update size (learning_rate, n_steps)
  the real issue?
- Could a **reward that explicitly penalizes velocity** help the drone settle
  rather than oscillate? The position reward alone may be insufficient — the drone
  learns to hover near the target but keeps moving.
- Is the success termination condition (0.0001m) even reachable with the current
  action space (ONE_D_RPM = single-axis thrust)?

## Suggested next experiment
Return to the quartic position reward (proven better) but add a velocity penalty:
```python
dist = np.linalg.norm(self.TARGET_POS_CUSTOM - state[0:3])
vel  = np.linalg.norm(state[10:13])
return max(0, 2 - dist**4) - 0.1 * vel
```
Hypothesis: penalizing velocity will reduce oscillation around the target,
potentially improving mean_reward above 474 and producing more stable hover.
