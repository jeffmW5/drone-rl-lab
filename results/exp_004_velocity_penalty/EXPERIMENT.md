# Experiment 004 — Velocity Penalty (Stability Test)

## What we changed
```python
# Before (quartic only)
return max(0, 2 - dist**4)

# After (quartic + velocity penalty)
dist = np.linalg.norm(self.TARGET_POS_CUSTOM - state[0:3])
vel  = np.linalg.norm(state[10:13])
return max(0, 2 - dist**4) - 0.1 * vel
```

## Why (the RL concept)
Pure position reward teaches the drone *where* to be but not *how* to get there
stably. A velocity penalty adds a second objective: minimize movement. In RL terms,
this shapes the reward landscape so that a drone drifting near the target with
high velocity is penalized relative to one hovering still. The hope was this would
push the drone to actively decelerate and settle rather than oscillate.

Note: raw mean_reward is expected to be lower than the quartic baseline because
velocity is always non-zero (the drone is always moving slightly). The real
metrics are episode length and training stability.

## Results
| Metric | exp_001 (quartic) | exp_004 (vel penalty) |
|--------|-------------------|-----------------------|
| mean_reward | 474.171 ✅ | 470.394 ❌ |
| std_reward | **0.000** | **0.288** |
| timesteps_trained | 63,489 | 108,001 |
| episode length | 242 (stable) | 242 (with crashes) |
| success termination | No | No |

**Quartic alone remains the best performer. Velocity penalty did not help.**

## Training curve — notable events

| Timestep range | Reward | Episode length | Notes |
|----------------|--------|----------------|-------|
| 1k → 10k | 333 → 432 | 242 | Fast climb, stable |
| 13k → 16k | 323 → 301 | 185 → 173 | **Crash event #1** |
| 17k → 20k | 448 → 455 | 242 | Recovery, new best |
| 21k → 90k | 410 → 375 | varied | Slow climb, instability |
| 91k → 104k | ~467 | 242 | Near-plateau |
| 105k → 108k | 362 → 321 | 189 → 168 | **Crash event #2 at cutoff** |

Training ended mid-crash — the final model was in a collapse state. The best
model (used for final eval) was saved during the stable ~91k–104k window.

## Key findings

**1. std_reward = 0.288 — first non-zero standard deviation across all experiments.**
The policy is no longer fully deterministic in evaluation. The velocity penalty
is introducing behavioral variance — the drone doesn't settle into a perfectly
consistent hover pattern.

**2. More frequent crashes than exp_001.** Two distinct crash events vs one in
exp_001. The velocity penalty appears to destabilize training rather than
stabilize it.

**3. Still no success termination.** Episode length never drops below 162 except
during crash truncations (early termination due to tilting/flying out of bounds,
not success).

**4. The velocity penalty coefficient (0.1) may be too aggressive.** At 0.1×vel,
the penalty for moving at typical hover velocity (~0.1–0.3 m/s) would be 0.01–0.03
per step. Small but potentially conflicting with the position signal during approach.

## What this tells us
The velocity penalty at weight 0.1 did not improve hover stability — it added
training instability (more crashes, non-zero eval std) and lowered the performance
ceiling. The quartic reward alone (474.171) remains the best result.

However, this doesn't rule out velocity penalty entirely. The coefficient may
simply be too large. A weight of 0.01 would be 10x smaller and might provide
the settling incentive without conflicting with the position gradient.

## Questions this opens up
- Is 0.1 the wrong coefficient for velocity penalty, or is any velocity penalty
  harmful with this PPO setup?
- The recurring policy collapses across exp_001–004 (always around 6k–25k steps)
  suggest PPO hyperparameters are causing instability. Could reducing `learning_rate`
  from 3e-4 to 1e-4 eliminate these collapses?
- At what point does the velocity penalty start hurting rather than helping?
  (0.1 hurts — does 0.01 help?)

## Suggested next experiment
**PPO hyperparameter tuning — lower learning rate.**
All four experiments have shown policy collapse during training. This is a PPO
stability issue independent of the reward function. Try:
```python
PPO_KWARGS = dict(
    learning_rate=1e-4,   # reduced from 3e-4
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
)
```
With the quartic reward (our best). If this eliminates the collapse pattern and
allows smoother training, we may push past 474.
