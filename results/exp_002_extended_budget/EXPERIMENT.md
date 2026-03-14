# Experiment 002 — Extended Budget (Convergence Test)

## What we changed
Only `TRAINING_BUDGET_SECONDS`: 180s → 360s (doubled).
Reward function, PPO hyperparameters, and all other settings unchanged from exp_001.

## Why (the RL concept)
Before changing the reward shape, we needed to know whether exp_001's plateau was
a **training time problem** or a **reward ceiling problem**. If more time helps,
we fix the budget. If it doesn't, we fix the reward.

This is a standard RL diagnostic: isolate variables one at a time.

## Results
| Metric | Previous best (exp_001) | This experiment |
|--------|-------------------------|-----------------|
| mean_reward | 474.171 | 474.206 ✅ (marginal) |
| timesteps_trained | 63,489 | 223,070 |
| wall_time | 182s | 360s |
| improvement per 1k steps (after 63k) | — | ~0.0002 |

**The quartic reward has a hard ceiling of ~474.**

## Training curve — notable events

| Timestep range | What happened |
|----------------|---------------|
| 1k → 6k | Fast climb: 334 → 435 |
| 6k → 25k | **Policy collapse**: 435 → 166 (catastrophic forgetting) |
| 25k → 100k | Recovery: 166 → 474 |
| 100k → 223k | **Flat plateau**: 472–474, no meaningful improvement |
| 201k | Brief instability spike: 313 → 233 → recovers |

## What this tells us
**The reward shape is the bottleneck, not training time.**

With 3.5x more timesteps (223k vs 63k), mean_reward moved by only 0.035 — from
474.171 to 474.206. The plateau was firmly established by ~100k steps. Additional
training provided no benefit.

**The policy collapse (6k–25k) is worth noting.** Early in training, the policy
briefly discovered a reasonable hover (reward 435), then degraded sharply before
recovering. This is consistent with PPO's on-policy nature: when `n_steps=2048`
collects bad data, a gradient update can temporarily push the policy in the wrong
direction before it corrects. The EvalCallback's best_model saved the 435-era
model, which is why the final eval score (474) is better than the collapse trough.

**The drone still never triggers success termination.** Episode length remained
at 242 throughout all 223k steps — the drone hovers near the target but never
locks on within 0.0001m. The quartic reward's flat basin near the target means
there's no strong gradient pulling it to the exact target once it's close enough.

## Questions this opens up
- The quartic reward is giving a gradient signal that the drone has learned to
  exploit fully. What does the drone's hover actually look like — is it tight and
  stable, or wide and oscillating? We can't tell from reward alone.
- Why does the policy collapse happen at 6k–25k? Is this a hyperparameter issue
  (n_steps too large, learning rate too high) or just normal PPO variance?
- Could a reward with a stronger gradient near the target (e.g. quadratic instead
  of quartic) push past 474?

## Suggested next experiment
**The reward shape is confirmed as the bottleneck. Time to change it.**

Hypothesis: replacing the quartic with a **negative distance reward** (no clamp,
always provides gradient signal) will achieve a higher mean_reward and potentially
trigger the success termination condition:

```python
dist = np.linalg.norm(TARGET_POS - state[0:3])
return -dist
```

This is unbounded below, so we'll also want to watch for instability. Alternatively,
a quadratic reward (`max(0, 2 - dist**2)`) keeps the same structure but provides
a stronger gradient near the target.
