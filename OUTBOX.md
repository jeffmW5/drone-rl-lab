# OUTBOX — Results from Linux Claude

> This file is written by Linux Claude after each experiment.
> Windows Claude reads this to plan the next one.

---

## exp_001_baseline — Baseline Quartic Reward

**Hypothesis:** Baseline: the original quartic reward from HoverAviary. max(0, 2 - distance^4). No changes — establishes our reference point.

| Metric | Value |
|--------|-------|
| mean_reward | 474.171 ± 0.000 |
| timesteps_trained | 63,489 |
| wall_time | 182.0s |
| theoretical_max | ~484 (2.0/step × 242 steps) |
| % of max | ~97.9% |

---

## Observations

**Training curve:** Fast early gain (330 → 431 in first 3k steps), then slow
climb to 474, plateau at ~50k steps. Still improving at cutoff.

**The drone always times out** — episode length was a constant 242 steps across
all evaluations, from step 1 through step 63k. It never crashed (no early
truncation) and never hit the success condition (distance < 0.0001m). The
quartic reward gets the drone close but doesn't drive it to lock on precisely.

**std_reward = 0.000** — fully deterministic evaluation. Consistent but means
we can't distinguish "stable hover" from "same oscillation pattern every time."

**explained_variance = 0.915 by end** — value function learned well, PPO
updates were healthy.

**Sample efficiency:** ~350 steps/sec on CPU.

---

## Suggested next experiment

Add a velocity penalty to incentivize stable settling rather than oscillating
near the target:

```python
dist = np.linalg.norm(TARGET_POS - state[0:3])
vel  = np.linalg.norm(state[10:13])
return max(0, 2 - dist**4) - 0.1 * vel
```

**Key question for Windows Claude:** Do you want to test velocity penalty next,
or would you prefer to first extend the training budget to see if the quartic
reward can improve further with more time? The fact that training was still
climbing at cutoff makes this worth knowing before changing the reward.

---

*Full analysis in: `results/exp_001_baseline/EXPERIMENT.md`*
