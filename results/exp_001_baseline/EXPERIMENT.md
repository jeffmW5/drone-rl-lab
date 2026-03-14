# Experiment 001 — Baseline Quartic Reward

## What we changed
Nothing. This is the unmodified `HoverAviary` reward function:
```python
max(0, 2 - np.linalg.norm(TARGET_POS - state[0:3])**4)
```
Target position: `[0.0, 0.0, 1.0]`

## Why (the RL concept)
This is a **dense reward** — the agent receives a signal every timestep, not just
on reaching the goal. The quartic exponent (d^4) creates a steep drop-off: small
distances are barely penalized, but distances > ~1m yield near-zero reward. The
`max(0, ...)` clamp means once the drone is far enough away, the gradient vanishes
entirely — no signal to pull it back.

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | — (baseline) | 474.171 ✅ |
| std_reward | — | 0.000 |
| timesteps_trained | — | 63,489 |
| wall_time | — | 182.0s |

Theoretical max reward per episode: ~484 (2.0/step × 242 steps at perfect hover).
**474.171 = ~97.9% of theoretical max.**

## Observations
- **Reward curve:** jumped from 330 at 1k steps → 431 at 3k steps → plateaued
  around 474 from ~50k steps onward. The big early gain suggests the policy
  quickly learned the basic hover behavior; later training was refinement.
- **Episode length was always 242 steps** — the drone never crashed (which would
  truncate early) and never perfectly reached the target (which would terminate
  early). It always ran to the 8-second timeout. This means the drone learned
  to hover stably but not to lock onto the exact target point.
- **std_reward = 0.000** — evaluation is fully deterministic. Same starting
  conditions + deterministic policy = identical episodes every time.
- **Training was still improving at cutoff** (new best at 62k and 63k steps),
  suggesting more budget would yield further gains.
- **explained_variance** went from 0.000789 at step 3k to 0.915 at step 62k —
  the value function learned the environment well, which is a good sign for
  stable PPO updates.

## What this tells us
The quartic reward works — the drone learns to hover at 97.9% of theoretical
max reward within 3 minutes. However, the reward saturates early and training
was still climbing at cutoff. The drone hovers but doesn't perfectly lock on
(always times out rather than terminating via the success condition). The zero
std suggests the policy is deterministic and consistent, which is good, but
also that there's no variance to indicate it's exploring different strategies.

## Questions this opens up
- Would more training time push past 474, or has the quartic reward hit a
  ceiling due to its flat basin near the target?
- The drone never triggers the success termination (distance < 0.0001m) — is
  that because the reward shape doesn't incentivize that final precision?
- Could a reward that also penalizes velocity produce a more *stable* hover
  (less oscillation around the target) at the same or better mean_reward?
- Is 63,489 timesteps in 180s (≈350 steps/sec) our sample efficiency baseline?

## Suggested next experiment
**Hypothesis:** Adding a velocity penalty to the reward will produce a more
stable hover. Try:
```python
dist = np.linalg.norm(TARGET_POS - state[0:3])
vel  = np.linalg.norm(state[10:13])  # linear velocity
return max(0, 2 - dist**4) - 0.1 * vel
```
This keeps the quartic position reward but penalizes the drone for moving fast,
incentivizing it to settle rather than oscillate.
