# Experiment 005 — PPO Hyperparameter Tuning (Stability Fix)

## What we changed
Reverted reward to quartic (same as exp_001). Changed PPO hyperparameters:

| Parameter | exp_001 (default) | exp_005 (tuned) |
|-----------|-------------------|-----------------|
| learning_rate | 3e-4 | **1e-4** |
| n_steps | 2048 | **4096** |
| batch_size | 64 | **128** |
| n_epochs | 10 | **5** |
| clip_range | 0.2 | **0.1** |

## Why (the RL concept)
Every prior experiment showed a policy collapse pattern — reward spikes then
crashes. The hypothesis was that the default PPO settings were too aggressive:
large gradient steps on noisy data caused the policy to overshoot. These changes
make PPO more conservative: smaller steps, more data per update, tighter clipping.

## Results
| Metric | exp_001 (default PPO) | exp_005 (conservative PPO) |
|--------|----------------------|---------------------------|
| mean_reward | **474.171** ✅ | 437.347 ❌ |
| std_reward | 0.000 | 0.083 |
| timesteps_trained | 63,489 | 94,753 |
| episode length | 242 (stable) | 242→201 (late degradation) |
| policy collapse events | 1 | **0** |

**Conservative PPO eliminated policy collapse but produced significantly worse
final performance: −36.824 mean_reward vs default settings.**

## Training curve

| Timestep range | Reward | Episode length | Notes |
|----------------|--------|----------------|-------|
| 1k → 65k | 335 → 437 | 242 | Steady climb, **no collapse** ✅ |
| 65k → 77k | 437 → 430 | 242 | Slight plateau |
| 78k → 94k | 396 → 359 | 242 → **201** | Gradual degradation + early truncation |

The conservative settings achieved the stated goal — zero catastrophic policy
collapses. Training was smooth and monotonically increasing for 65k steps.
But the learning was too slow and the ceiling was much lower.

A new failure mode appeared at ~78k: gradual degradation with episode length
dropping from 242 → 201. This isn't a crash (reward didn't drop 200+ points
in one step) but a slow behavioral drift where the drone begins flying out of
bounds more often. Training ended in this degraded state.

## What this tells us
**The default PPO collapses may be beneficial, not harmful.**

This is a known phenomenon in RL: policy collapses can function as implicit
exploration. The aggressive default settings (3e-4, n_steps=2048) allow the
policy to make large moves — sometimes it overshoots and collapses, but the
recovery often lands in a better region of policy space. The conservative
settings prevent overshooting but also prevent the "lucky jumps" that drove
exp_001 from 330 → 474 in 63k steps.

The data supports this interpretation:
- exp_001 peak performance: 474.171 after one collapse-and-recovery
- exp_005 peak performance: 437.347 with zero collapses

The collapse is a symptom of aggressive learning, not a bug. Eliminating it
eliminated the fast convergence as well.

## Questions this opens up
- Is there a middle ground? `learning_rate=2e-4` with other defaults unchanged
  might preserve the beneficial exploration while reducing the worst overshoots.
- The late-training degradation (78k–94k) is a new failure mode not seen in
  exp_001. What causes it? Is it the small clip_range limiting recovery once
  the policy starts drifting?
- We've now tested reward shape (exp_001–004) and PPO stability (exp_005). Both
  approaches failed to beat 474. Is the ceiling a fundamental limit of the
  environment and action space?

## Suggested next experiment
Two realistic paths forward:

**Option A — Accept the ceiling, change the question.**
474 may be the practical maximum for this environment/action space. Instead of
optimizing raw reward, focus on a more interesting objective: train a drone that
reaches the target *from a random starting position* (randomize `initial_xyzs`).
This tests generalization, not just memorization of one trajectory.

**Option B — Middle-ground learning rate.**
Try `learning_rate=2e-4` with all other params at defaults. A single targeted
change, easier to interpret than the 5-parameter bundle we changed in exp_005.
