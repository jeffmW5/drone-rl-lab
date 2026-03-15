# Experiment 010 — Racing Baseline (Level 0)

## What we changed
First racing backend experiment. CleanRL PPO on lsy_drone_racing Level 0 with 64 parallel JAX environments, random trajectory following. Config: `configs/exp_010_racing_baseline.yaml`.

## Why (the RL concept)
Establishes a performance baseline for the racing backend. Level 0 provides perfect gate knowledge with no randomization — the simplest possible racing scenario. This isolates the RL algorithm's ability to learn trajectory tracking without confounders.

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | N/A (first racing exp) | 7.359 +/- 0.014 |
| timesteps_trained | N/A | 499,712 |
| wall_time | N/A | 588.8s |

## Training dynamics
- Reward started deeply negative (-44.93) as the drone crashed immediately
- Rapid improvement in first ~50 iterations (reward ~3.5 by step 25k)
- Steady climb through 200k-400k steps (reward 3.5 -> 7.0)
- Plateau around 7.2-7.4 in final 100k steps, suggesting near-convergence
- Value loss occasionally spiked but recovered quickly

## What this tells us
- The racing environment and CleanRL PPO pipeline work correctly end-to-end on CPU
- 64 envs with 8-step rollouts provide enough signal for stable learning
- The agent learns smooth trajectory following within the 600s budget
- The reward plateau suggests either the policy is near-optimal for this task or needs architectural/hyperparameter changes to improve further

## Questions this opens up
- Would more timesteps (1M+) push past the 7.35 plateau?
- How does performance change with more envs (128, 256)?
- What does the learned policy actually look like — does it fly smoothly or oscillate?
- How much does Level 1 (with gate randomization) degrade performance?

## Suggested next experiment
Increase total_timesteps to 1M and num_envs to 128 to see if we can break through the 7.35 reward ceiling, or try Level 1 to test generalization.
