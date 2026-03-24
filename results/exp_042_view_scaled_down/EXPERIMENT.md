# exp_042 — View Scaled Down (0.1)

## Hypothesis
View reward scaled to 0.1 so progress (0.5/step) is 5x stronger. Small
directional hint, can't be gamed by sitting still. XY-only progress, random
gate spawns.

## Config
- `gate_in_view_coef`: 0.1
- `progress_coef`: 50.0
- `reward_mode`: add
- All other rewards: 0.0
- `max_episode_steps`: 1500
- `random_gate_start`: true, `random_gate_ratio`: 1.0

## Training
- **Timesteps**: 4,784,128 / 8,000,000 (time-budget stopped)
- **Wall time**: 5401s (~90 min)
- **Mean reward**: 7.737 ± 0.018
- **Reward curve**: Flat at ~7.75 from iter 200 onward

## Benchmark (mid-air sim)
- **Flight time**: 0.52s
- **Gates passed**: 0
- **Finished**: No
- **Behavior**: Immediate crash. Same as exp_041.

## Result
**FAILURE** — Identical to exp_041. Scaling view down to 0.1 makes no difference
when the drone crashes in 0.5s. Without survival incentive, progress + small view
reward is still gamed by spawn-drift momentum.
