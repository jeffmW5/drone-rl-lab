# exp_041 — Progress Only (no view reward)

## Hypothesis
Remove view reward entirely. exp_040 showed gate_in_view saturates at 1.0/step
(just face gate), dominating progress. Drone has gate-relative obs — doesn't need
a reward to look. XY-only progress, random gate spawns, train from scratch.

## Config
- `gate_in_view_coef`: 0.0
- `progress_coef`: 50.0
- `reward_mode`: add
- All other rewards: 0.0
- `max_episode_steps`: 1500
- `random_gate_start`: true, `random_gate_ratio`: 1.0
- `spawn_offset`: 0.75, `spawn_pos_noise`: 0.15

## Training
- **Timesteps**: 4,825,088 / 8,000,000 (time-budget stopped at iter 1179/1953)
- **Wall time**: 5404s (~90 min)
- **Mean reward**: 7.742 ± 0.018
- **Reward curve**: Flat at ~7.75 from iter 200 onward — no learning signal

## Benchmark (mid-air sim)
- **Flight time**: 0.52s
- **Gates passed**: 0
- **Finished**: No
- **Behavior**: Immediate crash. Drone gets reward from initial spawn momentum
  (0.75m offset toward gate gives ~0.019m/step of XY progress before crashing).

## Result
**FAILURE** — Same hover-or-crash pattern. Progress-only reward is gamed by
initial momentum from spawn position. Reward of 7.75 = 0.97/step × 50 coef ×
0.019m drift. No actual flight behavior learned.

## Key Insight
With all shaping rewards removed (no survive, no altitude, no RPY), the drone has
no incentive to stay airborne. It collects progress reward from spawn-drift and
crashes. Progress reward alone is insufficient — needs a survival or stability
component to keep the drone in the air long enough to learn flight.
