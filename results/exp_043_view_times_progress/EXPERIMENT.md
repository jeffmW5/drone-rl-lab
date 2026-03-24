# exp_043 — View * Progress (multiplicative)

## Hypothesis
Multiplicative reward: view * progress. Zero reward unless BOTH facing the gate
AND moving toward it. Facing alone = 0. Moving sideways = 0. Only flying toward
the gate while looking at it gets reward.

## Config
- `gate_in_view_coef`: 1.0
- `progress_coef`: 50.0
- `reward_mode`: multiply
- All other rewards: 0.0
- `max_episode_steps`: 1500
- `random_gate_start`: true, `random_gate_ratio`: 1.0

## Training
- **Timesteps**: 4,747,264 / 8,000,000 (time-budget stopped)
- **Wall time**: 5401s (~90 min)
- **Mean reward**: 7.748 ± 0.013
- **Reward curve**: Flat at ~7.75 from iter 200 onward

## Benchmark (mid-air sim)
- **Flight time**: 0.52s
- **Gates passed**: 0
- **Finished**: No
- **Behavior**: Immediate crash. Multiplicative mode doesn't help — drone starts
  facing gate AND drifting toward it from spawn, so both components are nonzero
  during the initial drift before crash.

## Result
**FAILURE** — Identical to exp_041 and exp_042. All three reward variants
(progress-only, view-scaled, multiplicative) produce the same 0.52s crash.
Without survival incentive, the drone crashes before learning anything.

## Key Insight
exp_040-043 conclusively prove: **progress/view reward alone cannot bootstrap
flight**. The drone needs a stability component (survive, altitude, or RPY
penalty) to stay airborne long enough to discover that moving toward gates is
rewarding. The challenge is adding enough stability to fly without creating the
hover trap (exp_026-038).
