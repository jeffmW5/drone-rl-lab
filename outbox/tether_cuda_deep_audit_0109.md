# Drone research update - 13 October 2026

The current simulator can produce takeoff, but reliable straight-up sustained hover is still unachieved. We found and tested a real learner defect, improved failure recording, and are now testing a more graduated distance reward.

## New distance reward

The previous reward paid more than 1 near perfect hover, while the learner clipped it to 1. Tested 0-10 mm height and horizontal errors therefore received the same score. Reward16 removes that flat peak and uses a 1 cm smooth distance scale, with continuous credit for small corrections.

For an upright, stationary carrier at target height, these are the actual rewards after learner clipping:

| Horizontal error | Previous reward15 | New reward16 |
|---|---:|---:|
| 0 mm | 1.000000 | 1.000000 |
| 1 mm | 1.000000 | 0.998953 |
| 10 mm | 1.000000 | 0.986120 |
| 50 mm | 1.000000 | 0.914463 |

Height receives its own fine distance cost. Slowing down raises reward. Inside the square, reward tapers toward its boundary; in the 1-2 m band the airborne penalty grades from 0 to -1. Crossing 2 m remains terminal failure. Low-ground reward remains negative and the angular-rate penalty stays zero.

Twelve CPU check groups passed, including old reward14/15 compatibility, real vector dispatch with fake CPU physics, reward signs/units, square boundaries, pre-reset terminal records, and millimeter/centimeter sensitivity after clipping. This demonstrates numerical reward resolution, not learned or hardware millimeter accuracy. The reward assumes normalized simulator quaternions; a fast inverted-state taper edge case is preserved for trajectory review.

The frozen comparison uses three training-seed pairs of reward15 versus16. Each starts from the same saved weights with fresh Adam and identical ground resets, physics, observation and action versions: 128 parallel environments, 65,536 transitions, batch4096, rollout horizon32, learning rate3e-6, evaluation every16,384 transitions. Reward is paid at50 Hz. The serial daemon continues; healthy training was allowed to finish before the new queue took priority. Results are judged by true failures, takeoff, speed and sustained hover, never reward magnitude alone.

## What the broader audit established

- The learner included an uncredited final rollout column in its losses. Its opt-in correction passed independent equation tests. A complete 12-run duration/mask comparison still achieved zero sustained hover; longer runs escaped in two of three seeds for both learner variants. Fixing a real defect has not demonstrated a consistent flight benefit.
- Two GPU fixed-policy reevaluations preserved every old metric and trajectory exactly while adding pre-reset terminal records. They show genuine lateral escapes at4.34/4.36 s, about1.6 m/s horizontal speed and10 degrees tilt. Later ground frames were automatic resets.
- The first neutral-warm/signed2-degree curriculum pair also failed sustained ground hover. Both final policies escaped; all16 recovery tests eventually failed. A slightly later escape in the tilt-trained arm is provisional counterevidence, not stable recovery.
- Signed gravity/thrust/rotation tests passed. These narrow checks do not calibrate tether material, contact or attachment dynamics. Training failure alone does not establish wrong physics.

The actor receives simulator quaternions, body angular rates, particle positions/velocities, motor forces and the goal. A shared128-128 tanh network outputs collective and roll/pitch/yaw motor commands plus a value estimate. Actor and critic use the same full state; this is not asymmetric actor-critic. There is no PID helper.

The physical validation seeds currently repeat one identical initial condition, so nominal24-case evaluations are provisional. Reserved audit seeds remain untouched. Original strict1 mm obstacle navigation and hardware precision remain unachieved. The simulator is uncalibrated.
