# AI-GP Control Calibration

## Current Findings

All findings come from synchronized command and telemetry time-series.

### Pitch Sign

- command `pitch_rate=-0.10 rad/s` visually tilts forward
- command `pitch_rate=+0.10 rad/s` visually tilts backward
- telemetry pitch-rate sign is opposite the visual/control interpretation

Forward tilt remains `pitch_rate=-0.10 rad/s`. The telemetry sign must not be
used alone to label forward/backward motion.

### Roll Sign

On June 14, six legal-start trials applied symmetric `+/-0.30 rad/s` roll
commands for `0.35 s` at thrust `0.30`.

| Command | Gate center X change | NED Y velocity change | Roll angle change |
|---|---:|---:|---:|
| `+0.30` | `-0.0773` | `-0.3321 m/s` | `+0.2045 rad` |
| `-0.30` | `+0.0754` | `+0.3065 m/s` | `-0.2013 rad` |

The values are medians across three trials per direction. Positive actuator
roll moves a gate left in the image. Therefore a policy requesting positive
roll for a gate right of center must be passed to the simulator without a roll
sign inversion.

Reported roll-rate has the opposite sign from command and roll-angle response:
`+0.30` produced about `-0.768 rad/s`, while `-0.30` produced about
`+0.768 rad/s`. Do not use raw roll-rate sign alone to change actuator mapping.

The trained surrogate uses the same actuator convention. At zero yaw, positive
roll tilts thrust toward negative world Y, which is toward a gate on the image
right under the environment projection. A direct policy probe also increased
roll monotonically from `-0.131` at gate-center X `-0.8` to `+0.130` at
gate-center X `+0.8`. The end-to-end roll mapping is therefore identity.

### Current Policy Failure

The June 14 identity-mapping batch repeated the same gate-0 collision three
times. The collision position was approximately `y=+0.48 m`; successful prior
runs crossed near `y=-0.49 m`, close to gate-0 center `y=-0.40 m`.

This is a policy generalization failure, not an actuator sign failure:

- the policy requests negative roll during the low-speed launch even with the
  gate slightly right of image center
- its training track moves from gate 0 at `y=0.0` to gate 1 at `y=+1.5`
- the simulator track moves from gate 0 at `y=-0.4` to gate 1 at `y=-2.5`
- the learned pre-turn is therefore opposite the real track transition

Flipping the live actuator sign makes the memorized pre-turn pass gate 0 by
accident, then makes the controller turn away from gate 1. Keep the measured
identity mapping and retrain the policy on the real track topology or on
randomized transition directions.

### Uniform-Authority Full Runs

Batch `full_policy_uniform_batch10` applied the former gate-1 settings from the
first command with no pre-gate-0 authority rule:

- track-pose observations
- `50 Hz` policy and command rate
- roll/pitch/yaw limits `0.20/0.20/0.15 rad/s`
- thrust spans `+0.02/-0.15`
- governor slew scale `4`
- measured identity roll mapping

All three attempts missed gate 0 without collision. At the gate plane, lateral
error was `+3.47` to `+4.73 m` and vertical error was `+2.90` to `+3.71 m`.
Applying the high-authority settings from launch amplifies the current policy's
launch bias. It does not reproduce the batch-06 gate-1 approaches.

### Timed Authority Switch

Hard switches after `0.5`, `1.0`, and `2.0 s` also missed gate 0. The visible
sharp down/lateral movement is caused by discontinuous action mapping, not a
new policy decision. Around the `2.0 s` switch:

| Command | Before | After |
|---|---:|---:|
| thrust | about `0.290` | `0.188-0.229` |
| roll rate | about `-0.0027` | `-0.095` to `-0.112 rad/s` |
| pitch rate | about `-0.052` | `-0.140` to `-0.152 rad/s` |
| yaw rate | about `+0.0014` | about `+0.021 rad/s` |

The same normalized negative collective and roll outputs are suddenly mapped
through much larger spans. Command rate also changes from `12.5` to `50 Hz`.
The thrust cut reverses vertical velocity toward descent after response delay,
while the roughly 40x roll-command increase creates the sharp lateral motion.
Any future authority transition must interpolate physical command limits and
thrust spans over time instead of switching them in one command.

Batch `full_policy_ramped_launch_batch14` kept the restrained launch for
`2.0 s`, then interpolated all authority settings over `1.0 s`. The largest
per-command changes during the ramp were approximately:

- thrust: `0.009`
- roll rate: `0.017 rad/s`
- pitch rate: `0.019 rad/s`
- yaw rate: `0.002 rad/s`

The sharp switch motion was removed. All three runs still missed gate 0, but
the measured plane errors improved to `0.78-0.84 m` lateral and
`0.51-0.58 m` vertical. This is substantially closer than either the hard
switch or immediate high-authority batches. The remaining error is trajectory
tracking, not transition discontinuity.

### Authority And Thrust-Gain Matrices

The June 14 authority matrix ran three attempts each at 50% and 100% release of
the thrust governor, directional governor, and both governors. None of the 18
attempts passed gate 0.

- thrust release ended in collision in all six attempts
- directional release ended in tilt abort in all six attempts
- combined release produced five collisions and one tilt abort

The separate `125%`, `150%`, and `200%` policy-thrust-gain matrix also ran three
attempts per setting. None passed gate 0. All nine reached the configured
gate-plane miss termination without collision.

Increasing policy authority does not correct the trajectory. It amplifies the
existing policy and surrogate-track mismatch.

### Continuing After A Missed Gate

Batch `full_policy_gain200_no_plane_abort_batch15` disabled only the gate-plane
miss abort and repeated the `200%` thrust-gain configuration three times.
Collision, tilt, altitude, and speed protections remained active.

All three attempts missed gate 0 and continued to approximately `x=-44 m`,
`y=-9 m`. They then hit the altitude-rise safety limit after climbing about
`4.5-4.7 m`.

The time-series exposed a controller defect:

- track-pose projection returned no observation after gate 0 moved behind the
  drone
- policy updates stopped near `x=-22.1` to `-22.9 m`
- the runner held the last controller command for `2.7-4.1 s`
- the stale command drove the later lateral and vertical excursion

Allowing a gate-plane miss is diagnostic only. A deployable controller must
explicitly handle lost or missed gates by terminating, entering a bounded
recovery mode, or advancing only when the simulator reports a valid gate pass.
It must never hold a non-neutral command indefinitely when observations stop.

### Race Start

The June 9 calibration runner armed and began its command clock before the
simulator's legal race-start signal. The simulator disqualified those runs for
early start. Their command-to-motion data remains useful, but they are not valid
race attempts.

Future flight runs must wait until race status reports a valid start before
arming or sending motion commands.

### Thrust With Pitch `-0.10`

| Thrust | Result |
|---|---|
| `0.50` | aborted near 2.2 s; 20.7 m climb, 20.3 m/s upward |
| `0.55` | aborted near 1.9 s; 21.5 m climb, 21.8 m/s upward |
| `0.60` | aborted near 1.7 s; 21.7 m climb, 23.4 m/s upward |
| `0.90-1.00` | saturated; aborted near 1.2 s around 20 m climb |

### Thrust With Backward Pitch `+0.10`

| Thrust | Result |
|---|---|
| `0.30` | completed 5 s; 10.6 m climb, peak 9.2 m/s upward |
| `0.35` | aborted near 3.6 s; 20.2 m climb |
| `0.40` | aborted near 2.7 s; 20.5 m climb |

`0.30` completing the experimental window does not make it safe. It still
climbs far too quickly for a racing controller.

## Current Control Map

- normalized thrust range is `0.0-1.0`
- `0.145` produces no meaningful motion
- `0.20` produces measurable motion
- sustained `0.30+` produces excessive vertical motion
- useful operation is likely between `0.20` and `0.30`
- body-rate response begins roughly `0.12-0.18 s` after the pitch command

### Legal Forward-Tilt Runs

On June 9, the runner was fixed to wait through the three-second countdown.
Thrust `0.30`, `0.35`, and `0.40` were then run for five seconds with actual
forward pitch `-0.10 rad/s`.

| Thrust | X displacement | Peak upward speed | Pitch change |
|---|---:|---:|---:|
| `0.30` | `-55.7 m` | `17.4 m/s` | `+1.22 rad` |
| `0.35` | `-64.6 m` | `19.6 m/s` | `+1.22 rad` |
| `0.40` | `-72.7 m` | `21.6 m/s` | `+1.22 rad` |

The runs were legal and collision-free, but not controlled flight. Holding a
body-rate command continuously causes the attitude to keep rotating. A usable
launch must use a short pitch pulse followed by an opposite braking pulse or
attitude feedback.

## Next Work

The six-gate topology and missed-gate termination are now implemented. The
first topology-correct teacher and one reward-corrected rerun both failed
promotion; the better checkpoint reached `11.3%` nominal gate-0 passage and no
gate-1 passages.

Stop increasing live command authority or tuning the current surrogate reward.
Fit surrogate dynamics only from synchronized command and telemetry time-series,
then retrain the structured-state teacher.
