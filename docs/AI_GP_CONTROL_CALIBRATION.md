# AI-GP Control Calibration

## Current Findings

All findings come from synchronized command and telemetry time-series.

### Pitch Sign

- command `pitch_rate=-0.10 rad/s` visually tilts forward
- command `pitch_rate=+0.10 rad/s` visually tilts backward
- telemetry pitch-rate sign is opposite the visual/control interpretation

Forward tilt remains `pitch_rate=-0.10 rad/s`. The telemetry sign must not be
used alone to label forward/backward motion.

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

## Next Test

Use thrust near `0.30`. Test a short forward-pitch (`-0.10`) pulse followed by a
positive pitch braking pulse and then zero rate. Measure whether pitch settles,
gate-relative progress continues, and vertical speed remains bounded.
