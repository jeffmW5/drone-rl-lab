# Pilot Agent Guide — Crazyflie Real Flight

This document is for AI agents operating `fly.py` to deploy RL policies on a
real Crazyflie 2.1 drone. Read this entire document before running any flight
command.

## Golden rule

**A human must be physically present with the drone and ready to power it off.**
Never run `hover` or `fly` without explicit confirmation from the user that
they are standing by the drone. `check` is safe to run anytime.

## Hardware setup

| Component | Details |
|-----------|---------|
| Drone | Crazyflie 2.1 (cf21B_500, 43.4g) |
| Radio | Crazyradio PA USB dongle |
| Positioning | Lighthouse V2 base stations (primary) or Flow Deck V2 |
| Radio URI | `radio://0/80/2M` (configured in `config.yaml`) |
| VM USB | VirtualBox USB passthrough required — udev rules already in `/etc/udev/rules.d/99-bitcraze.rules` |

### Pre-session checklist

1. Confirm Crazyradio PA is attached to the VM: `lsusb | grep 1915`
2. Confirm drone is powered on (user tells you)
3. Run `python3.11 real_flight/fly.py check` and verify:
   - Battery > 3.2V (3.7V+ is healthy, below 3.4V is low)
   - Position variance < 0.001 (estimator converged)
   - Z near 0.0 when on ground (Lighthouse calibrated)
   - Quaternion near [0, 0, 0, 1] when level

If any check fails, **do not proceed to flight**. Diagnose first.

## Flight modes

### `check` — Connection test (no motors)

```bash
python3.11 real_flight/fly.py check
```

Safe to run anytime. Tests radio, reads battery, position, quaternion, and
estimator variance. No motors spin.

### `hover` — PID hover (no RL policy)

```bash
python3.11 real_flight/fly.py hover
```

**The drone will take off.** Uses the high-level commander (onboard PID), not an
RL policy. Flies to `hover_height` (default 0.5m) for `hover_duration` (default
5s), then lands.

Run this first before any RL flight to verify:
- Thrust is sufficient (drone lifts off)
- Position hold is stable (Lighthouse/FlowDeck is reliable)
- Landing works cleanly
- No oscillations or drift

Ctrl+C triggers a safe landing sequence.

### `fly` — RL policy deployment

```bash
python3.11 real_flight/fly.py fly
python3.11 real_flight/fly.py fly --checkpoint path/to/model.ckpt
python3.11 real_flight/fly.py fly --no-gates
```

**The drone will take off and execute the RL policy.** Flight sequence:

1. **PID takeoff** — high-level commander lifts to hover_height (2s)
2. **Hover stabilization** — holds position for hover_duration (5s default)
3. **RL handoff** — switches to low-level attitude control, policy runs at 50Hz
4. **Ctrl+C or safety violation** — lands via high-level commander

`--no-gates` zeroes out gate/obstacle observations for simpler hover-like
behavior. Use this as an intermediate step between `hover` and full `fly`.

## Progressive test sequence

Always follow this order. Do not skip steps.

```
check  →  hover  →  fly --no-gates  →  fly
```

Each step must succeed before advancing. "Succeed" means:
- No crashes, no emergency stops
- Stable position hold (for hover / fly --no-gates)
- Smooth landing
- No safety violations in logs

If any step fails, stop and diagnose before retrying.

## Safety systems

### Geofence (automatic)

The safety monitor enforces a bounding box. If the drone exits it, motors stop.

| Axis | Min | Max |
|------|-----|-----|
| X | -2.5m | 2.5m |
| Y | -1.5m | 1.5m |
| Z | 0.0m | 2.0m |

### Attitude limits (automatic)

Policy outputs are clamped before sending to the drone:
- Roll: ±30°
- Pitch: ±30°
- Yaw rate: ±200°/s
- Thrust PWM: 20000–50000 (out of 65535 max)

### Battery cutoff (automatic)

Flight aborts if battery drops below 3.2V.

### Position variance (automatic)

Flight aborts if Kalman filter variance exceeds 0.001 (positioning unreliable).

### Manual abort

Ctrl+C at any time triggers: stop motors → switch to high-level commander → land.

### Emergency stop

On any unhandled exception, `emergency_stop()` sends a motor kill command.

## What you must NEVER do

- **Never run `hover` or `fly` without the user confirming they are by the drone**
- **Never modify safety limits** in config.yaml (geofence, attitude limits,
  battery cutoff, thrust limits) without explicit user approval
- **Never disable the tumble check** (`firmware.tumble_check: false`) unless
  the user specifically requests it for aggressive maneuvers
- **Never increase `max_thrust`** beyond 50000 PWM without user approval
- **Never skip the progressive test sequence** (check → hover → fly --no-gates → fly)
- **Never run `fly` with an untested checkpoint** — if the checkpoint changed,
  go back to `fly --no-gates` first
- **Never ignore a safety violation** — if check reports warnings, diagnose
  before flying

## What you should do

- **Always run `check` first** at the start of any session
- **Always confirm battery level** before each flight (not just at session start)
- **Always save and review flight logs** after each flight
- **Always report what happened honestly** — a crash is useful data, a hidden
  crash is dangerous
- **Wait for estimator convergence** — if position variance is high after
  `check`, wait 10-20s and rerun
- **Monitor the 2s status prints** during `fly` mode — they show position,
  battery, and gate progress

## Interpreting check output

```
Battery:  3.93 V             ← Healthy (>3.7V good, <3.4V charge soon, <3.2V won't fly)
Position: [0.12, -0.05, 0.01] ← Should be near where you placed it. Large values = bad calibration
Quaternion: [0.00, 0.01, 0.00, 1.00]  ← Should be ~[0,0,0,1] when level
Pos variance: 0.000042       ← Should be < 0.001. Higher = estimator hasn't converged
```

**Common issues:**
- Position values like [12.8, 11.2, -0.3] = Lighthouse geometry not calibrated
  to your room origin. Drone will fly but gate positions in config won't match.
- High variance after 2s warmup = Lighthouse base stations may not have line of
  sight, or FlowDeck surface is not textured enough.
- Z offset when on ground = normal if small (<0.1m), re-place drone on flat
  surface if large.

## Observation format

The policy checkpoint auto-determines its expected obs_dim. `fly.py`
constructs observations in this order:

```
pos(3) + quat(4) + vel(3) + ang_vel(3) + target_gate(1) +
gates_pos(n*3) + gates_quat(n*4) + gates_visited(n) +
obstacles_pos(m*3) + obstacles_visited(m)
```

If the constructed obs is longer than obs_dim, it truncates. If shorter, it
zero-pads. This is a workaround — the correct fix is matching the exact
training wrapper order.

If the checkpoint includes `obs_norm_mean`/`obs_norm_var` keys (exp_071+), the
normalizer is loaded automatically and applied before inference.

## Changing checkpoints

To deploy a different experiment's policy:

```bash
python3.11 real_flight/fly.py fly --checkpoint results/exp_071_obs_normalization/model.ckpt
```

Or edit `policy.checkpoint` in `config.yaml`. If the new checkpoint was trained
with obs normalization (`obs_normalize: true` in training config), set
`policy.obs_normalize: true` in config.yaml — though `fly.py` auto-detects
this from checkpoint keys regardless.

**After changing checkpoints, restart from `fly --no-gates`** to verify the new
policy doesn't immediately crash.

## Analyzing flight logs

```bash
python3.11 real_flight/analyze.py real_flight/logs/flight_YYYYMMDD_HHMMSS.npz
```

Produces a 4-panel plot (trajectory, position vs time, velocity, actions) and
prints summary stats. Use this after every flight to assess:
- Was the trajectory stable or divergent?
- Were actions smooth or oscillating?
- Did the drone hold altitude?
- How do real velocities compare to sim?

## Config reference

Config lives at `real_flight/config.yaml`. Key sections an agent might need:

| Setting | Path | Default | Notes |
|---------|------|---------|-------|
| Radio URI | `radio.uri` | `radio://0/80/2M` | Match your Crazyradio channel |
| Hover height | `control.hover_height` | 0.5m | For hover and fly pre-RL phase |
| Hover duration | `control.hover_duration` | 5.0s | Time at hover before RL handoff |
| Control freq | `control.freq` | 50Hz | Must match training env freq |
| Checkpoint | `policy.checkpoint` | exp_069 | Relative to drone-rl-lab root |
| Gate positions | `policy.gates` | level2.toml nominal | Update to measured positions |

## Firmware variable names

The Crazyflie firmware exposes state via log variables. The correct names for
this drone (verified against the onboard TOC):

| Data | Variable | Group |
|------|----------|-------|
| Position | `stateEstimate.x/y/z` | State |
| Velocity | `stateEstimate.vx/vy/vz` | State |
| Quaternion | `stateEstimate.qx/qy/qz/qw` | Attitude |
| Gyro rates | `gyro.x/y/z` | Gyro |
| Battery | `pm.vbat` | System |
| Pos variance | `kalman.varPX` | System |

Note: `stabilizer.qx/qy/qz/qw` does NOT exist in this firmware. The original
`fly.py` was written assuming it did — this was fixed to use
`stateEstimate.qx/qy/qz/qw`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `lsusb` shows no Bitcraze device | USB not forwarded | VBox menu → Devices → USB → click Bitcraze |
| Connection timeout | Drone off or wrong channel | Power cycle drone, verify URI channel |
| `Variable X not in TOC` | Wrong firmware log variable name | Check `cf_cache/*.json` for available vars |
| `Log configuration too large` | Too many vars in one LogConfig | Split into groups (max ~26 bytes = 6 floats per group) |
| High position variance | Lighthouse not converged | Wait longer, check base station visibility |
| Large position offset on ground | Lighthouse geometry not set | Run geometry calibration in cfclient |
| Drone flips on takeoff | Props on wrong motors, or Mellinger not set | Check prop direction, verify `controller: 2` |
| Policy oscillates wildly | Obs format mismatch or bad policy | Try `--no-gates` first, check obs dim logs |
