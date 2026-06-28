# AI-GP Windows Agent Prompt

You are working on the Windows-side AI Grand Prix simulator integration.

## Current Policy

Use the tracked structured-state export:

```text
exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json
```

SHA-256:

```text
1581cc4cb0a0753eb7ba87ae1e34a09dd8d7badbd048b1fce823a28775d9da60
```

This is the same export as the local generated result:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

The `results/ai_gp_*` directory is intentionally ignored by Git, so do not
expect checkpoints or full telemetry to arrive through `git pull`.

## Validation Evidence

`040` clears the structured-state surrogate threshold:

```text
seed 1001: 98.83% success, 5.98 / 6 gates, 0.00% collision
seed 1002: 99.41% success, 5.99 / 6 gates, 0.00% collision
seed 1003: 99.41% success, 5.99 / 6 gates, 0.00% collision
average:   99.22% success, 5.99 / 6 gates, 0.00% collision
nominal:   100.00% success, 6.00 / 6 gates, 0.00% collision
```

This is a structured-state simulator policy, not a camera-only/live-vision
policy.

## Runtime Contract

Load the JSON artifact and implement `observation_contract =
structured_teacher_v2`.

The actor input is 26 values in the exact order embedded in the artifact:

1. active gate relative position in body frame, scaled by `30 m`
2. active gate normal in body frame
3. next gate relative position in body frame, scaled by `30 m`
4. next gate normal in body frame
5. body-frame velocity, scaled by `8 m/s`
6. body-frame gravity unit vector
7. body angular rates, scaled by `[3, 3, 2] rad/s`
8. previous normalized action
9. active gate index divided by `gate_count - 1`

Actions are tanh-normalized:

```text
[collective_offset, roll_rate, pitch_rate, yaw_rate]
```

The artifact embeds the measured AI-GP command mapping, track gate centers,
gate normals, layer weights, and test vectors. Use the test vectors first to
verify the Windows loader before flying.

## Immediate Task

Run the exported `040` policy in AI-GP simulator shadow/integration mode and
log time-series telemetry:

- active gate index
- position, velocity, attitude, angular rates
- computed 26-value observation
- normalized policy action
- mapped simulator command
- gate-plane crossing offsets and margins
- pass/miss/collision/out-of-bounds flags

Compare those logs against the surrogate expectation:

- gate 0 passage should be effectively `100%`
- full-course completion should be near `99%` under comparable randomization
- collisions should be zero
- nominal runs should complete all six gates with healthy margins

Do not move to camera-only/live vision until this structured-state controller
transfers in the Windows simulator runtime.
