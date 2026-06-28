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

## 041 Windows A/B Candidate

Also test the tracked `041` export against `040`:

```text
exports/ai_gp/ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json
```

SHA-256:

```text
583762c48fa7e24a7a5ea69dfa1269104a54481b61f3282143e9eaabb4f42ca7
```

`041` was trained from `040` on the Windows gate-1/gate-2 hard-case envelope.
It is not a clean surrogate promotion over `040`: randomized surrogate average
success is `98.83%` for `041` versus `99.22%` for `040`, with zero collisions
for both. Use `041` only as a Windows simulator A/B candidate. Promote it only
if Windows testing shows it clears farther than active gate index `2`.

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

## Windows Watch Loop

Use the simulator venv Python and the structured runner for repeated visible
attempts:

```powershell
& 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe' -B scripts\run_ai_gp_structured_windows.py `
  --continuous `
  --duration 30 `
  --target-gates 0 `
  --allow-gate-plane-miss `
  --thrust-multiplier 1.10 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id watch_structured_040_YYYYMMDD_HHMMSS
```

Each attempt resets the simulator and writes a separate replay session under:

```text
tmp/ai-grand-prix-stack-remote/replay/sessions/
```

Use `--target-gates 0` for watching beyond gate 0. `--target-gates 1` is only a
gate-0 smoke test and intentionally resets immediately after the simulator
advances to gate 1.

With full-run watching enabled, aborts after gate 0 are collision-driven, not
target-gate stops. In the first full-run samples with `1.10` thrust, `2.00`
roll-rate, and `2.00` yaw-rate, most collisions occurred while the simulator's
active gate was still `1`; several were then credited as gate-1 passes within
roughly `0.02-0.15 s`. Treat this as clipping/hitting gate 1 during the crossing
unless `active_gate_index` was already `2` before the collision.

June 28 Windows sim status: the structured runner loads and commands the `040`
policy, but early active attempts did not pass gate 0. One run missed the gate
plane; a later run collided near gate 0 before the simulator advanced the active
gate index. The simulator event log reported gate quaternions equivalent to
90-degree yaw for all gates, while the JSON export uses inferred segment
normals, so gate orientation/normal mismatch is the first transfer issue to
check.

For live watching, runtime overrides can offset visibly sluggish transfer
behavior. Current watch settings are `1.10` thrust, `2.00` roll/yaw rate, and
`1.00` pitch-rate. These are Windows runtime test overrides, not training-side
policy changes.

`--use-sim-gate-normals` is a reversible runtime A/B flag. It uses simulator
`track.gate` quaternions for gate normals instead of the normals embedded in the
JSON export. It does not retrain or modify the policy artifact. The June 28
y-axis sim-normal A/B run was worse than exported normals, so do not use it as
the default watch setting.

June 28 tuning/sweep result: tuning is not the answer for `040`. The best
12-config sweep result used thrust `1.12`, roll `2.00`, pitch `1.00`, yaw
`2.00`, reached active gate index `2`, and still failed before clearing gate 2.
Use the Linux handoff in
`docs/AI_GP_TRANSFER_TRAINING_HANDOFF_2026_06_28.md` for the next `041`
training run and result.

June 28 `041` result: the RunPod hard-case training completed and exported a
Windows A/B JSON. Use the same full-run command pattern, changing `--policy` to
the `041` export path and keeping the best sweep baseline multipliers:

```powershell
& 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe' -B scripts\run_ai_gp_structured_windows.py `
  --policy exports\ai_gp\ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json `
  --continuous `
  --duration 30 `
  --target-gates 0 `
  --allow-gate-plane-miss `
  --thrust-multiplier 1.12 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id watch_structured_041_YYYYMMDD_HHMMSS
```

For the actual promotion decision, prefer the dedicated A/B runner. It runs
`040` and `041` under identical conditions, writes a ranked JSON summary under
`tmp/`, includes policy SHA-256 values and collision hard cases, and applies
the Windows retest target:

```powershell
& 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe' -B scripts\run_ai_gp_policy_ab_windows.py `
  --attempts-per-policy 5 `
  --duration 30 `
  --thrust-multiplier 1.12 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id structured_ab_040_041_YYYYMMDD_HHMMSS
```

Promote only if the A/B summary shows `041` beats `040` and satisfies:

```text
gate0_pass_rate >= 0.90
mean_max_gate > 2.0
best_max_gate >= 3
```

The hard-case entries include the 26-value structured observation,
`observation_features`, normalized policy action, mapped command, gate-plane
offsets, and collision context. Preserve that summary; it is the input for the
next transfer-training loop if neither policy passes.

The structured runner does not use camera imagery, camera intrinsics, or a
camera tilt/extrinsic model. Vision is disabled. It does subtract the initial
telemetry pitch as a body-frame reference, but that is not the same as modeling
the simulator camera's upward tilt. Any camera-only/live-vision runner must
explicitly account for the simulator camera tilt in its projection or detector
feature contract.
