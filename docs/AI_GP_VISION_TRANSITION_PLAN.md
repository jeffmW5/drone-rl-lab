# AI-GP Vision Transition Plan

## Current State

The `040` controller is a structured-state teacher/runtime test. It uses known
gate centers, gate normals, telemetry pose, velocity, attitude, angular rates,
and active gate index. It does not use camera frames.

This is useful for:

- validating control authority and simulator transfer
- collecting failure logs
- generating teacher labels
- benchmarking dynamics and command scaling

It is not the final race stack if the competition runtime withholds reliable
position and gate coordinates.

## Vision Target

Build a vision/localization stack that estimates the active gate from the FPV
camera plus IMU/body-rate telemetry, then feeds either:

- a learned live-vision policy, or
- a vision-estimated gate-pose controller distilled from the structured teacher

The camera model must include the simulator camera tilt/extrinsics. The current
structured runner does not model camera tilt because vision is disabled.

## Transition Steps

1. Keep structured `040` as the oracle teacher and control baseline.
2. Capture synchronized sessions with telemetry, commands, race status, and
   vision frames.
3. Calibrate the camera projection, including the upward camera tilt and FOV.
4. Train or implement gate detection from frames: center, size, corners, and
   confidence.
5. Convert detections into relative gate pose or image-feature observations.
6. Distill teacher actions onto vision-derived observations.
7. Run shadow mode: vision policy predicts actions while structured policy flies.
8. Run bounded active vision tests gate-by-gate.
9. Fine-tune on real simulator failures with DAgger/hard-case aggregation.
10. Remove structured-state dependencies before final race-style validation.

## Training Vision Pilots

Use learning-by-cheating: train with privileged state, deploy without it.

1. Fly the structured teacher while recording camera frames and telemetry.
2. Auto-label frames offline using the known simulator gate geometry, camera
   intrinsics, and 20-degree upward camera tilt.
3. Train a gate detector/keypoint model to output gate center, size, corners,
   and confidence.
4. Train a vision student policy from detector features plus telemetry to match
   the structured teacher's actions.
5. Run the student in shadow mode while the structured teacher flies.
6. Collect disagreement/failure cases and relabel them with the teacher.
7. Fine-tune the student with the expanded dataset.
8. Promote to bounded active tests one gate at a time.

Prefer a staged student before raw end-to-end pixels:

- `image -> gate detector -> temporal gate features + telemetry -> policy`

This keeps training debuggable. End-to-end image policies can come later after
the detector-based policy is stable.

## Promotion Gates

- Gate detector runs at control-relevant latency.
- Vision shadow policy agrees with structured teacher near gates.
- Active vision runner passes gate 0 repeatedly.
- Active vision runner beats structured-runtime baseline under race-like sensor
  constraints.
- No dependency remains on ground-truth gate centers, active gate index, or
  simulator pose if those are not available in the official runtime.
