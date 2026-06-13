# AI-GP State Estimation Benchmark - 2026-06-13

## Scope

This work kept the privileged Swift teacher separate from every deployable
student. It added and screened:

- sixteen-frame box history
- per-frame GRU state estimation
- measured four-corner history
- measured corners plus a GRU
- explicit optical center/expansion motion
- a randomization curriculum
- a deployable action governor

All GPU work ran on a RunPod RTX A5000 after CUDA smoke tests. The seven
2.10M-interaction screens completed before three candidates were promoted to
10.09M interactions.

## Observation And Deployment Changes

- `corner_temporal_live_v1`: ordered TL/TR/BR/BL image corners, validity,
  confidence, age, telemetry, and prior action
- `recurrent_live_v1` and `corner_recurrent_live_v1`: GRUCell students with
  explicit reset state and exported recurrent test sequences
- `motion_live_v1`: one 20D live frame plus gate-center and logarithmic
  width/height deltas
- artifact-declared normalized action governors using only body velocity,
  gravity, and prior executed action
- deterministic telemetry reports with episode paths and exact gate-plane
  crossing events

## Screening

The recurrent students diverged vertically by 2.10M interactions. Longer box
history peaked at `0.77` mean gates, measured-corner history at `1.03`, and
optical motion at `1.29`. The first corner governor reduced vertical runaway
but converted failures into collisions. These were screening results, not
promotion claims.

## 10M Benchmarks

Independent 256-episode nominal trajectory evaluation of selected checkpoints:

| Experiment | Mean gates | Completion | Collision | Bounds | Vertical |
|---|---:|---:|---:|---:|---:|
| `ai_gp_014_long_box_history_mlp_10m` | 1.76 | 0.8% | 0.4% | 51.2% | 54.3% |
| `ai_gp_015_corner_temporal_mlp_10m` | 2.40 | 0.0% | 87.9% | 12.1% | 12.1% |
| `ai_gp_016_optical_motion_mlp_10m` | 1.61 | 18.8% | 43.8% | 37.5% | 11.3% |
| `ai_gp_017_motion_safety_ppo_10m`, ungoverned | 3.43 | 63.3% | 16.0% | 20.7% | 1.6% |
| `ai_gp_017_motion_safety_ppo_10m`, governed | 3.99 | 99.6% | 0.0% | 0.4% | 0.0% |

The governed checkpoint used slew limits `[0.15, 0.25, 0.25, 0.35]`, upward
braking above `1.0 m/s`, and gain `0.10`.

The result is based on time-series telemetry, not reward:

- 1,022 exact gate-plane crossings
- all 256 episodes crossed gates 0 and 1
- 255 crossed gates 2 and 3 and completed
- one episode exited bounds after two gates
- minimum crossing margin: `10.5 mm`
- altitude p95: `2.07 m`

Randomized seed 10702 remains weaker: `94.1%` gate-0 passage, `2.73` mean
gates, `44.9%` completion, `22.7%` collisions, `32.4%` bounds failures, and
`11.7%` vertical runaway. This is a Windows evaluation candidate, not a
robustness or racing claim.

## Measured Corners

The Windows capture
`rl_capture_20260607_202413` contains 180 detection-bearing records. The
OpenCV JPEG path matched the recorded primary gate and returned ordered corner
geometry for 179/180 records at about 217 frames/s on this VM. The p95 measured
corner displacement from the axis-aligned box was `0.0389` normalized image
units, confirming that exported corners are measured geometry rather than
fabricated box corners.

## Windows Boundary

The governed JSON artifact is
`ai-grand-prix-stack/policy/models/ai_gp_017_motion_safety_governed.json`.
Its status is `surrogate_passed_pending_windows_simulator`; command eligibility
is false.

Receive-only evaluation on the recorded Windows capture processed 147
synchronized rows with finite actions. The governor intervened on `3.4%` of
rows, and no command was sent.

The next required evidence is active Windows simulator time-series evaluation.
Do not describe this surrogate completion result as Windows simulator gate
passage.
