# AI-GP Data Bridge

## Purpose

Convert a real receive-only simulator session into the same 18-value actor
observation used by PPO.

## Capture

Start the simulator and enter an active session where telemetry and vision are
streaming. Then run:

```powershell
cd C:\Users\JefferyWhitmire\Desktop\Shared\drone-rl-lab\tmp\ai-grand-prix-stack-remote
$runId = "rl_capture_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& "C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe" `
  .\run_manager.py `
  --manifest .\run\manifests\cal_receive_only.json `
  --run-id $runId `
  --duration 30 `
  --dataset-capture `
  --print-summary
```

This is receive-only. It does not arm or send flight commands.

## Export

From the lab root:

```powershell
& "C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe" `
  .\scripts\process_ai_gp_capture.py `
  ".\tmp\ai-grand-prix-stack-remote\replay\sessions\$runId"
```

Outputs:

- `rl_features.jsonl`: one synchronized actor observation per vision frame
- `rl_features_summary.json`: row count, detection coverage, persisted frames,
  and telemetry synchronization age

Detection runs after capture so image decoding cannot block UDP packet intake.

## Required Acceptance Checks

- `persisted_frame_count > 0`
- `full_detection_rows > 0`
- `row_count > 0`
- `max_telemetry_age_s <= 0.10`
- each `actor_observation` contains 18 finite values

## Coordinate Contract

Telemetry is currently interpreted as NED world coordinates with an FRD body
frame. Velocity and gravity are converted into the policy's FLU body frame.
Angular rates retain the adapter signs because the live command contract uses
those signs directly.

This convention is an explicit hypothesis. It must be verified from
time-series roll, pitch, and yaw calibration before policy commands are enabled.

## Existing Sessions

Sessions recorded before June 6, 2026 contain detection counts but not gate
boxes/confidence and do not persist JPEG bytes. They can validate timestamp
joining, but they cannot measure the real visual feature distribution.
