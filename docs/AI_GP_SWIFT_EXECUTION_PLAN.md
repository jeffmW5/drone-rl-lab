# AI-GP Swift-Inspired Execution Plan

## Objective

Build one deployable PPO pilot that launches forward, centers the active gate,
passes gates in order, and avoids vertical runaway.

We are using Swift's useful engineering choices. We are not reproducing its
paper or running academic comparison studies.

## Chosen Architecture

- Teacher actor input: position, velocity, rotation matrix, four active-gate
  corners in the body frame, and previous action (31 values).
- Critic input: actor input plus privileged simulator state.
- Actor output: collective-thrust offset and roll, pitch, and yaw rates.
- Network: two hidden layers with 128 units.
- Reward: gate progress, gate passage, camera alignment, smooth actions, and
  crash/out-of-bounds penalties.
- Training: nominal GPU surrogate first, then dynamics randomization and
  fine-tuning using discrepancies measured from AI-GP telemetry and vision logs.

The teacher is intentionally privileged. It proves trajectory control and
produces demonstrations. It is not loaded directly into the current 18-value
live controller. Deployment requires either a state estimator that recreates
the teacher inputs or distillation into the live vision-feature actor.

## Session Plan

### Session 1: Train the Nominal Swift Teacher

- Implemented in `configs/ai_gp_002_swift_teacher_gpu_ppo.yaml`.
- Uses a Torch-vectorized environment with 4,096 parallel environments.
- Random resets include states near every gate and headings aligned toward the
  active gate.
- Reward uses gate progress, camera alignment, gate completion, rate/action
  penalties, and crash/out-of-bounds penalties.
- No alive or hover reward.

Run:

```powershell
.\scripts\runpod_ai_gp.ps1 Train `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml `
  -AutoStopPod
```

Exit gate: the remote CUDA smoke test passes, training creates checkpoints, and
deterministic evaluation shows increasing gate passage.

### Session 2: Measure the AI-GP Control Contract

- Measure thrust response, rate signs, saturation, latency, and reset behavior.
- Record time-series for every command test.
- Define the safe command envelope used by deployment.

Current status:

- command-to-telemetry analyzer implemented
- `0.02-0.14` thrust produced no measurable response in valid historical runs
- historical `0.15` run is invalid and unsafe
- historical `0.16` run is invalid because it started off-reset and collided
- stationary reference-pose preflight implemented
- next controlled test is `cal_thrust_0145_clean`

June 9 high-thrust result:

- `0.90`, `0.95`, and `1.00` were tested with forward pitch beginning `0.1 s`
  after launch
- all three saturated the response and hit the motion abort after about `1.2 s`
- observed motion was approximately `20 m` altitude rise, `29 m/s` upward
  speed, and `10 m` forward displacement
- this range is unsuitable for sustained control; useful thrust modulation must
  be below `0.90`

Detailed time-series findings are maintained in
`docs/AI_GP_CONTROL_CALIBRATION.md`.

Exit gate: repeated commands produce predictable signed motion and a reliable
reset. No policy flight before this is true.

### Session 3: Build the Gate Observation

- Upgrade the detector output from a box to four normalized gate corners where
  possible.
- Until corner detection is reliable, use the four corners of the detected box
  as the initial contract.
- Synchronize each detection with telemetry and retain confidence and age.

Exit gate: recorded sessions produce finite, time-aligned observations with
stable gate tracking across consecutive frames.

### Session 4: Distill the Teacher Into the Live Actor

- Generate teacher trajectories containing teacher state, live-style gate
  features, and teacher actions.
- Train the existing 18-value live actor to imitate teacher actions.
- Keep true gate position and full state available only to the critic.
- Remove alive/hover reward.
- Use Swift-style bilateral progress, camera alignment, action-change penalty,
  body-rate penalty, gate bonus, and crash penalty.
- Mix near-gate starts with the real race start.

Exit gate: tests pass and a CUDA smoke run completes without observation or
action-contract mismatch.

### Session 5: Validate In AI-GP And Collect Residuals

- Run a 10-million-interaction benchmark.
- Inspect deterministic trajectories, not training reward.
- Fix only demonstrated contract, reward, or environment problems.
- When behavior is directionally correct, run the 100-million-interaction job.

Promotion target:

- at least 95% gate-0 passage in deterministic surrogate evaluation
- mean of at least two gates passed
- collision rate below 20%
- no persistent vertical runaway

- Run the trained actor in receive-only shadow mode.
- Compare predicted actions and surrogate state transitions with recorded AI-GP
  transitions.
- Progress to bounded command tests, then gate-0 attempts.
- Log command, telemetry, detection, collision, reset, and gate progress.

Exit gate: enough clean time-series exist to measure command-to-motion and
observation discrepancies. Do not infer them from individual frames.

### Session 6: Fine-Tune And Race

- Fit a compact residual dynamics model from state, action, and next-state data.
- Fit measured vision noise, dropout, and delay distributions.
- Add those residuals to the surrogate.
- Fine-tune the same policy for 20 million interactions.
- Promote through shadow, bounded flight, gate 0, then multi-gate runs.

Success is measured in the AI-GP simulator: gates passed, completion rate,
collision rate, elapsed time, and repeatability.

## Compute

- One RTX 4090 or L40S RunPod
- 8 or more vCPUs
- 32 GB RAM
- 40 GB persistent disk

Expected initial budget:

- 10-million interaction benchmark: 5-20 minutes
- 100-million interaction base training: 45-120 minutes
- 20-million interaction fine-tune: 10-30 minutes

Actual time will be set by the first throughput benchmark.

## Not In Scope Yet

- raw-pixel PPO
- opponent modelling
- racing strategy adaptation
- motor-level control
- multi-GPU training
- broad experiment matrices

## Next Action

Launch `ai_gp_002_swift_teacher_gpu_ppo` on RunPod. The remote smoke test is the
first executable checkpoint because local Python currently has no PyTorch
installation.
