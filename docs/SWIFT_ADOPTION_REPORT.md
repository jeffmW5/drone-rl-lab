# Swift Adoption Report

## What Swift Actually Does

Swift is a hybrid perception, estimation, and reinforcement-learning system. It
does not train PPO directly from camera pixels.

- A gate network detects four gate corners.
- Gate geometry, a known track map, VIO, PnP, and a Kalman filter produce a
  stable state and relative gate pose.
- PPO consumes a normalized 31D observation: estimated state, four relative
  gate corners, and the previous action.
- The policy outputs collective thrust and body rates.
- Actor and critic are two-layer MLPs with 128 units per layer.
- Reward combines gate progress, camera alignment, smooth commands, and crash
  termination.
- Training starts agents near random gates and previously observed gate-passing
  states instead of repeatedly learning the whole launch sequence.
- Initial training uses the nominal simulator. Transfer is handled by fitting
  residual models from real rollouts, then fine-tuning the policy.

Published training used 100 parallel agents, 100 million interactions, and
about 50 minutes on an RTX 3090. Fine-tuning used another 20 million
interactions. Residual identification used three complete runs, about 50
seconds of flight, and 800-1,000 dynamics samples.

## How We Should Adapt It

Swift had a known track map and strong VIO. AI-GP currently gives us telemetry
and images, but our detector provides a box rather than reliable gate corners
and relative 3D pose. Copying Swift's policy inputs directly would therefore
create a deployment mismatch.

Use this architecture:

1. Train a privileged teacher with full simulated state, relative 3D gate
   corners, and previous action.
2. Upgrade perception to return four 2D corners, confidence, and age. Estimate
   relative gate pose when gate dimensions and camera calibration are known.
3. Train a deployable student from telemetry, visual gate features, and a short
   history by imitating the teacher.
4. Fine-tune the student with PPO using the same progress, perception,
   smoothness, and crash objectives.
5. Fit observation and dynamics residuals from synchronized AI-GP
   command/telemetry/vision logs, then run a 20-million-step fine-tune.

If legal track coordinates and a reliable global pose are available, skip
teacher-student distillation and deploy a Swift-style state-and-gate-pose actor
directly.

## Training Program

- Stage 0, validation: verify action signs, hover thrust, saturation, latency,
  resets, gate passage, and synchronized logging.
- Stage 1, teacher PPO: 5-10 million interaction smoke run, then 100 million
  interactions with three deterministic seeds.
- Stage 2, perception/student: train corner detection or pose estimation on
  saved and synthetic frames; distill teacher trajectories into the student.
- Stage 3, measured adaptation: collect complete and controlled simulator runs,
  fit observation residuals and command-to-motion residual dynamics, then
  fine-tune for 20 million interactions.
- Stage 4, promotion: require gate-0 completion, multi-gate completion, bounded
  altitude, and deterministic improvement before live command authority.

The current captured stream supports a roughly 50 Hz control policy while
holding the most recent visual feature and its age. Unique images arrive more
slowly than telemetry, which is normal and matches Swift's separation of fast
state updates from slower gate detections.

## Compute And Time

One GPU is enough. Multi-GPU training is unnecessary for this network.

Recommended RunPod class:

- RTX 4090 24 GB or L40S 48 GB
- 8 or more vCPUs
- 32 GB RAM
- 40 GB persistent disk

Expected control-training time must be benchmarked on our implementation:

- 5-10 million step smoke run: 5-20 minutes
- 100 million step base PPO run: 45-120 minutes
- 20 million step residual fine-tune: 10-30 minutes
- three seeds plus focused ablations: 8-16 total GPU-hours

A compact supervised gate-corner detector should fit on the same GPU. Budget
roughly 2-8 GPU-hours after a usable labelled or synthetic dataset exists.
Dataset construction and simulator validation will take longer than the neural
network training.

## Immediate Checklist

- [x] Add a privileged teacher contract.
- [ ] Add four-corner live gate observations or a reliable gate-pose estimate.
- [x] Add Swift-style camera-alignment reward and remove unnecessary alive
      reward.
- [x] Add deterministic random-gate and race-start evaluation suites.
- [ ] Log every command with synchronized telemetry and visual detections.
- [ ] Implement residual dynamics fitting and residual observation sampling.
- [x] Benchmark 10 million interactions on one RTX 3090 before reserving the
      full training budget.

The June 13 temporal benchmark added separate box width/height plus four-frame
history. It improved multi-gate progress but did not pass bounds or vertical
criteria. Raising student-state DAgger to 100% caused collision collapse, so
the next experiment must improve state estimation rather than only increasing
student rollout exposure.

The active implementation sequence is defined in
`docs/AI_GP_SWIFT_EXECUTION_PLAN.md`.

## Sources

- [Nature paper](https://www.nature.com/articles/s41586-023-06419-4)
- [Published pseudocode and data](https://zenodo.org/records/7955278)
