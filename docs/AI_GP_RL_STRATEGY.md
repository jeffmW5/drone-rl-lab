# AI-GP Reinforcement Learning Strategy

## Objective

Build a controller that launches immediately, keeps the active gate in view,
passes gates in order, and completes the race quickly without vertical runaway
or collisions.

The immediate target is not a full autonomous champion policy. It is:

- reliable forward launch from the real start state
- reliable gate 0 passage
- measurable improvement in gates passed and race progress
- a training/deployment interface that can be iterated safely

## Core Philosophy

### 1. Optimize the race task, not hover

Hover is a constraint and recovery behavior, not the objective. Reward forward
progress, gate passage, completion, and bounded flight. Do not pay the policy to
remain stationary.

### 2. Train the command level we can deploy

The policy action is:

1. collective-thrust offset
2. roll rate
3. pitch rate
4. yaw rate

This matches the live stack's normalized body-rate/thrust contract. Negative
pitch means forward in the current AI-GP adapter convention.

The policy does not learn raw motor mixing yet. Motor-level learning adds
actuator complexity before the simulator control response is understood.

### 3. Keep the actor sensor-realistic

The actor receives only features available at deployment:

- body-frame velocity
- gravity direction in body frame
- body angular rates
- detected gate center, area, confidence, and age
- previous action

It does not receive the true gate position or simulator track coordinates.

### 4. Give the critic privileged state

During training, the critic additionally receives true relative gate position,
distance, world position/velocity, attitude, and gate index. This asymmetric
actor-critic design improves value estimation without creating a deployment
dependency.

### 5. Train visual control features before raw pixels

The first PPO policy consumes gate-detection features, not images. Pixel PPO is
far more expensive and usually sample-inefficient. Perception and control can be
improved independently while saved simulator frames are collected.

### 6. Use curriculum without hiding the real start

Early training mixes easy near-gate spawns with ground starts. The mix moves
toward mostly real starts as training progresses. This teaches gate approach
without repeating the previous mistake of evaluating a policy on a start state
it never saw during training.

### 7. Adapt from measured residuals

Initial training uses a nominal model with narrow parameter variation. After
controlled simulator runs, fit residual observation and dynamics models from
synchronized commands, telemetry, and vision. Sample those measured residuals
during fine-tuning.

Broad domain randomization remains a fallback, not the primary transfer method.
This follows Swift's finding that empirical residual models preserve more racing
performance than generic randomization.

### 8. Evaluate behavior, not training reward

Checkpoint selection is ordered by:

1. deterministic race completion rate
2. mean gates passed
3. distance reduction toward the active gate

Training reward is diagnostic only. Prior racing experiments proved that higher
reward can coexist with zero benchmark gates.

### 9. Promote policies in stages

A checkpoint is not immediately flight-ready:

1. deterministic surrogate evaluation
2. replay and observation-contract tests
3. live receive-only shadow predictions
4. short calibrated command tests with hard abort limits
5. gate 0 attempts
6. multi-gate runs

## Current PPO Approach

The `ai_gp` backend uses one Torch device for both environment simulation and
PPO. This avoids the JAX-to-Torch transfer and external `lsy_drone_racing`
dependency.

Initial design:

- 4,096 parallel environments
- 50 Hz policy step with two physics substeps
- asymmetric 2x128 actor/critic
- PPO with clipped objective and GAE
- progressive near-gate-to-race-start curriculum
- soft collisions early, hard termination later
- deterministic evaluation from the race start

The surrogate is intentionally simple. Its purpose is fast policy iteration,
not claiming exact simulator fidelity. See `docs/SWIFT_ADOPTION_REPORT.md` for
the teacher-student and residual-fine-tuning plan derived from Swift.

## Reward Design

Primary terms:

- bilateral distance progress toward the active gate
- gate passage bonus
- race completion bonus
- small velocity reward only when moving toward the gate

Constraints:

- collision and out-of-bounds penalties
- excess vertical speed and altitude penalties
- angular-rate and action-change penalties
- near-zero alive reward

The alive reward stays small to avoid recreating the hover local optimum.

## Alternatives

### Scripted visual servo controller

Best short-term baseline and safety reference. It may pass early gates without
RL. It will likely become brittle at speed and through changing gate geometry.

### Residual RL

Recommended alternative if direct PPO remains unstable. A scripted controller
provides launch, altitude envelope, and basic centering; PPO learns bounded
corrections. This reduces exploration risk and transfer burden.

### Teacher-student imitation

Train a privileged teacher in the surrogate, collect trajectories, then train a
sensor-only student. This is useful if asymmetric PPO does not produce a strong
deployable actor.

### Behavior cloning from simulator runs

Useful after successful scripted or human-controlled segments exist. It can
bootstrap launch and gate approach, then PPO can fine-tune.

### Model-based RL or Dreamer

Potentially better for pixel observations and limited data, but materially more
complex. It is not the first implementation while the control mapping remains
unknown.

### SAC/TQC

More sample-efficient off-policy alternatives. They are harder to scale cleanly
across thousands of synchronous environments and can be less stable for the
current first-pass workflow.

### End-to-end pixel policy

Not recommended yet. It combines perception, dynamics, and control failures
into one opaque training problem.

### Simulator-in-the-loop RL

Highest fidelity, but only practical after headless launch, deterministic reset,
parallel instances, and reliable automation exist. The current GUI simulator is
better used for validation and data collection.

## What We Can Change

- action level: body rates versus motors or residual commands
- actor inputs: detector features, gate corners, optical flow, telemetry history
- network: width, recurrence, temporal convolution, separate visual encoder
- algorithm: PPO, recurrent PPO, SAC, Dreamer, imitation, residual RL
- curriculum: spawn mix, track complexity, speed targets, collision hardness
- reward weights and sparse/dense balance
- dynamics fidelity and domain-randomization ranges
- policy/control rate and action latency
- deterministic checkpoint selection criteria
- track geometry and start-state distribution

Change one major variable per experiment when possible.

## Material Unknowns

### Blocking live transfer

- calibrated hover thrust and safe thrust range
- actual body-rate response, signs, saturation, and latency
- whether attitude-rate control is the final safe live interface
- simulator coordinate and gravity conventions under command

### Environment fidelity

- exact qualifier track geometry and gate dimensions
- camera FOV, image orientation, latency, and detector error distribution
- collision geometry and gate-passage rules
- start-state variability and any initial race velocity
- aerodynamic behavior at racing speed

### Training validity

- whether surrogate success predicts simulator gate passage
- whether PPO learns launch and navigation jointly or needs residual/imitation
- required network memory for temporary gate loss
- useful randomization ranges versus destructive randomization
- throughput and convergence rate on the selected RunPod GPU

## Decision Gates

- Do not map collective policy output to live thrust until calibration exists.
- Do not call a checkpoint improved based only on reward.
- Do not increase dynamics complexity until deterministic gate metrics improve.
- Switch to residual RL if direct PPO cannot reliably pass gate 0.
- Add recurrence only after time-series evaluation shows memory is required.
- Move toward pixels only after feature-policy performance is established.

## Immediate Execution Order

1. Capture a uniquely named 30-second receive-only simulator session.
2. Export it with `scripts/export_ai_gp_session.py`.
3. Verify detector coverage, persisted JPEG count, and telemetry age.
4. Use repeated captures to estimate vision noise, dropout, and latency.
5. Complete body-rate/thrust calibration from time-series.
6. Fit residual observation and dynamics models from those measurements.
7. Run the RunPod CUDA smoke test and one-gate PPO training.

See `docs/AI_GP_DATA_BRIDGE.md` for exact commands.
