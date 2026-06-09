# AI-GP Agent Handoff

## Objective

Train a Swift-inspired PPO teacher in a fast GPU surrogate, then transfer its
gate-flight behavior into the AI Grand Prix live vision controller.

Do not return to one-off manual thrust pulses as the primary workstream.
Simulator runs are for measuring the real control contract and validating
trained policies with time-series telemetry.

## Current Implementation

- `ai_gp_rl/env.py`: Torch-vectorized quadrotor and gate environment.
- `ai_gp_rl/model.py`: asymmetric PPO actor-critic with two 128-unit layers.
- `ai_gp_rl/contract.py`: live 18D contract and Swift teacher 31D contract.
- `train_ai_gp.py`: PPO training, evaluation, metrics, and checkpoints.
- `configs/ai_gp_002_swift_teacher_gpu_ppo.yaml`: active training config.
- `scripts/runpod_ai_gp.ps1`: deploy, smoke, train, monitor, and pull workflow.
- `tmp/ai-grand-prix-stack-remote/`: local simulator-control worktree. It is
  intentionally ignored because it contains generated sessions and captures.

The Swift teacher observes:

1. position and velocity
2. body-to-world rotation matrix
3. four active-gate corners in the body frame
4. previous action

It outputs normalized collective-thrust offset and roll, pitch, and yaw rates.
The reward contains gate progress, camera alignment, gate/finish bonuses,
body-rate and action-change penalties, and crash/out-of-bounds penalties.

## Critical Architecture Boundary

The 31D teacher is privileged and cannot be loaded directly into the current
18D live controller.

Transfer requires one of:

- build a live estimator that recreates the teacher state and gate corners
- collect teacher trajectories and distill actions into the 18D live actor

The recommended next implementation after teacher training is trajectory
export plus behavior-cloning distillation into the live actor.

## Immediate Next Action

From PowerShell:

```powershell
cd "C:\Users\JefferyWhitmire\Desktop\Shared\drone-rl-lab"

$env:RUNPOD_API_KEY = "..."
$env:RUNPOD_POD_ID = "..."
$env:DRONE_RL_DEPLOY_KEY = "$HOME\.ssh\id_ed25519_runpod"

.\scripts\runpod_ai_gp.ps1 Train `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml `
  -AutoStopPod
```

The launcher uploads a minimal bundle, verifies CUDA, runs a 256-environment
forward/backward smoke test, and starts the job.

Monitor:

```powershell
.\scripts\runpod_ai_gp.ps1 Status `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml `
  -Experiment ai_gp_002_swift_teacher_gpu_ppo

.\scripts\runpod_ai_gp.ps1 Logs `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml `
  -Experiment ai_gp_002_swift_teacher_gpu_ppo
```

Retrieve:

```powershell
.\scripts\runpod_ai_gp.ps1 Pull `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml `
  -Experiment ai_gp_002_swift_teacher_gpu_ppo
```

## First-Run Checks

- CUDA smoke test succeeds.
- Observation shape is `(N, 45)`: 31 actor values plus 14 critic values.
- Training logs remain finite.
- Evaluation mean gates and gate-0 passage improve.
- Collision rate decreases.
- Deterministic trajectories move through gates rather than exploit reward.

Do not judge the policy from reward alone.

## Validation Status

- Python syntax compilation passed.
- Non-Torch contract tests passed.
- Local Torch rollout was not run because neither local Python environment has
  PyTorch installed.
- The RunPod CUDA smoke test is therefore the next required validation.

## Known Unknowns

- Placeholder gate positions do not yet match the AI-GP track.
- Surrogate thrust and rate dynamics are not fitted to clean live telemetry.
- Live four-corner gate detection is not implemented.
- Reset and race-start synchronization require continued validation.
- Teacher-to-live distillation tooling is not implemented.

## Promotion Criteria

Before live policy control:

- deterministic surrogate evaluation passes gate 0 at least 95%
- mean gates passed is at least 2
- collision rate is below 20%
- no persistent vertical runaway
- live thrust/rate signs and scaling are confirmed from time-series logs

Use bounded commands and shadow mode before unrestricted simulator flight.

## References

- `docs/AI_GP_SWIFT_EXECUTION_PLAN.md`
- `docs/SWIFT_ADOPTION_REPORT.md`
- `docs/AI_GP_RL_STRATEGY.md`
- `docs/AI_GP_CONTROL_CALIBRATION.md`
- `docs/AI_GP_RUNPOD.md`
