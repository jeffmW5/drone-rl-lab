# AI-GP RunPod Workflow

The AI-GP trainer does not need `lsy_drone_racing`, JAX, Pixi, or a remote Git
checkout. The Windows launcher uploads only the files required by the selected
AI-GP experiment.

## One-Time Windows Setup

Set these in the PowerShell session or your user environment:

```powershell
$env:RUNPOD_API_KEY = "..."
$env:RUNPOD_POD_ID = "..."
$env:DRONE_RL_DEPLOY_KEY = "$HOME\.ssh\id_ed25519_runpod"
```

The target pod should use a RunPod PyTorch image with SSH enabled.

## Commands

```powershell
# Resume pod, discover SSH, upload, verify CUDA, and start training.
.\scripts\runpod_ai_gp.ps1 Train `
  -Config configs/ai_gp_002_swift_teacher_gpu_ppo.yaml

# Same operation with automatic pod stop after the trainer exits.
.\scripts\runpod_ai_gp.ps1 Train -AutoStopPod

# Inspect or follow the run.
.\scripts\runpod_ai_gp.ps1 Status -Experiment ai_gp_002_swift_teacher_gpu_ppo
.\scripts\runpod_ai_gp.ps1 Logs -Experiment ai_gp_002_swift_teacher_gpu_ppo

# Retrieve results without Git commits or pushes.
.\scripts\runpod_ai_gp.ps1 Pull -Experiment ai_gp_002_swift_teacher_gpu_ppo

# Stop only the training process or stop the whole pod.
.\scripts\runpod_ai_gp.ps1 StopTraining -Experiment ai_gp_002_swift_teacher_gpu_ppo
.\scripts\runpod_ai_gp.ps1 StopPod
```

For a manually started pod, bypass API discovery:

```powershell
.\scripts\runpod_ai_gp.ps1 Train -HostName 1.2.3.4 -Port 22022
```

## What `Train` Does

1. Resumes the configured pod unless an SSH endpoint was supplied.
2. Waits for SSH.
3. Creates a compressed bundle containing:
   - `ai_gp_rl/`
   - `train.py`
   - `train_ai_gp.py`
   - the selected YAML config
   - the small remote job helper
4. Uploads and extracts it into `/root/drone-rl-lab`.
5. Installs PyYAML only when missing.
6. Verifies PyTorch CUDA access.
7. Runs a 256-environment CUDA step and backward-pass smoke test.
8. Launches training under `nohup`.
9. Writes logs and PID/exit state under `/root/drone-rl-lab/logs`.

Private SSH keys are not copied to the pod.

## Outputs

Remote:

```text
/root/drone-rl-lab/results/<experiment>/
/root/drone-rl-lab/logs/<experiment>.log
/root/drone-rl-lab/logs/<experiment>.pid
/root/drone-rl-lab/logs/<experiment>.exit
```

Local `Pull` destination:

```text
results/<experiment>/
```

## Current Limitation

The workflow was exercised again on an RTX 3090 on June 13, 2026. The required
teacher CUDA smoke produced `(256, 45)` observations, and the temporal student
smoke produced `(256, 94)` observations. Checkpoint creation, result pull, and
deterministic telemetry evaluation all worked.

The generic `manage_pod.sh` replacement-pod path was also fixed so progress
logging cannot corrupt a captured and persisted pod ID.

RunPod success does not promote a policy to Windows control. Only an exported
live-contract student that passes Windows simulator shadow and bounded
evaluation can be marked command-eligible.
