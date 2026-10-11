# PufferLib migration -- 2026-10-11

Implemented the PufferLib 3 Python-compatible trainer for the unchanged MuJoCo
NavigationTaskV2 environment. Native version 5 remains a separate C environment
port; do not present this as native5 or seconds-scale training.

Four tests passed: physics/reward/contact equivalence, transferred Gaussian action
means/critic values, buffer/horizon resets, CPU/CUDA advantage kernel parity.
CUDA smoke completed 4096 transitions and four updates. Maximum policy parameter
change 0.0024704039096832275. Frozen normalizer SHA identical before/after.
One held-out seed61001: no crashes for PD/PPO, no successes for either; payload
RMSE PD .136622684m / PPO .136764740m. No broad improvement claim.
Evidence: tether_sim/evidence/puffer_cuda_smoke.json.

PufferLib3 sdist NO_OCEAN packaging bug repaired by initializing c_extension_paths.
Isolated WSL venv reuses the existing Torch runtime through a .pth file; current
NumPy/Gymnasium exceed upstream declared bounds, so compatibility is scoped to
these tests and actual smoke execution. Optional Ocean/cloud logging omitted.
CUDA12.6 nvcc/cudart/cccl redistributions SHA verified; extension built for sm61.
setup_tether_puffer.py reproduces installation with one build job.

Earlier attempts exposed missing trainer env label and detached VecNormalize save
attributes; repaired; failed artifacts retained. WSL stopped while training and a
compiler build were concurrent; exact cause not established. V007 final pair saved
at157696 steps; its final held-out evaluation was interrupted. CPU workers now fork
before CUDA initialization and share imported library memory. Build and training
performed separately. WSL was already on D:.

Main frozen config tether_puffer_cuda_v002 resumes V007 final policy, optimizer and
fixed normalization, eight workers, 4096 rollouts, 1024 minibatches, 200000 additional
steps / 1200 training seconds plus evaluation. Puffer reward clipping, terminal
horizon semantics and value clipping differ from SB3. Main launch and broader
held-out results are pending at this checkpoint.

Native app adds Train PufferLib and PufferLib lesson, monitors Puffer processes and
uses exported SB3 ZIP/normalizer pairs for playback. Teaching guide saved to Codex
outputs/PufferLib-Training-Guide.md.
Follow-up: CUDA smoke with final shutdown code completed 4096 steps in 26.343 seconds including final evaluation; no worker processes remained. v002 main launch hit upstream physical-core guard before learning. Frozen v003 uses eight environments across four workers and resumes the same V007 final pair. Checkpoints saved before workers are stopped; worker SIGTERM defaults restored before fork.
