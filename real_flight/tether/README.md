# AI-deck tether perception pipeline

Source-controlled home for the pipeline brought up on 2026-10-08. This is
observation-only software; the four labels are absent/left/center/right in the
camera image. It does not estimate tension or implement lift control.

## Current evidence

- WSL Ubuntu-Laya and GTX 1070 Ti: CUDA training verified in the separate
  `/home/n33du/ai/crazyflie-tether-venv` environment.
- Integer ONNX, Python and scalar host C agree on 512 synthetic validation images.
- DORY GAP8_L2 generated kernels agree byte for byte with a scalar oracle on
  16 simulator inputs, 112 layer executions. Camera image generation passes.
- STM32 observation receiver builds for cf2 and cf21bl. No device was flashed
  or flown in this bring-up.

Snapshot manifests in `evidence/` retain original paths and artifact hashes.
These are dated evidence from the prior workspace, not live build results from
this checkout. Re-run verification after changes.

The repo already records real DroNet hardware runs and camera timing in
`../GAP8_DORY_RESULT.md` and `../GAP8_PERF_RESULT.md`. Those older measurements
are separate from this model's simulation results. The hardware inventory in
`../STATUS.md` identifies a Crazyflie 2.1, Flow Deck v2, Crazyradio PA and
Olimex ARM-USB-TINY-H; current device connection and AI-deck revision still
require verification.

## Work from the native WSL checkout

```bash
cd /home/n33du/ai/drone-rl-lab/real_flight/tether
source /home/n33du/ai/crazyflie-tether-venv/bin/activate
python train.py --synthetic --epochs 5
python export.py runs/smoke/model.pt
bash generate-tether.sh
python prepare_suite.py
```

Vendor trees, generated files, checkpoints and recordings stay ignored. The
existing verified workspace is `/home/n33du/ai/crazyflie-tether`; local ignored
directories can be linked from it using `link-existing-workspace.sh`. This
preserves the artifacts while running the source from this repo. It refuses
to overwrite existing paths.

```powershell
wsl -d Ubuntu-Laya -u root -- /home/n33du/ai/crazyflie-tether-venv/bin/python /home/n33du/ai/drone-rl-lab/real_flight/tether/sdk_build.py golden
wsl -d Ubuntu-Laya -u root -- /home/n33du/ai/crazyflie-tether-venv/bin/python /home/n33du/ai/drone-rl-lab/real_flight/tether/sdk_build.py camera
```

The build wrapper pins the SDK digest and exposes no USB devices. Use explicit
image/run targets; the SDK's `make all` also invokes a flash recipe.
Recheck the physical chip revision and recovery image before any hardware step.

## Real data and capture

Use the existing native Windows testbed in `../windows_testbed` for capture:
the repo records VirtualBox NAT stream stalls, so do not move dataset capture
into Omarchy. `capture.py` is a bounded protocol receiver, tested with fixtures;
its live deck connection has not been validated during this bring-up.

Train real data using `python train.py --data data --output runs/real` with
`data/train/{absent,left,center,right}` and matching `data/val` directories.
Split recording sessions, not adjacent frames. Update the checkpoint and data
arguments explicitly before integer export; `generate-tether.sh` defaults to
the synthetic smoke checkpoint.

Omarchy is the VirtualBox VM at `D:\VirtualBox VMs\Omarchy`, currently awaiting
user unlock. Keep CUDA training in WSL. See [PROTOCOL.md](PROTOCOL.md) and
[FLIGHT-READINESS.md](FLIGHT-READINESS.md) for integration limits and remaining
hardware validation.
