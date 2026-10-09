# Tether Lab / Flight Studio

An executable starting point for Crazyflie **lift and carry** research, with
MuJoCo physics, a live 3D browser viewer, and a Gymnasium environment.
This is not yet a validated state-of-the-art simulator.

First CUDA residual-PPO training: [training setup and limits](TRAINING.md).

## Run on this machine

```powershell
wsl -d Ubuntu-Laya --cd D:\drone-rl-lab -- /home/n33du/ai/tether-sim-venv/bin/python -m tether_sim.viewer
```

Open http://localhost:8766. Orbit by dragging; scroll to zoom. Follow tracks the
drone, Inspect CAD pauses and moves close, Onboard shows the downward camera.
Restart starts the selected scenario. Wind pulse applies 0.025 N horizontally
for one simulated second. Episodes repeat after 20 seconds in the viewer.
Physics targets wall-clock time; FPS and simulation backlog are measured live.
The verified local 3-second sample averaged 25.06 FPS (24.69–25.88), with
less than 20 ms backlog at 960x540. See `evidence/viewer.json`. The camera
updates every third rendered frame. This is not a sustained-load benchmark.
The viewer must remain running for the page to work. It binds localhost only.

For a fresh Linux environment:

```bash
python3 -m venv /path/to/tether-sim-venv
/path/to/tether-sim-venv/bin/pip install -r tether_sim/requirements.txt
/path/to/tether-sim-venv/bin/python -m tether_sim.viewer
```

OpenGL/WSLg is needed for rendering. Headless deployments may use MuJoCo's
EGL or OSMesa backend if available; those backends were not verified here.
The dedicated venv avoids changing the existing Crazyflie training stack.

## Implemented physics and contracts

- Free 6-DOF drone, four thrust forces at motor sites and alternating yaw
  moments, first-order motor lag, linear translational drag, injected wind.
- Default recorded drone mass 43.4 g, assumed 5 g payload and 1 g cable.
- Cable: twelve 5 cm rigid capsule links, ball joints, small damping and
  numerical armature, ground/contact/self-collision, sphere payload.
- 2 ms integration and 20 ms control interval. The attachment is 2 cm below
  the drone center, so maximum center-to-payload span is 62 cm, not 60 cm.
- Action: four motor commands in [0,1], mapped to [0,0.20] N each. Out-of-range
  values are clipped; malformed/nonfinite actions raise an error.
- Observation: qpos, qvel, four actual motor thrusts, target xyz, wind force
  xyz, simulation time. This is privileged simulation state, not sensor data.
- Reward: negative drone target distance minus 0.01 sum of squared commands.
  It is a baseline reward, not yet a tuned payload-task objective.
- Baseline: privileged position/attitude PD with motor allocation. Lift/carry
  rises to 1.2 m, then shifts the target 0.6 m between t=3 and t=7 seconds.
- Camera: synthetic downward pinhole grayscale, 324x244 or 64x64. Camera
  calibration, distortion, exposure, noise and real AI-deck latency are absent.
- Attachment axial load comes from the first link's internal wrench projected
  onto its local cable axis. This includes link/payload dynamics, not just mg.
- Numerical solver warnings invalidate the rollout with an exception.

Official MIT-licensed Bitcraze meshes provide the visual assembly. See
[asset provenance](assets/bitcraze/PROVENANCE.md). Visuals have zero mass and
no contacts. AI/Flow deck boxes are illustrative. Propeller animation is
cosmetic; it does not represent identified RPM. These meshes are not
manufacturing CAD, and the rendering is not photorealistic.

## Verification and vision demonstrations

```bash
python -m pytest tether_sim -q
python -m tether_sim.verify_viewer  # while the viewer is running
python -m tether_sim.evaluate --output tmp/tether-sim/evaluation.json \
  --dataset tmp/tether-sim/lift-carry-demo.npz
```

Twelve checks cover Gymnasium contracts, zero-thrust falling, mass budget,
five baseline scenarios, rigid length bounds, deterministic replay, timestep
convergence, static attachment weight, and ground contact/folding. The ground
fold test starts with a small tilt; a perfectly symmetric chain can stand in
compression. Tests are model checks, not proof of real-world fidelity.
Gymnasium warns about intentionally unbounded privileged observations.

`evidence/baseline.json` records 12-second deterministic baseline runs.
The optional NPZ contains 120 64x64 uint8 grayscale images, privileged states,
and motor actions from the combined task (10 Hz sampling). These are synthetic
demonstrations, not ground-truth real-flight training data. No vision policy
has been trained or deployed by this package.

## Limits and development gates

1. **Cable fidelity:** rigid links can transmit compression and have numerical
   joint armature; this is an approximation to rope. Compare against a
   tension-only model and a discrete elastic rod/XPBD formulation. Validate
   pendulum period, sag, slack-to-taut impulses, contact friction and energy
   dissipation against measurements; test segment/timestep convergence.
2. **Hardware identification:** measure mass/inertia, motor thrust curves,
   saturation, lag, cable diameter/mass/stiffness, attachment geometry and
   payload. Current values except recorded drone mass are assumptions.
3. **Task robustness:** add seeded parameter/initial-state randomization,
   varying payloads, gust spectra, pickup/placement, obstacles and success
   metrics for payload accuracy, swing, peak load and safety margins. Run
   held-out seeds and stress sweeps before comparing policies.
4. **Vision:** import calibrated AI-deck camera geometry and square markers,
   real texture/lighting distributions, motion blur and measured latency.
   Evaluate perception separately before closed-loop vision policies.
5. **Training:** connect batched rollouts to the existing GTX 1070 Ti policy
   training stack, benchmark throughput/memory and recurrent policies.
   Current physics is CPU MuJoCo; GPU physics/training is not implemented here.
6. **Presentation:** add higher-detail deck CAD, scene assets, capture/export
   and optional native GPU rendering. Benchmark sustained frame rate before
   claiming 30/60 FPS. MJPEG currently supplies the live browser video.

References: [MuJoCo modeling](https://mujoco.readthedocs.io/en/stable/modeling.html),
[Python/rendering](https://mujoco.readthedocs.io/en/stable/python.html),
[official Bitcraze meshes](https://github.com/bitcraze/crazyflie-simulation).
