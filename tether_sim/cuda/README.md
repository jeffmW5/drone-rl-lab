# C host and CUDA tether prototype

Build in Ubuntu-Laya native repository:

```
bash scripts/build_tether_cuda.sh
```

Host API is C11, compiled with gcc warnings as errors. Physics kernels use
C-style CUDA extensions and nvcc for sm61. They contain no C++ classes/templates;
CUDA compilation still uses NVIDIA's compiler. Python is the PufferLib adapter,
GPU controller/learner and HTTP viewer host, not the physics integrator.

State stays in device buffers: drone, twelve cable particles, payload. Quaternion
rigid-body rotation, gyroscopic term, four first-order motors, wind, linear drag,
gravity, mass-weighted link constraints and point/sphere box detection. FP64
positions/velocities, FP32 motor/rotation state. Twenty1ms substeps/control step,
32 link iterations. One GPU thread/environment, batched128 for the learner and
1024 in the throughput probe. No CPU physics workers. PufferLib3 still transfers
actions through CPU; this is not an entirely device-resident rollout path.

Basic checks: scripts/check_tether_cuda.py. Swing/timestep diagnostic:
scripts/check_tether_cuda_swing.py. Frozen evidence in tether_sim/evidence/.
The first build failed on CUDA runtime linking, then on missing CUDA context;
both were repaired. Trainingv001 failed because nvcc wasn't on PATH. Directv002
is INVALID: old kernel8parameter layout used with new9parameter host after lease
rejected compilation. Preserved runtime artifacts; source hash guard now requires
successful rebuild. Never promote those invalid results.

Hybrid trainingv002:32768 transitions/10.2155s,128 environments, PufferLib PPO.
Direct controlv003 is a separate experiment: ground start, motors zero, four
policy outputs mapped directly to[0,1] throttle, no PD or residual corrections.
Ground supports approximate normal projection/friction; initial rope lies along
ground. Reward uses destination progress/error, angular velocity and crash cost.
First takeoff strip excludes nearby boxes; this is a learning diagnostic, not
the old randomized obstacle-navigation benchmark.

Live view: http://localhost:8767/ (runner serves during evaluation/training and
ten minutes of saved-policy playback). Cable rendering follows actual GPU node
positions. Viewer process holds GPU lease; do not compile or train concurrently.

Not equivalent to MuJoCo: no offset attachment torque, distributed link rotational
inertia, cable bending/contact response, capsule collisions or rotated boxes.
Approximate ground contact. Collision flags terminate episodes rather than solve
obstacle contact. No MuJoCo trajectory parity or hardware calibration established.
Do not compare easier5second destination errors to original90second sustained1mm
navigation results. A32k training run finishing10s is not a10min->10s matched speedup.
Training duration excludes initial/final evaluations and playback.
