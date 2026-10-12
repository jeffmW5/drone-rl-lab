# CUDA C simulator and learned takeoff experiments

C11 host compiled with gcc; C-style CUDA kernels compiled for sm61. Python
PufferLib adapter and live renderer, no CPU physics workers. GPU uses FP64 for
particle positions/velocities. Twelve cable particles and a payload with mutual
mass-weighted link constraints; rigid-body drone, quaternion attitude and motors.
This is a reduced model, not MuJoCo/hardware equivalence. See cuda/README.md.

Basic model checks passed.1s hover drift0.216mm, maximum link stretch0.0205mm;
100ms free fall0.0495405m matches semi-implicit analytical integration with drag
disabled. These are isolated unit scenarios, not navigation accuracy. Box collision
detection checked; new ground mode supports approximate normal projection/friction.
Basic checks reproduced after ground-mode addition (default ground mode off).

Small-angle near-fixed cable support period1.52923s versus massless reference
1.55389s; distributed cable has1g mass and payload5g. Halving timestep changes
period0.0000793s; stretch4.97um->1.24um. This does not prove complex contact or
coupled attachment torque fidelity. Full trajectories retained in evidence.

1024env physics-only throughput24,073/s initially and23,033/s on later check.
Includes synchronized20x1ms integration,32 iterations; excludes PPO/rendering.
128env hybrid PufferLib training32768 transitions/10.215s (~3208/s). The simpler
5second destination task error55.055->54.200mm on eight paired domains,0 crashes.
One route worsens. No comparison with original90second navigation leaderboard.

User requested pure learned control from takeoff. Direct control has zero motors
initially and drone on ground, cable lying on ground, policy maps four actions
directly to motor throttles. No PD invocation: bypass check replaces baseline
with a function that raises and confirms direct stepping still works. Takeoff
strip clear of boxes; full obstacle curriculum not yet established.

Directv003 completed131072 transitions,0/8 deterministic takeoffs (>0.3m drone
height), error1494.251->1497.226mm (worse). No crash is not success: remained on
ground. Full counterevidence retained. Directv004 recovery starts equal network
motor biases at0.61 throttle, logstd=-4; this is an initialization prior, not
feedback PD. Motors still start zero. Bounded262144 transitions/120training seconds.
No success claim until measured. Selected original MuJoCo policy unchanged.

Failed startupv001 lacked nvcc PATH. Directv002 INVALID old8parameter kernel/new
9parameter host ABI after compile lease rejection; retained artifact. Source-hash
guard now rejects stale CUDA builds. Frozen versioned folders preserve outcomes.
Completed playbacks were stopped to release lease before new builds/training;
healthy training was not interrupted. One GPU job at a time.

Viewer http://localhost:8767/ serves real GPU node positions and controller mode;
training followed by10minute deterministic playback. HTML served/state endpoint
verified; screenshot inspection unavailable due computer-use tool initialization
error. User opened live page in in-app browser.
