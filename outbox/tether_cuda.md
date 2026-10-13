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
height), error1493.873->1497.226mm (worse). No crash is not success: remained on
ground. Full counterevidence retained. Directv004 recovery starts equal network
motor biases at0.61 throttle, logstd=-4; this is an initialization prior, not
feedback PD. Motors still start zero. Bounded262144 transitions/120training seconds.
No success claim until measured. Selected original MuJoCo policy unchanged.

Directv004 complete262144 transitions/81.473s (~3218/s),128 parallel environments.
Initialized policy8/8 takeoffs, mean error981.460mm; trained policy0/8 takeoffs,
mean error1576.165mm. Peak trained drone height only0.1397m. This is a rejected
learning result: near-hover initialization enabled lift, then PPO regressed.
Crashes0 does not imply task success. Viewer deliberately plays this final failed
policy; it is not substituted with the better initialization. Three-seed original
leaderboard and reserved audit seeds untouched. Prototype evaluation uses seed51011
to construct eight domains, not eight independent seed replications.

Hypotheses, not facts: reward may encourage low-throttle ground behavior; aggressive
updates with narrow exploration may destabilize learned motor balance. Next bounded
trial should add explicit takeoff/hover objectives and smaller updates, with initial,
periodic and final deterministic checkpoints selected by takeoffs/crashes/error.
Diagnose ground/tether contact fidelity before long navigation training. No promotion
of either direct-trained checkpoint. Original hybrid remains available as a reference.

Failed startupv001 lacked nvcc PATH. Directv002 INVALID old8parameter kernel/new
9parameter host ABI after compile lease rejection; retained artifact. Source-hash
guard now rejects stale CUDA builds. Frozen versioned folders preserve outcomes.
Completed playbacks were stopped to release lease before new builds/training;
healthy training was not interrupted. One GPU job at a time.

Viewer http://localhost:8767/ serves real GPU node positions and controller mode;
training followed by10minute deterministic playback. HTML served/state endpoint
verified; screenshot inspection unavailable due computer-use tool initialization
error. User opened live page in in-app browser.

## 2026-10-13 completed tenfold campaign and next reward test

Review0112 closes the completed six long runs885–890 and catches up the usage-limit review gap through936, using a dated931–936 addendum. All six completed655360 transitions each. None of240 periodic checks or six finals achieved sustained1s ground zone hover; maximum temporary hold0.54s. Fine reward16 improved five-second escape-free checkpoint counts in each training seed (11/26/36 versus6/10/7;73/120 versus23/120). This is a provisional correlated checkpoint signal, not independent validation or reliable hover. Final890 avoids escape but is still moving about1.25m/s; low altitude endpoints are not millimeter precision. Regressions, true failures and missing default terminal traces remain preserved.

One falsifiable reward17 adaptation is frozen: score horizontal velocity against an inward reference proportional to center displacement, capped at0.5m/s, tending to zero at the center. It changes reward only; no action helper, physics, reset, exploration or credit-horizon change. Three655360-transition treatments at70601–70603 reuse immutable long reward16 controls886/888/890, fixed-source759 and identical learner/simulator/action settings. Ten CPU groups passed, including old14/15/16 parity, direction/sign/cap/boundaries, unchanged motor actions and reset/failure logic; existing fine-reward checks passed12 groups. Leased new-run physics checks and actual initial/source comparisons are separately recorded; full learned outcomes remain pending.

Actual frozen trial937 is velocity_reference_reward17_seed70601. Daemon554137 remains active/unpaused with one trainer; no restart, healthy interruption, duplicate GPU work or concurrent compilation occurred. Its leased physics checks passed; backend/physics/host hashes and all10 initial policy tensors match the immutable886 control. Actual initial physical metrics remain pending in the final result, not inferred from weights alone. Sources, versioned review and queue are in state/cuda_research/methodology/review_0112 and configs/cuda_research/methodology_queue_v000_user_abca_00_velocity_reference.json. Fresh official HF retrieval and original-method fallbacks motivate the hypothesis but are not simulator results. Hardware/audit seeds remain reserved; original strict1mm obstacle navigation and hardware accuracy remain unachieved.

