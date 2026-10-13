# Independent CPU physics/control audit, 2026-10-13

Scope: unchanged production physics5/6 and fixed checkpoints. No GPU calls/context, no GPU compilation, no trainer pause/restart and no production source edits. The C++ translation changes only the CUDA qualifier and environment-index expression; runs environment0 with original arithmetic. CPU/CUDA numerical differences are measured, so this is not a new benchmark backend or hardware calibration. Files: audit.py, derivatives.py, result.json, policy_derivatives.json, physics_v005_cpu.cpp, physics_v006_cpu.cpp. Native path /home/n33du/ai/drone-rl-lab/tmp/deep_audit_physics.

## Invariants and controller authority

- Physics5 lines17-20,24-32; cuda_task.py263-265: motor mixer matches physical torque allocation. Multiplication gives diagonal torque/action [0.000512,0.000512,0.000096]Nm. Positive/negative roll/pitch/yaw pulses passed bothphysics versions; positive roll tilts thrust toward negativeY and positive pitch toward positiveX. Quaternion updated-axis timing is consistent.
- cuda_backend.py30-31: total mass0.0494kg, motor max0.2N, roll/pitch inertia2.5e-5kgm². Equilibrium command0.6057675approximately. Maximum differential roll/pitch action gives20.48rad/s² after motor settling, so insufficient physical torque authority is not established.
- At20ms from balanced motors, action1roll gives0.00103555rad attitude and0.133258rad/s rate; yaw gives0.000121353rad/0.0156162rad/s. Motor25ms lag matters: these are smaller than instantaneous-force predictions. Sign reversals mirror.
- Taut-hover balanced motors5s at1ms/0.5ms and zero-thrust corrected ground1s were finite and nonfailed for5/6. No artificial unpowered carrier launch above0.051m in this test. Evidence in physical_checks; this does not calibrate tether/contact/material fidelity.
- Physics5 rope attaches at the carrier center and applies position corrections only (lines39-46); no attitude/tether attachment torque. The rotational dynamics are therefore mechanically decoupled from rope constraint/contact torque. Material/shortened contour, sphericalcarrier floor and floorprojection ordering remain approximations, not new discoveries of a numerical defect.

## Fixed-policy actual-flight reproductions and perturbations

Five unchanged ground replays reproduce previously saved GPU first-boundary times exactly at20ms resolution:
7585.64s;7595.88s;7618.56s;7635.20s;7135.64s. Shared-frame maximumcarrierposition discrepancy0.582,0.304,4.177,0.231,0.530mm respectively. Scope is comparison against trial771's saved trajectories, not bitwise-equivalent CPU/CUDA. No hover from ground.

Counterfactual start tests retain the full14-node tether and production physics5:

|Selected checkpoint|Ideal hover initial hold(s)|Ideal hover firstescape(s)|+10degree roll firstescape(s)|Initial uniform vx+.5m/s firstescape(s)|Ground physics6 firstescape(s)|
|---|---:|---:|---:|---:|---:|
|758|1.62|13.76|1.72|5.04|5.44|
|759 obs2|1.70|5.50|1.72|4.98|5.82|
|761 obs2|1.68|4.80|1.74|7.70|5.20|
|763 obs2|1.74|4.54|1.74|5.70|4.74|
|713 obs2|1.66|5.38|1.72|5.22|5.48|

The ideal-hover hold comes from placing the drone at0.8m, tether hanging, zero velocities, equilibrium motor forces. It is not learned takeoff-and-hover success, and every tested policy eventually escaped. The tilted tests hold only0.1s, reflecting a small initialspeed interval before accelerating laterally. The physics6 counterfactual did not rescue these policies. It is inappropriate to count this as promotion or use training failure to claim wrong physics.

## Learned local attitude response

Autograd at a synthetic exact-hover state uses loaded checkpoint source and architecture metadata (obs1/2); no learned weight changes. Selected759:

- Mean roll action -0.00053018, pitch -0.00026728; both nonzero at ideal equilibrium. These steady biases accelerate angular velocity.
- d(rollaction)/d(rollangle)=-9.9101e-5 perrad; d(rollaction)/d(wx)=-1.3564e-4 perrad/s. Correct restoring/damping signs, extremely small authority in actual action units.
- With20.48rad/s² peraction, purelocalroll damping coefficient≈0.00278/s (360s timescale). Position/velocity crosscoupling also exists, so this is an isolated derivative interpretation, not the full recovery time.
- d(pitchaction)/d(wy)=+1.0571e-5 at this anchor, weak anti-damping. Other checkpoints sometimes negative; no global reversedsign or universal rate-sign claim.
- UniformX displacement affects rollaction0.0022911/m versus uniformY0.0016202/m. Crossaxis coupling can exceed same-axis coupling. A neuraloutput with locally correct ownaxis signs can still be dynamicallyunstable.

A reduced rigid-lateral analytical model (globalX/Y translation, commonvelocity, roll/pitch/rates and two motor torque states) incorporates these realpolicyderivatives, fixedphysical inertia, thrust and25msmotorlag. Maximumrealpart of eigenvalues758/759/761/763/713 is0.46994/0.45889/0.48376/0.49403/0.46994 persecond. This supports an unstable localresponse hypothesis, not a rigorous fullcable/contact linearization: internal rope modes omitted and the anchor has nonzero policybias, not an exact fixedpoint. Actual perturbed flight failures are stronger directcounterevidence to robuststability.

## Recommendation

Priority: directly test restoring attitude/rate response under the existing serial GPU lease before changing another reward coefficient or physics version. Replay selected checkpoints from fulltether exacthover with ±5/±10degree roll/pitch, ±0.1/±0.5rad/s rates and ±0.25/±0.5m/s lateral velocities; add unchangedground control and fixedmotorpulse authority controls. Capture pre-reset states, physicalmotor torques, actualtheta/rate changes, displacement/speed and 1s hold until first truefailure. CPUaudit predicts angle/rate recovery fails despite validactuator sign/authority. Reject the hypothesis if CUDA reliably recovers acrossbothaxes/signs whileCPUcounterfactuals disagree.

If confirmed, target learned innerattitude dynamics/credit rather than uniformoutputamplification: prior64x roll/pitch gain trials alreadyfailed and are explicit counterevidence. A versioned curriculum with perturbations/independent axes and measured recovery could be justified, but is a newexperiment and not tested here. No PID/helper, hiddenangularratepenalty or sourcepatch added. No originalstrict1mm, navigation or hardwareaccuracy claim.
