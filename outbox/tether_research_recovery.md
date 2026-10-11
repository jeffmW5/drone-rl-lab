# Numerical failure and recovery diagnostic

2026-10-11 heartbeat: longer-training job had a dead Process-3 (PID19466),
MuJoCo warning `Nan, Inf or huge value in QACC at DOF6`, simulated time55.104s,
and FloatingPointError. Parent19414 kept waiting with stale progress237568.
Stopped only its verified process group; retained every checkpoint and log.
Last saved pair checkpoint_200704; no final evaluation. Marked invalid/failed,
no promotion. Scope: one stochastic rollout; exact seed/domain and physical
cause unavailable, so finer integration is a hypothesis rather than a diagnosis.
Evidence: tether_sim/evidence/research002_numerical_failure.json and runtime
tmp/tether-research/log_tether_research_002_more_steps.txt.

Added worker watchdog: unexpected worker exit raises in the parent, saves
interrupted artifacts and terminates its own children. Research runner stops
at failed trials rather than skip them. Optional numerical-failure capture
saves pre-step qpos/qvel, motor state, wind, residual action, domain and physics;
invalid transitions still raise, never silently reset or count as successes.
Twelve harness/adapter/watchdog checks passed, including real failed child,
failure serialization, 1ms physics /20ms control clock and duplicate lease.

Frozen recovery config configs/tether_research_recovery_v001.json resumes
checkpoint_200704 with integration1ms (previous2ms), 100000 transitions or600
training seconds plus3-seed validation. This changes the physics benchmark;
compare its initial/final results within this version only, no promotion
against the original2ms campaign. New failures capture their pre-step state.
Reserved audit seeds untouched. Recovery is the third research training launch
in the rolling24h window; defer the original long-horizon trial and new training
until the limit allows. Review recovery evidence before preparing campaignv002.

23:49 UTC review: recovery completed102400 transitions in441.800s including
final evaluation. No recorded numerical-failure snapshots or logged instability
in this shorter trial; this does not establish a fix for the237568-step failure.
Three validation seeds: zero crashes, zero1mm successes,3/3 routes complete.
Mean final payload destination error199.116mm initial ->194.025mm final (2.56%);
PD211.579mm. No cross-timestep promotion, original selected policy unchanged.
Evidence: tether_sim/evidence/recovery_v001_evaluations.json and
state/tether_research/recovery_v001.json. Further launches permitted only after
2026-10-12 22:48:33 UTC under the rolling24h allowance.

Single-seed PD interventions, seed51001,1ms model, final10s summaries:
randomized final error386.858mm; wind off379.417mm; wind off + matched motor
gains367.434mm. All routes complete, no crashes. Mean vertical error remained
about-368mm even with wind off and matched gains. Therefore wind/motor imbalance
alone does not explain this particular case. This is one diagnostic, not a
general causal conclusion; tether_sim/evidence/recovery_bias_probe.json.

Attitude-gain intervention on that same randomized case: default .015 final
error386.858mm, route complete, mean tilt12.44deg. Lower .006 gain reduced mean
tilt to1.83deg and mean vertical error from-0.368m to+0.051m, but route remained
1.335m incomplete and final error1425.945mm. Reject that gain change as a
controller improvement. Evidence: tether_sim/evidence/attitude_gain_probe.json.
Hypothesis (low confidence): coupled motor lag/attitude tracking and outer-loop
steady error contribute; decreasing gain alone trades height behavior for route
tracking. Next falsification: paired PD sweeps over intermediate gain/damping,
plus inspect saturation and directional errors on all three validation seeds.
Consider an opt-in anti-windup integral controller only after those diagnostics;
never silently change the frozen benchmark/controller or count this as learned
policy progress. Reproduction: scripts/probe_tether_precision_bias.py and
scripts/probe_tether_attitude_gain.py from the native checkout.
