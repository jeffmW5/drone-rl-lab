# AI-GP One-Page Report - 2026-06-27

## Current State

The current best AI-GP structured-state pilot is:

```text
results/ai_gp_039_all_gate_soft_floor_ppo_30m/best_policy.pt
results/ai_gp_039_all_gate_soft_floor_ppo_30m/ai_gp_structured_policy.json
```

This is a simulator/shadow-test candidate, not a finished Swift-level pilot.
Nominal evaluation is solved at `100%` success over 512 episodes, but randomized
evaluation is only about `67-71%` full-course success across held-out seeds.

## What The Numbers Mean

The policy is not passing only `66%` of individual gates. It completes all six
gates in roughly `69%` of randomized episodes. Mean gates are about `5 / 6`,
which means most failures happen late enough to pass several gates first, but
the controller is still not robust enough for a Swift-level claim.

Swift-level for this project means at least `95%` randomized full-course
success, `5.8-6.0 / 6` mean gates, near-zero collisions, and stable gate-plane
margins under randomized spawn pose, active gate, dynamics, latency, mass, drag,
camera FOV, and slight gate-pose noise.

## Latest Training

Run `038` trained on RunPod from `036` with a soft-floor altitude penalty to
reduce final-gate ground contacts. It helped collisions but did not clearly
promote:

- success: `68.75% -> 69.47%`
- mean gates: `5.05 -> 4.89`
- missed gates: `29.04% -> 29.17%`
- collisions: `2.28% -> 1.37%`

The failure distribution moved from late/final-gate collisions into earlier
gate-1 and gate-3 misses. That is useful evidence, but not a reliable
improvement.

Run `039` then started from `036`, kept the soft-floor penalty, and trained
randomized near-gate starts across all six active gate indices. It is the new
candidate, but the gain is small:

- success: `68.75% -> 69.60%`
- mean gates: `5.053 -> 5.057`
- missed gates: `29.04% -> 28.26%`
- collisions: `2.28% -> 2.15%`

## Current Bottleneck

The bottleneck is randomized robustness and transfer, not raw thrust authority
or CPU training time. CPU tests are useful for syntax, unit tests, and small
diagnostics. Real policy learning belongs on RunPod/GPU.

The controller must learn the active-gate task generally: given any active gate,
relative gate pose, vehicle state, previous action, and dynamics variation, pass
that gate quickly and set up the next gate. Narrow replay on one failing segment
can improve that segment while damaging other gates.

## Work In Progress

Run `039` is exported in:

```text
results/ai_gp_039_all_gate_soft_floor_ppo_30m/ai_gp_structured_policy.json
```

The next learning step should improve the teacher/controller target for hard
randomized states. More narrow replay on one gate is unlikely to produce the
`95%+` randomized success needed for Swift-level behavior.
