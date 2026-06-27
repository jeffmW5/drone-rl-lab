# AI-GP One-Page Report - 2026-06-27

## Current State

The current best AI-GP structured-state pilot remains:

```text
results/ai_gp_036_weighted_final_approach_ppo_20m/best_policy.pt
results/ai_gp_036_weighted_final_approach_ppo_20m/ai_gp_structured_policy.json
```

This is a simulator/shadow-test candidate, not a finished Swift-level pilot.
Nominal evaluation is solved at `100%` success over 512 episodes, but randomized
evaluation is only about `66-71%` full-course success across held-out seeds.

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

## Current Bottleneck

The bottleneck is randomized robustness and transfer, not raw thrust authority
or CPU training time. CPU tests are useful for syntax, unit tests, and small
diagnostics. Real policy learning belongs on RunPod/GPU.

The controller must learn the active-gate task generally: given any active gate,
relative gate pose, vehicle state, previous action, and dynamics variation, pass
that gate quickly and set up the next gate. Narrow replay on one failing segment
can improve that segment while damaging other gates.

## Work In Progress

Run `039` is prepared in:

```text
configs/ai_gp_039_all_gate_soft_floor_ppo_30m.yaml
```

It starts from `036`, keeps the useful soft-floor penalty from `038`, and trains
randomized near-gate starts across all six active gate indices. The goal is
general active-gate competence, not a hand-tuned gate-5 recovery behavior.

Promotion requires multi-seed randomized evaluation beating `036` on full-course
success and mean gates without increasing missed gates or collisions.
