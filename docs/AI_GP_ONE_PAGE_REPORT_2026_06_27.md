# AI-GP One-Page Report - 2026-06-27

## Current State

The current best AI-GP structured-state pilot is:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

This is the first structured-state candidate that clears the Swift-level
surrogate threshold. Nominal evaluation is solved at `100%` success over 512
episodes, and randomized held-out evaluation is `98.8-99.4%` full-course
success across seeds `1001-1003`.

## What The Numbers Mean

The policy is not passing only `66%` of individual gates. It completes all six
gates in roughly `99%` of randomized episodes. Mean gates are about `5.99 / 6`.

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

Hybrid evaluation then found the real lever: keeping the `039` actor except
inside `10 m` of the active gate plane, where the geometric teacher takes over,
completed all three randomized validation seeds at `100%` with zero failures.
Run `040` behavior-cloned that hybrid target into a pure actor.

`040` randomized held-out validation:

- success: `69.60% -> 99.22%`
- mean gates: `5.057 -> 5.990`
- missed gates: `28.26% -> 0.78%`
- collisions: `2.15% -> 0.00%`

## Current Bottleneck

The immediate bottleneck has moved from surrogate learning to transfer
verification. CPU tests are useful for syntax, unit tests, and small
diagnostics. Real policy learning and validation belong on RunPod/GPU.

The controller must learn the active-gate task generally: given any active gate,
relative gate pose, vehicle state, previous action, and dynamics variation, pass
that gate quickly and set up the next gate. Narrow replay on one failing segment
can improve that segment while damaging other gates.

## Work In Progress

Run `040` is exported in:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

The next engineering step is AI-GP simulator integration/shadow testing with
time-series comparison against surrogate gate crossings. Do not move to
camera-only/live vision until this structured-state controller transfers.
