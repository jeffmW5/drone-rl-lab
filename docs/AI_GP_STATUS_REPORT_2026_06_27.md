# AI-GP Pilot Status - 2026-06-27

## Short Answer

We are not "passing 66% of gates." The best current structured-state policy is
roughly `60-66%` full-course success under randomized surrogate evaluation,
depending on seed and eval settings. That means it finishes all six gates in
about that fraction of episodes. Its mean gate count is about `4.8-5.1 / 6`.

Nominal, non-randomized evaluation is already solved by `031`: `100%` success
over 512 episodes. The remaining problem is robustness and transfer, not basic
gate sequencing.

## Current Best Artifact

Use this as the current integration/shadow-test candidate, not as a finished
Swift-level pilot:

```text
results/ai_gp_031_randomized_full_course_ppo_120m/best_policy.pt
results/ai_gp_031_randomized_full_course_ppo_120m/ai_gp_structured_policy.json
```

Randomized validation for `031`:

- seed `1001`: `59.77%` success, `4.89` mean gates, `38.87%` missed gates
- seed `1002`: `61.33%` success, `4.95` mean gates, `36.91%` missed gates
- seed `1003`: `60.55%` success, `4.81` mean gates, `36.91%` missed gates

Run `034` added safer anchored behavior cloning toward a better geometric
teacher, but it did not clearly promote beyond `031`. Its logged evals hovered
near the starting policy instead of becoming a better policy.

## What Swift-Level Means Here

For this project, a Swift-level structured-state AI-GP controller should hit:

- at least `95%` randomized full-course success across several held-out seeds
- mean gates `5.8-6.0 / 6`
- near-zero collisions, out-of-bounds, and vertical-runaway failures
- stable margins through the gate plane, not last-centimeter saves
- good behavior from varied spawn poses, active gate indices, dynamics, latency,
  mass, drag, camera FOV, and slight gate-pose perturbations

After that, vision or live-sim integration should be treated as a transfer
problem. A camera-only/live policy should not be expected to outperform the
structured-state controller until the structured controller is robust.

## Current Bottleneck

The bottleneck is missed gates under randomization. More thrust authority is not
the issue. CPU training is not the route to solve it either; CPU is useful for
unit tests, smoke tests, and short deterministic diagnostics. Real training
belongs on the GPU/RunPod path.

The mild tuned geometric teacher is a useful benchmark: it reached about `69%`
randomized full-course success with zero collisions in GPU evaluation. That
shows a better controller exists in this environment, but direct behavior
cloning from `031` destabilized the neural actor unless heavily anchored, and
heavy anchoring prevented clear improvement.

## Work In Progress

Added `scripts/extract_ai_gp_hard_cases.py` to turn time-series eval failures
into gate-frame diagnostic samples. It reports pre-failure plane, lateral,
vertical, and velocity offsets by active gate.

Running it on the existing `031` randomized evals found:

- tracked trajectories: `24`
- failed tracked trajectories: `10`
- extracted pre-failure samples: `1010`
- failure type: all `missed_gate`
- active-gate sample concentration: gate `4` and gate `5` are highest, with gate
  `0` still present

The existing telemetry only tracks a small subset of episodes, so the next GPU
diagnostic should record many more trajectories before the next training run.

## Next Engineering Steps

1. Run a dense GPU evaluation of `031` with `128-512` tracked trajectories per
   seed.
2. Extract hard cases and convert them into targeted reset distributions or
   DAgger samples.
3. Improve the teacher/controller on those states first, then train the neural
   actor with anchoring that preserves the solved nominal course.
4. Promote only if multi-seed randomized validation reaches the Swift-level
   thresholds above.
5. Prepare AI-GP sim usage as shadow/integration testing until the policy clears
   those thresholds.
