# AI-GP Pilot Status - 2026-06-27

## Short Answer

We are not "passing 66% of gates." The best current structured-state policy is
roughly `66-71%` full-course success under randomized surrogate evaluation,
depending on seed. That means it finishes all six gates in about that fraction
of episodes. Its mean gate count is about `4.9-5.1 / 6`.

Nominal, non-randomized evaluation is already solved by `036`: `100%` success
over 512 episodes. The remaining problem is robustness and transfer, not basic
gate sequencing.

## Current Best Artifact

Use this as the current integration/shadow-test candidate, not as a finished
Swift-level pilot:

```text
results/ai_gp_036_weighted_final_approach_ppo_20m/best_policy.pt
results/ai_gp_036_weighted_final_approach_ppo_20m/ai_gp_structured_policy.json
```

`036` is now the current best structured-state candidate. `035` and `031`
remain previous baselines:

```text
results/ai_gp_035_hard_case_final_approach_ppo_20m/best_policy.pt
results/ai_gp_035_hard_case_final_approach_ppo_20m/ai_gp_structured_policy.json
results/ai_gp_031_randomized_full_course_ppo_120m/best_policy.pt
results/ai_gp_031_randomized_full_course_ppo_120m/ai_gp_structured_policy.json
```

Randomized validation for `036`:

- seed `1001`: `71.29%` success, `5.11` mean gates, `26.37%` missed gates
- seed `1002`: `68.55%` success, `5.08` mean gates, `28.91%` missed gates
- seed `1003`: `66.41%` success, `4.96` mean gates, `31.84%` missed gates

Previous randomized validation for `035`:

- seed `1001`: `69.34%` success, `5.05` mean gates, `28.13%` missed gates
- seed `1002`: `65.82%` success, `5.00` mean gates, `32.42%` missed gates
- seed `1003`: `66.41%` success, `4.88` mean gates, `32.23%` missed gates

Previous randomized validation for `031`:

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

That dense diagnostic has now been run for seed `1001` with `512` episodes and
`128` tracked trajectories. It reproduced the same aggregate eval result:
`59.77%` success, `4.89` mean gates, `38.87%` missed gates. The denser tracked
subset contained `47 / 128` failed trajectories:

- missed gate: `46`
- collision: `1`
- final active-gate failures: gate `5` = `17`, gate `4` = `11`, gate `1` = `8`,
  gates `2` and `3` = `4` each, gate `0` = `3`

The final `0.25 s` pre-failure window shows a centering problem, not an
authority problem. Gates `0-1` are mostly low at the plane, while later gates
are high or mixed with larger lateral misses.

## Current Training Result

`configs/ai_gp_035_hard_case_final_approach_ppo_20m.yaml` completed on RunPod.
It started from `031`, saved the initial actor as `best_policy.pt` before PPO
updates, used a stronger actor anchor, and focused near-gate replay on the last
`0.5-10 m` before the gate plane with wider vertical/lateral offsets derived
from the dense hard cases.

Embedded `035` best eval:

- success: `69.14%`
- mean gates: `5.15 / 6`
- missed gate: `27.15%`
- collision: `3.71%`

Held-out comparison against the same three randomized seeds used for `031`:

- average success improved from `60.55%` to `67.19%`
- average mean gates improved from `4.89` to `4.98`
- average missed-gate rate improved from `37.57%` to `30.92%`

Nominal `035` validation over 512 episodes remains clean: `100%` success, `6.0`
mean gates, zero failures, and `0.725 m` minimum crossing margin. The structured
policy export was written to:

```text
results/ai_gp_035_hard_case_final_approach_ppo_20m/ai_gp_structured_policy.json
```

Dense `035` telemetry then showed remaining final failures across gates 1-5:
gate `1` = `10`, gate `5` = `7`, gates `2-4` = `6` each in the tracked failed
set. `ai_gp_036_weighted_final_approach_ppo_20m` added weighted replay for those
active gates and completed on RunPod.

Embedded `036` best eval:

- success: `73.83%`
- mean gates: `5.23 / 6`
- missed gate: `23.05%`
- collision: `3.13%`

Held-out comparison against the same three randomized seeds:

- average success improved from `67.19%` to `68.75%`
- average mean gates improved from `4.98` to `5.05`
- average missed-gate rate improved from `30.92%` to `29.04%`
- average collision rate increased from `1.89%` to `2.28%`

Nominal `036` validation over 512 episodes remains clean: `100%` success, `6.0`
mean gates, zero failures, and `0.702 m` minimum crossing margin. The structured
policy export was written to:

```text
results/ai_gp_036_weighted_final_approach_ppo_20m/ai_gp_structured_policy.json
```

Linux RunPod helpers were added:

- `scripts/runpod_ai_gp_eval.sh`
- `scripts/runpod_ai_gp_train.sh`

## Next Engineering Steps

1. Run dense telemetry on `036` to prove the remaining failure distribution and
   target the next hard-case set.
2. Improve the teacher/controller on remaining hard states, then train the
   neural actor with anchoring that preserves the solved nominal course.
3. Promote to Swift-level only if multi-seed randomized validation reaches the
   Swift-level thresholds above.
4. Prepare AI-GP sim usage as shadow/integration testing until the policy clears
   those thresholds.
