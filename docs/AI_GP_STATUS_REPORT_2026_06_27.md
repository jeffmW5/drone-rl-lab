# AI-GP Pilot Status - 2026-06-27

## Short Answer

We are not "passing 66% of gates." The best current structured-state policy is
roughly `98.8-99.4%` full-course success under randomized surrogate evaluation,
depending on seed. That means it finishes all six gates in about that fraction
of episodes. Its mean gate count is about `5.98-5.99 / 6`.

Nominal, non-randomized evaluation is solved by `040`: `100%` success
over 512 episodes. The remaining problem is robustness and transfer, not basic
gate sequencing.

## Current Best Artifact

Use this as the current structured-state AI-GP sim/shadow-test candidate:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

`040` is now the current best structured-state candidate. `039`, `036`, `035`,
and `031` remain previous baselines:

```text
results/ai_gp_039_all_gate_soft_floor_ppo_30m/best_policy.pt
results/ai_gp_039_all_gate_soft_floor_ppo_30m/ai_gp_structured_policy.json
results/ai_gp_036_weighted_final_approach_ppo_20m/best_policy.pt
results/ai_gp_036_weighted_final_approach_ppo_20m/ai_gp_structured_policy.json
results/ai_gp_035_hard_case_final_approach_ppo_20m/best_policy.pt
results/ai_gp_035_hard_case_final_approach_ppo_20m/ai_gp_structured_policy.json
results/ai_gp_031_randomized_full_course_ppo_120m/best_policy.pt
results/ai_gp_031_randomized_full_course_ppo_120m/ai_gp_structured_policy.json
```

Randomized validation for `040`:

- seed `1001`: `98.83%` success, `5.98` mean gates, `1.17%` missed gates
- seed `1002`: `99.41%` success, `5.99` mean gates, `0.59%` missed gates
- seed `1003`: `99.41%` success, `5.99` mean gates, `0.59%` missed gates

Previous randomized validation for `039`:

- seed `1001`: `71.29%` success, `5.12` mean gates, `27.15%` missed gates
- seed `1002`: `70.31%` success, `5.10` mean gates, `26.76%` missed gates
- seed `1003`: `67.19%` success, `4.96` mean gates, `30.86%` missed gates

Previous randomized validation for `036`:

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

`ai_gp_037_gate1_gate5_final_approach_ppo_20m` tested a heavier gate-1/gate-5
replay continuation from `036`. It is not promoted. Embedded training eval
briefly matched `036` at `73.83%` success and `5.25` mean gates, but held-out
validation traded missed gates for collisions:

- average success: `68.75% -> 67.84%`
- average mean gates: `5.05 -> 5.06`
- average missed-gate rate: `29.04% -> 28.52%`
- average collision rate: `2.28% -> 3.71%`

This means the next useful step is not simply more gate-1/gate-5 replay. The
next change should reduce collision while preserving the missed-gate reduction,
or use a better teacher/controller target for the final-approach states.

`ai_gp_038_soft_floor_final_gate_ppo_20m` added a soft-floor altitude penalty
and moderate gate-1/gate-5 replay from `036`. It is not promoted over `036`.
The embedded best eval reached `75.98%` success, but three held-out randomized
seeds showed a mixed and small result:

- average success: `68.75% -> 69.47%`
- average mean gates: `5.05 -> 4.89`
- average missed-gate rate: `29.04% -> 29.17%`
- average collision rate: `2.28% -> 1.37%`

Failure distribution explains the tradeoff. `038` reduced gate-5 collision and
late misses, but increased early gate-1 and gate-3 misses. That makes it useful
evidence, not a clear export candidate.

`ai_gp_039_all_gate_soft_floor_ppo_30m` trained on RunPod from `036`, kept the
soft-floor penalty from `038`, and trained near-gate starts across all six
active gate indices. It is promoted as the current candidate because it slightly
improved the three-seed average without increasing missed gates or collisions:

- average success: `68.75% -> 69.60%`
- average mean gates: `5.053 -> 5.057`
- average missed-gate rate: `29.04% -> 28.26%`
- average collision rate: `2.28% -> 2.15%`

Nominal `039` validation over 512 episodes remains clean: `100%` success, `6.0`
mean gates, zero failures, and `0.705 m` minimum crossing margin. The structured
policy export was written to:

```text
results/ai_gp_039_all_gate_soft_floor_ppo_30m/ai_gp_structured_policy.json
```

Hybrid validation then found the missing lever: using the geometric teacher only
inside `10 m` of the active gate plane produced `100%` success and zero failures
on randomized seeds `1001`, `1002`, and `1003`. `040` behavior-cloned that
hybrid target from `039`: teacher actions in the near-gate envelope and frozen
`039` actor actions elsewhere.

`ai_gp_040_near_gate_teacher_bc_30m` is promoted. Three-seed randomized
validation versus `039`:

- average success: `69.60% -> 99.22%`
- average mean gates: `5.057 -> 5.990`
- average missed-gate rate: `28.26% -> 0.78%`
- average collision rate: `2.15% -> 0.00%`

Nominal `040` validation over 512 episodes is clean: `100%` success, `6.0`
mean gates, zero failures, and `0.836 m` minimum crossing margin. The structured
policy export was written to:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

Linux RunPod helpers were added:

- `scripts/runpod_ai_gp_eval.sh`
- `scripts/runpod_ai_gp_train.sh`

## Next Engineering Steps

1. Treat `040` as the current structured-state AI-GP sim/shadow-test candidate.
2. Run the exported `040` policy in the actual AI-GP simulator integration path
   and compare time-series gate crossings to surrogate telemetry.
3. Keep `039` and the hybrid-teacher reports as fallback/diagnostic baselines.
4. Do not move to camera-only/live vision until the structured-state `040`
   behavior is verified in the AI-GP simulator runtime.
