# AI-GP Agent Handoff

## Objective

Train a Swift-inspired PPO teacher in a fast GPU surrogate, then transfer its
gate-flight behavior into the AI Grand Prix live vision controller.

Do not return to one-off manual thrust pulses as the primary workstream.
Simulator runs are for measuring the real control contract and validating
trained policies with time-series telemetry.

## Current June 28 Status

The current structured-state export candidate is:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt
results/ai_gp_040_near_gate_teacher_bc_30m/ai_gp_structured_policy.json
```

`040` solves nominal six-gate evaluation and clears the structured-state
Swift-level surrogate threshold: `99.22%` average full-course success across
randomized seeds `1001`, `1002`, and `1003`, `5.99 / 6` mean gates, and zero
collisions.

`038` is not promoted. Its soft-floor penalty reduced average collisions from
`2.28%` to `1.37%`, but mean gates fell from `5.05` to `4.89` and early gate-1
and gate-3 misses increased. It is evidence for the next run, not the export
target.

`039` started from `036`, kept the soft-floor reward, and trained randomized
near-gate starts across all six active gate indices. It slightly improves the
three-seed average:

```text
success:    68.75% -> 69.60%
mean gates: 5.053  -> 5.057
missed:     29.04% -> 28.26%
collision:  2.28%  -> 2.15%
```

The previous blocker was the lack of a better teacher/controller target for hard
randomized states. That learning step was `040`: hybrid evaluation proved the
geometric teacher
solved near-gate corrections inside `10 m` of the active gate plane, while the
`039` actor remained useful outside that envelope. `040` behavior-cloned that
hybrid target into a pure actor:

```text
success:    69.60% -> 99.22%
mean gates: 5.057  -> 5.990
missed:     28.26% -> 0.78%
collision:  2.15%  -> 0.00%
```

Windows simulator integration showed that `040` does not transfer far enough.
Manual tuning and a 12-config runtime sweep reached active gate index 2 at best,
then failed around gate 1 or gate 2. `041` trained from `040` using the Windows
hard-case handoff in
`exports/ai_gp/ai_gp_windows_transfer_handoff_2026_06_28.json`.

`041` is exported for Windows A/B testing at:

```text
exports/ai_gp/ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json
```

It is not a clean surrogate promotion over `040`: nominal evaluation is
`100%`, but randomized three-seed average success is `98.83%` versus `99.22%`
for `040`, with zero collisions for both. Only promote `041` if Windows
simulator testing shows it clears farther than active gate index 2.

Use the dedicated Windows A/B runner for that decision:

```powershell
python -B scripts\run_ai_gp_policy_ab_windows.py `
  --attempts-per-policy 5 `
  --duration 30 `
  --thrust-multiplier 1.12 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id structured_ab_040_041_YYYYMMDD_HHMMSS
```

Do not start camera-only/live vision transfer until the structured-state
teacher can reliably clear more of the Windows simulator course.

## Current Implementation

- `ai_gp_rl/env.py`: Torch-vectorized quadrotor and gate environment.
- `ai_gp_rl/track.py`: canonical measured six-gate NED track and explicit
  NED-to-surrogate transform.
- `ai_gp_rl/model.py`: asymmetric PPO actor-critic with two 128-unit layers.
- `ai_gp_rl/contract.py`: legacy 18D, temporal 80D, and Swift teacher 31D contracts.
- `train_ai_gp.py`: PPO training, evaluation, metrics, and checkpoints.
- `distill_ai_gp_teacher.py`: behavior cloning and DAgger-style live-policy transfer.
- `scripts/evaluate_ai_gp_checkpoint.py`: deterministic time-series telemetry.
- `scripts/export_ai_gp_live_policy.py`: guarded legacy or temporal JSON policy export.
- `configs/ai_gp_002_swift_teacher_gpu_ppo_10m.yaml`: completed teacher benchmark.
- `configs/ai_gp_003_distilled_live_student.yaml`: failed behavior-cloning transfer.
- `configs/ai_gp_004_dagger_live_student.yaml`: failed DAgger transfer.
- `configs/ai_gp_005_temporal_dagger_live_student.yaml`: failed four-frame transfer.
- `configs/ai_gp_006_temporal_full_dagger_student.yaml`: failed full-rollout follow-up.
- `configs/ai_gp_018_real_track_teacher_10m.yaml`: failed first topology-correct
  teacher benchmark.
- `configs/ai_gp_019_real_track_teacher_reward_fix_10m.yaml`: failed one-shot
  crossing/altitude reward correction.
- `scripts/runpod_ai_gp.ps1`: deploy, smoke, train, monitor, and pull workflow.
- `tmp/ai-grand-prix-stack-remote/`: local simulator-control worktree. It is
  intentionally ignored because it contains generated sessions and captures.

The Swift teacher observes:

1. position and velocity
2. body-to-world rotation matrix
3. four active-gate corners in the body frame
4. previous action

It outputs normalized collective-thrust offset and roll, pitch, and yaw rates.
The reward contains gate progress, camera alignment, gate/finish bonuses,
body-rate and action-change penalties, and crash/out-of-bounds penalties.

## Critical Architecture Boundary

The 31D teacher is privileged and cannot be loaded directly into a live
controller.

Transfer requires one of:

- build a live estimator that recreates the teacher state and gate corners
- collect teacher trajectories and distill actions into a live-observation actor

The first 18D transfers and the June 13 temporal transfers were tested. None
produced a safe live-contract student. Four frames of box geometry improved
gate progress, but full student-state DAgger exposed unresolved observation
aliasing and collapsed into collisions.

## Benchmark Status

See `docs/AI_GP_10M_BENCHMARK_2026_06_09.md`.
See `docs/AI_GP_TEMPORAL_STUDENT_BENCHMARK_2026_06_13.md`.
See `docs/AI_GP_REAL_TRACK_TEACHER_BENCHMARK_2026_06_14.md`.

- CUDA smoke passed on an RTX 3090.
- The 10.09M teacher benchmark completed in about 70 seconds.
- Nominal surrogate evaluation completed all four gates in 256/256 episodes.
- Three randomized seeds completed 764/768 episodes with zero collisions.
- Time-series telemetry found no vertical runaway but found sub-centimeter gate
  margins and detoured trajectories.
- The 18D behavior-cloning and DAgger students both failed promotion.
- The 80D temporal student reached 2.55 nominal mean gates with zero collisions,
  but had 31.2% out-of-bounds and 15.2% vertical-runaway rates.
- The full-rollout temporal follow-up collapsed to 98.4-99.6% collisions.

Those benchmark claims are surrogate-only. The governed `motion_live_v1`
candidate passed gate 0 once in run `bounded_policy_20260613_103101`, but later
repeatability testing did not reproduce that result.

June 14 Windows evidence:

- symmetric live pulses proved the end-to-end roll mapping is identity
- the AI-GP six-gate NED centers and `2.72 m` gate dimensions are canonical in
  `ai_gp_rl/track.py` and reused by `scripts/run_ai_gp_bounded_windows.py`
- the current surrogate track turns the opposite direction after gate 0 from
  the AI-GP track, explaining the learned pre-turn failure
- a smooth `2.0 s` launch plus `1.0 s` authority ramp removed the hard-switch
  command discontinuity but still missed gate 0
- 18 authority-release attempts and nine thrust-gain attempts produced zero
  gate-0 passes
- allowing three missed-gate runs to continue exposed stale command holding:
  policy observations stopped and the last command remained active for
  `2.7-4.1 s` until altitude abort

The current policy is not approved for unrestricted control.

June 14 Linux real-track benchmark:

- the measured six-gate centers and `2.72 m` dimensions now have one canonical
  training/live definition
- NED maps to surrogate world as `(-north, east, altitude_offset - down)`
- active-gate plane misses now terminate training episodes and are reported
- the initial 10.09M teacher passed gate 0 in `0/256` nominal episodes
- one evidence-based reward correction improved nominal gate-0 passage to
  `29/256` (`11.3%`) but passed no gate 1
- three randomized seeds averaged `1.4%` gate-0 passage, `37.4%`
  out-of-bounds, and `39.1%` vertical-runaway rates

The topology-correct teacher failed promotion. No new student or live policy was
exported.

## Immediate Next Action

Fit the topology-correct surrogate to measured AI-GP command response before
another long PPO run.

1. Build a command-aligned dataset from synchronized Windows command and
   telemetry time-series.
2. Fit collective response, body-rate time constants, drag, latency, and
   effective mass/thrust scaling.
3. Obtain actual gate orientations if the simulator exposes them. Current gate
   centers and dimensions are measured, but plane yaw is inferred from the
   incoming course segment.
4. Revalidate the fitted environment with held-out command trajectories.
5. Retrain one structured-state teacher only after those checks pass.

Do not start another live-student distillation, reward matrix, or authority
matrix before dynamics fitting.

The bounded runner is preserved at:

```text
scripts/run_ai_gp_bounded_windows.py
```

It uses `AI_GP_RUNTIME_ROOT` when the operational execution worktree is not at
the default `tmp/ai-grand-prix-stack-remote` path.

Export syntax after a future student passes:

```bash
python3 scripts/export_ai_gp_live_policy.py \
  results/<student>/best_student.pt \
  results/<student>/best_deterministic_telemetry.json \
  results/<student>/windows_shadow_candidate.json
```

## Validation Status

- The earlier four-gate privileged teacher passed RunPod CUDA smoke, PPO,
  deterministic telemetry, and randomized telemetry.
- The topology-correct six-gate teacher passed execution validation but failed
  policy promotion in nominal and randomized telemetry.
- Training metrics alone were not used for acceptance.
- Every student transfer failed at least one collision, out-of-bounds, or
  vertical-behavior criterion.
- The exporter rejects privileged teachers, unknown live contracts, and
  students that fail deterministic telemetry promotion.

## Known Unknowns

- Gate orientations are not present in the captured track payload; surrogate
  plane yaw is inferred from the incoming horizontal course segment.
- Surrogate thrust and rate dynamics are not fitted to clean live telemetry.
- Live four-corner gate detection is not implemented.
- Reset and race-start synchronization require continued validation.
- Four frames spanning 60 ms are insufficient for reliable closed-loop
  transfer under the tested box-only observation.
- The governed student has no repeatable gate-0 or multi-gate evidence.
- Lost-gate handling currently permits stale command holding when the
  diagnostic gate-plane abort is disabled.
- The topology-correct teacher reached only `11.3%` nominal gate-0 passage and
  no gate-1 passages after one evidence-based reward correction.

## Promotion Criteria

Before live policy control:

- deterministic surrogate evaluation passes gate 0 at least 95%
- mean gates passed is at least 2
- collision rate is below 20%
- no persistent vertical runaway
- live thrust/rate signs and scaling are confirmed from time-series logs

Use bounded commands and shadow mode before unrestricted simulator flight.
Surrogate passage must never be described as Windows simulator passage.

## References

- `docs/AI_GP_SWIFT_EXECUTION_PLAN.md`
- `docs/SWIFT_ADOPTION_REPORT.md`
- `docs/AI_GP_RL_STRATEGY.md`
- `docs/AI_GP_CONTROL_CALIBRATION.md`
- `docs/AI_GP_RUNPOD.md`
- `docs/AI_GP_LINUX_AGENT_PROMPT.md`
- `docs/AI_GP_TRANSFER_TRAINING_HANDOFF_2026_06_28.md`
- `docs/AI_GP_VISION_TRANSITION_PLAN.md`
- `docs/AI_GP_REAL_TRACK_TEACHER_BENCHMARK_2026_06_14.md`
