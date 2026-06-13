# AI-GP Agent Handoff

## Objective

Train a Swift-inspired PPO teacher in a fast GPU surrogate, then transfer its
gate-flight behavior into the AI Grand Prix live vision controller.

Do not return to one-off manual thrust pulses as the primary workstream.
Simulator runs are for measuring the real control contract and validating
trained policies with time-series telemetry.

## Current Implementation

- `ai_gp_rl/env.py`: Torch-vectorized quadrotor and gate environment.
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

All of those claims are surrogate-only. No Windows simulator policy candidate
currently exists.

## Immediate Next Action

1. Replace the four-frame box-only state with longer/recurrent memory or
   measured gate-corner/pose features.
2. Retrain the student without assuming more aggressive DAgger fixes the
   remaining partial observability.
3. Export only a `distilled_live_student` checkpoint with a passing
   deterministic telemetry report.
4. Run receive-only Windows shadow evaluation on a real simulator capture.
5. Require explicit `windows_simulator_passed` status before command authority.

Export syntax after a future student passes:

```bash
python3 scripts/export_ai_gp_live_policy.py \
  results/<student>/best_student.pt \
  results/<student>/best_deterministic_telemetry.json \
  results/<student>/windows_shadow_candidate.json
```

## Validation Status

- RunPod CUDA smoke, PPO benchmark, deterministic telemetry, and randomized
  telemetry passed for the privileged teacher.
- Training metrics alone were not used for acceptance.
- Every student transfer failed at least one collision, out-of-bounds, or
  vertical-behavior criterion.
- The exporter rejects privileged teachers, unknown live contracts, and
  students that fail deterministic telemetry promotion.

## Known Unknowns

- Placeholder gate positions do not yet match the AI-GP track.
- Surrogate thrust and rate dynamics are not fitted to clean live telemetry.
- Live four-corner gate detection is not implemented.
- Reset and race-start synchronization require continued validation.
- Four frames spanning 60 ms are insufficient for reliable closed-loop
  transfer under the tested box-only observation.
- No student checkpoint is ready for Windows simulator shadow evaluation.

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
