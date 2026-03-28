# Facts

> Direct observations only. No explanations here.

## FACT-001
- **Statement:** The current best legacy trajectory-following lap is `exp_016` at 13.49s with 2/10 Level 2 finishes.
- **Type:** fact
- **Scope:** Legacy trajectory-following racing line
- **Supported by:** `README.md`, `outbox/exp016_l2_benchmark.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** A committed legacy controller result beats 13.49s or 2/10.

## FACT-002
- **Statement:** The early direct-racing high-water mark for average gates is `exp_023`, with 0.8 average gates and 0 finishes on 10 Level 2 runs.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv` line, Level 2 benchmark
- **Supported by:** `results/exp_023_racecore_oob/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** A committed direct-racing result exceeds 0.8 average gates.

## FACT-003
- **Statement:** `exp_056` reached 28.92 mean training reward and 0 gates with 0.64s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, bilateral progress config
- **Supported by:** `results/exp_056_bilateral_progress/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-004
- **Statement:** `exp_057` reached 9.78 mean training reward and 0.2 average gates with 0.63s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, body-frame obs plus reduced progress
- **Supported by:** `results/exp_057_body_frame_obs/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-005
- **Statement:** `exp_058` reached 37.84 mean training reward and 0 gates with 1.22s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, soft-collision curriculum
- **Supported by:** `results/exp_058_soft_collision/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-006
- **Statement:** `exp_060` reached 28.02 mean training reward and 0 gates with 0.66s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, combined body-frame + soft collision + strong progress
- **Supported by:** `results/exp_060_combined/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-007
- **Statement:** `exp_059` reached 32.502 +/- 1.149 mean training reward and 0 gates with 0.79s average matched mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, asymmetric critic, `level2_midair` benchmark
- **Supported by:** `results/exp_059_asymmetric_critic/metrics.json`, `results/exp_059_asymmetric_critic/benchmark.json`, `results/exp_059_asymmetric_critic/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-008
- **Statement:** The preexisting generic deployment controller could not load `exp_059`'s asymmetric checkpoint correctly; actor-only asymmetric loading support was required before benchmarking.
- **Type:** fact
- **Scope:** Evaluation tooling for asymmetric direct-racing checkpoints
- **Supported by:** local checkpoint load test on 2026-03-28, local patch to `lsy_drone_racing/control/attitude_rl_generic.py`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** A later benchmark path loads the same checkpoint correctly without architecture-aware logic.
