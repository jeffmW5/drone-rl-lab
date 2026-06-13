# AI-GP Temporal Student Benchmark - 2026-06-13

## Scope

This benchmark tested deployable students only. The privileged 31D teacher
remained separate and was used only to label surrogate training states.

The new `temporal_live_v1` contract contains four 20D frames:

- body velocity, gravity, and angular rates
- gate center, separate width and height, area, confidence, and age
- previous normalized action

The flattened actor input is 80D. The full actor-critic observation is 94D.

## CUDA Validation

RunPod provisioned an RTX 3090 replacement pod after the prior pod's GPU was
unavailable.

- required teacher smoke: 256 envs, observation `(256, 45)`, passed
- temporal smoke: 256 envs, observation `(256, 94)`, passed
- targeted upstream tests on the pod: 14 passed

## Benchmark Results

Both runs used 10,092,544 interactions.

| Experiment | Time | Nominal result |
|---|---:|---|
| `ai_gp_005_temporal_dagger_live_student` | 78.9 s | final: 2.55 mean gates, 7.4% completion, 0% collision, 31.2% out of bounds, 15.2% vertical runaway |
| `ai_gp_006_temporal_full_dagger_student` | 62.4 s | best: 2.05 mean gates, 1.6% completion, 98.4% collision |

The `ai_gp_005` final randomized evaluation at seed 10045 produced:

- 91.4% gate-0 passage
- 2.33 mean gates
- 18.8% completion
- 11.7% collisions
- 47.3% out of bounds
- 42.2% vertical runaway

No checkpoint passed deterministic telemetry promotion. No JSON policy artifact
was exported.

## Time-Series Findings

The four-frame student passed real surrogate gate planes, not reward-only gate
proxies. Nominal final telemetry recorded 652 gate crossings across 256
episodes, but the minimum crossing margin was only 0.91 mm.

Failure distributions were not inferred from snapshots:

- `ai_gp_005` final had 159/256 episodes pass three gates, but only 19 finished.
- vertical failures occurred after gate transitions and included sustained
  positive vertical velocity with large positive collective commands.
- fully student-driven comparison against teacher labels measured RMSE of
  approximately 0.44 collective, 0.44 roll, 0.22 pitch, and 0.12 yaw.
- collective exceeded absolute 0.8 on about 12% of fully student-driven samples.

The full-rollout follow-up increased student-state exposure to 100% and weighted
collective/roll loss. It removed vertical runaway by collapsing into
98.4-99.6% collision, demonstrating that rollout exposure alone does not resolve
the remaining live-observation aliasing.

## Implemented Changes

- backward-compatible temporal observation contract and history in the GPU env
- separate gate width and height in exported session features
- artifact metadata for base features, history length, and flattened features
- dependency-light temporal history assembly for Windows receive-only shadow
- action-channel loss weights and per-channel training diagnostics
- evaluation and checkpoint scoring that include bounds and vertical behavior

## Decision

Do not run a 100M student job and do not grant command authority.

The next high-value experiment must add longer/recurrent state estimation or
measured four-corner/gate-pose features. Any future candidate still requires
deterministic surrogate telemetry, receive-only Windows shadow, and bounded
Windows simulator evaluation before a racing claim.
