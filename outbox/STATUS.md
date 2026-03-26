# Status -- Last Updated 2026-03-25

## BREAKTHROUGH: Bilateral Progress Reward (exp_056)
**Training reward 40.96 at iter 480 (2M steps) — new all-time high and still climbing.**

exp_056 uses `bilateral_progress=true`: raw delta `(prev_dist - curr_dist)` instead of
`max(prev_dist - curr_dist, 0)`. Moving AWAY from the gate is now penalized, creating
a proper gradient for the deterministic mean policy. Previous one-sided reward only
rewarded stochastic movements that happened to get closer.

Training on RunPod RTX 3090, ~37min into 60min budget.

## Structural Changes Implemented (Ready to Train)
Three literature-based structural changes (exp_057/058/059) have been implemented and
are ready for training once exp_056 completes:

| Exp | Change | Obs | Key Mechanism |
|-----|--------|-----|---------------|
| 057 | Body-frame gate obs | 55D | Gate rel pos + normal in body frame |
| 058 | Soft-collision curriculum | 57D | Phase 1: no crash termination, Phase 2: hard |
| 059 | Asymmetric actor-critic | 85D (57+28) | Critic sees all gate pos/quat |

## In Progress
- **exp_056** -- bilateral_progress=true, training on RunPod (reward 40.96 at 2M/20M steps)

## Recent Results (exp_053-055)
| Exp | Key Change | Train Reward | Benchmark | Gates |
|-----|-----------|-------------|-----------|-------|
| 053 | spawn_offset=1.5 | ? | 0.70s | 0 |
| 054 | race start (ground) | 8.89 (flat) | FAILURE | 0 |
| 055 | takeoff incentive | training | ? | ? |
| **056** | **bilateral progress** | **40.96 (2M steps)** | **pending** | **pending** |

## Current Best
- **Racing L2 (training reward):** exp_056 -- 40.96 mean at 2M steps (new record, still climbing)
- **Racing L2 (benchmark):** exp_046 -- 0 gates, 1.3s consistent flights toward gate
- **Racing L2 (lap time, legacy):** exp_016 -- 13.49s, 2/10 finishes

## Queue
- In progress: exp_056 (bilateral progress, training on RunPod)
- Ready to train: exp_057 (body-frame obs), exp_058 (soft-collision), exp_059 (asymmetric critic)
- Recommended order: exp_057 → exp_058 → exp_059
