# exp_058 — Soft-Collision Curriculum (2-phase)

## Hypothesis
Hover-or-crash binary trap (Hard Rule #28, #34): hard termination on crash means the policy
can never learn to navigate THROUGH crashes. Fix: 2-phase curriculum inspired by "Mastering
Diverse Cluttered Tracks" (RA-L 2025, arXiv 2512.09571). Phase 1 (0-10M steps): crashes
don't terminate, just apply -5.0 penalty + respawn near random gate. Policy can collect gate
bonuses across multiple "lives" per episode. Phase 2 (10M+ steps): hard termination restored
once navigation is learned. Uses bilateral_progress from exp_056.

## Config
- Base: exp_046 (tight logstd, survive=0.05, random gate starts)
- Changes: `soft_collision: true`, `soft_collision_penalty: 5.0`, `soft_collision_steps: 10000000`
- Also: `bilateral_progress: true`, `progress_coef: 50.0` (from exp_056)
- File: `configs/exp_058_soft_collision.yaml`

## Training Results
- **Mean reward: 37.84** (high — soft collision allows accumulating reward across respawns)
- Steps: 5,373,952 / 20,000,000 (budget-limited at 3601s)
- Wall time: 3601s on RTX 3090 (community, ~$0.23/hr)
- GPU cost: ~$0.23

## Benchmark Results (level2_midair)
| Run | Flight time (s) | Gates | Notes |
|-----|----------------|-------|-------|
| 1 | 2.20 | 0 | Crash |
| 2 | 0.84 | 0 | Crash |
| 3 | 0.84 | 0 | Crash |
| 4 | 0.82 | 0 | Crash |
| 5 | 1.40 | 0 | Crash |

**Average: 1.22s, 0 gates, 0 finishes**

## Diagnosis
1. **Soft collision boosted training reward** — 37.84 mean vs exp_056's 28.92. Multiple lives
   per episode let the agent accumulate more reward, but this doesn't translate to benchmark.
2. **Benchmark performance unchanged** — 1.22s crash, same ~1.2s ceiling as exp_046-052.
   The soft collision curriculum doesn't help the deterministic deployment policy.
3. **Domain gap persists** — Training uses mid-air spawns (random_gate_start=1.0, spawn_offset=0.75).
   Benchmark starts at ground level. The model never trained on ground starts.
4. **Only 5.37M steps completed** — Didn't reach phase 2 (10M step transition to hard termination).
   Longer budget might help, but the domain gap is the more fundamental issue.

## Key Takeaway
Soft collision alone does NOT break the 1.2s benchmark ceiling. The bottleneck is the
training-benchmark domain gap (mid-air spawns vs ground start), not crash termination.
Even allowing the agent to survive crashes during training doesn't teach it to take off
from ground level at benchmark time.
