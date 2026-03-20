# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_037** -- survive_coef=0.3 (final bracket step), fine-tune from exp_034
- **Training:** Mean reward 18.10 ± 9.66 (bimodal sawtooth 5→28), 8M steps, 3880s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 1.62s** (4/5 crash ~0.85s, ONE 4.64s outlier)
- **Finding:** survive_coef=0.3 at phase transition edge — mostly crashes but one intermediate flight. Bracket COMPLETE: no survive_coef sweet spot. Reward tuning exhausted.

## survive_coef Bracket (COMPLETE — no sweet spot)
| survive_coef | Flight Time | Gates | Training Reward | Behavior |
|-------------|-------------|-------|-----------------|----------|
| 0.5 | 29.98s | 0 | 17.26 | Hover trap |
| 0.3 | 1.62s | 0 | 18.10 | Crash (edge, 4.64s outlier) |
| 0.15 | 0.93s | 0 | 28.61 | Crash (highest reward!) |
| 0.0 | 0.96s | 0 | 11.32 | Crash |

## STRATEGY PIVOT: Reward Tuning → Policy Optimization
Reward tuning (exp_026-037) produced two modes with no transition:
- **Hover**: stable but never navigates (survive≥0.5)
- **Crash**: stochastic policy navigates during training but deterministic mean crashes (survive<0.5)

Next: **entropy regularization** — ent_coef 0.007→0.05 to prevent policy mode collapse.

## In Progress
- **exp_038** -- ent_coef=0.05 (7x increase), survive=0.5, fine-tune from exp_034
- Strategy pivot from reward tuning to policy optimization
- Tests whether higher entropy prevents the deterministic mean from collapsing to hover

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-037
- In progress: exp_038 (high entropy, strategy pivot)
