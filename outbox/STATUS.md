# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_037** -- survive_coef=0.3 (final bracket step), fine-tune from exp_034
- **Training:** Mean reward 18.10 ± 9.66 (bimodal sawtooth 5→28 throughout), 8M steps, 3880s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 1.62s** (4/5 crash ~0.85s, ONE outlier 4.64s)
- **Finding:** survive_coef=0.3 is at the phase transition edge — mostly crashes but one run survived 4.64s (first intermediate flight time in bracket). Bracket COMPLETE. Reward tuning alone cannot solve hover-or-crash. The 4.64s outlier suggests the deterministic mean policy is marginally unstable at 0.3 — it CAN fly briefly but can't sustain it.

## survive_coef Bracket (COMPLETE)
| survive_coef | Flight Time | Gates | Training Reward | Behavior |
|-------------|-------------|-------|-----------------|----------|
| 0.5 (exp_034) | 29.98s | 0 | 17.26 | Hover trap |
| **0.3 (exp_037)** | **1.62s** | **0** | **18.10** | **Crash (edge, one 4.64s outlier)** |
| 0.15 (exp_036) | 0.93s | 0 | 28.61 | Crash (highest reward!) |
| 0.0 (exp_035) | 0.96s | 0 | 11.32 | Crash |

## Key Conclusions
1. **Sharp phase transition** between survive_coef 0.3-0.5 — no smooth sweet spot
2. **Training reward inversely correlated** with benchmark below 0.5 (28.61 at 0.15 vs 17.26 at 0.5)
3. **Policy mode collapse**: stochastic exploration navigates, deterministic mean crashes
4. **Reward tuning exhausted** — need fundamentally different approach

## Recommended Next Directions
- **Higher ent_coef** (e.g., 0.05): keep deterministic mean closer to exploration distribution
- **Curriculum learning**: shorter episodes (max_episode_steps=300) to force fast learning
- **Stochastic deployment**: use stochastic policy at test time instead of deterministic mean
- **Architecture**: separate hover/navigate policy heads

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 028 | 16.95 | 0.2 | 0.94s | speed=1.0, old proximity (first gate!) |
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover) |
| 035 | 11.32 | 0 | 0.96s | survive=0 (crash) |
| 036 | 28.61 | 0 | 0.93s | survive=0.15 (crash, highest reward!) |
| **037** | **18.10** | **0** | **1.62s** | **survive=0.3 (crash edge, 4.64s outlier)** |

## In Progress
- None — awaiting orchestrator direction on strategy pivot

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-037
- In progress: none
