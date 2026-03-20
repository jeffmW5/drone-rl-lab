# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_034** -- PBRS + speed_coef=0.7, fine-tune from exp_033
- **Training:** Mean reward 17.26 ± 5.50, 8M steps, 3299s. Peaks to 24.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 29.98s** (perfect hover, all 5 runs)
- **Finding:** PBRS eliminated the speed crash (exp_031=2.02s → exp_034=29.98s). But deterministic mean policy still hovers — stochastic exploration gets reward during training but mean policy stays at hover. survive_coef(0.5)+alt_coef(1.5)=2.0/step guaranteed for inaction is the anchor.

## Key Insight: survive_coef Is the Hover Anchor
| Reward Component | Hover | Navigate (1 m/s) |
|-----------------|-------|-------------------|
| survive_coef=0.5 | 0.5/step | 0.5/step |
| alt_coef=1.5 | ~1.5/step | ~1.5/step |
| progress_coef=50 | 0/step | ~1.0/step |
| speed_coef=0.7 | 0/step | ~0.7/step |
| **Total** | **~2.0/step** | **~3.7/step** |

The 1.7/step advantage of navigation isn't enough to shift the policy mean from the risk-free hover baseline. With survive=0, hover=1.5 vs navigate=3.2 — a 2x advantage.

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 026 | 9.77 | 0 | 28.8s | Stable hover baseline |
| 028 | 16.95 | 0.2 | 0.94s | speed_coef=1.0, old proximity (first gate!) |
| 032 | 11.87 | 0 | 2.92s | PBRS progress, speed=0.3 (crash) |
| 033 | 14.37 | 0 | 24.58s | + truncation fix (hover restored) |
| **034** | **17.26** | **0** | **29.98s** | **PBRS + speed=0.7 (no crash, still hover)** |

## In Progress
- **exp_035** -- survive_coef=0 (remove hover anchor), fine-tune from exp_034
- Single variable change: survive_coef 0.5 → 0.0. Tests whether free survival reward is the hover anchor.

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-034
- In progress: exp_035 (survive_coef=0)
