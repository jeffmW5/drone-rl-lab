# Status -- Last Updated 2026-03-20

## Last Completed
- **exp_032** -- PBRS delta-progress reward (progress_coef=50, speed_coef=0.3)
- **Training:** Mean reward 11.87 ± 3.76, 8M steps, 3014s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 2.92s**
- **Finding:** PBRS successfully broke the hover local optimum (model attempts lateral movement instead of hovering in place). But crashes before reaching gates (2.92s avg). The reward change is working directionally — the model is no longer stuck hovering — but needs better value estimation to learn stable navigation.

## Experiment Summary (exp_028-032)
| Exp | speed_coef | Reward | Gates | Flight Time | Key Result |
|-----|-----------|--------|-------|-------------|------------|
| 026 | 0.1 | 9.77 | 0 | 28.8s | Stable hover baseline |
| 028 | 1.0 | 16.95 | 0.2 | 0.94s | First gate! But crash |
| 029 | 0.4 | 16.52 | 0 | 29.98s | Perfect hover, no nav |
| 030 | 0.55 | 15.64 | 0 | 25.98s | Edge of transition |
| 031 | 0.70 | 10.44 | 0 | 2.02s | Crash, no nav |
| **032** | **0.3+PBRS** | **11.87** | **0** | **2.92s** | **Hover broken, attempts nav** |

## In Progress
- **exp_033** -- Truncation fix (Pardo et al. 2018) + PBRS. Fine-tune from exp_032.
- Uses `terminated_buf` instead of `dones_buf` for GAE nextnonterminal — bootstraps through timeouts.

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_029 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-032
- In progress: exp_033 (truncation fix + PBRS)
