# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_039** -- max_episode_steps=300 (6s curriculum), survive=0.5, fine-tune from exp_034
- **Training:** Mean reward 20.50 ± 6.45 (bimodal 6-26 throughout), 8M steps, 3810s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 18.35s** (BISTABLE: 3/5 hover 29.98s + 2/5 crash ~0.9s)
- **Finding:** NEW behavior — first bistable policy. Short episodes shifted the deterministic mean to the phase transition edge. The policy is marginally stable: some initial conditions hover, others crash. Still 0 gates — curriculum alone didn't produce navigation, but it DID change the policy's stability profile.

## Approaches Attempted
1. **Reward tuning** (exp_026-037): survive, speed, PBRS sweeps — all firmly hover-or-crash
2. **Entropy regularization** (exp_038): destroys hover → crash, no navigation
3. **Short episodes** (exp_039): bistable (hover+crash), edge of transition, still 0 gates

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover baseline) |
| 035 | 11.32 | 0 | 0.96s | survive=0 (crash) |
| 036 | 28.61 | 0 | 0.93s | survive=0.15 (crash) |
| 037 | 18.10 | 0 | 1.62s | survive=0.3 (crash edge) |
| 038 | 24.29 | 0 | 0.96s | ent=0.05 (crash) |
| **039** | **20.50** | **0** | **18.35s** | **300-step episodes (BISTABLE: 3/5 hover + 2/5 crash)** |

## In Progress
- None — awaiting orchestrator direction

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-039
- In progress: none
