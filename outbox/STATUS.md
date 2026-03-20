# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_038** -- ent_coef=0.05 (7x increase, strategy pivot), survive=0.5, fine-tune from exp_034
- **Training:** Mean reward 24.29 ± 2.08 (less bimodal! stabilized ~16 then climbed to ~24), 8M steps, 4064s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.96s** (crashes instantly)
- **Finding:** Higher entropy DESTROYED hover stability. exp_034 (ent=0.007, survive=0.5) hovered at 29.98s. exp_038 (ent=0.05, survive=0.5) crashes at 0.96s. Entropy pushed deterministic mean from hover mode to crash mode, NOT to navigation. The wider distribution's mean is not viable.

## Key Insight
The deterministic mean policy is fundamentally the problem. Every approach that reduces hover (lower survive, higher entropy) produces crash — never navigation. The mean of any policy distribution we've trained lands in an unstable region of the dynamics.

## Approaches Exhausted
1. **Reward tuning** (exp_026-037): survive_coef bracket shows sharp phase transition, no sweet spot
2. **Entropy regularization** (exp_038): destroys hover without producing navigation

## Remaining Options
- **Stochastic deployment**: use stochastic policy at test time (the exploration policy DOES navigate)
- **Curriculum learning**: shorter episodes (max_episode_steps=300) to force faster learning
- **Architecture**: separate hover/navigate policy heads, or attention-based gate selection
- **Lower entropy compromise**: ent_coef=0.02 (midpoint) with survive=0.3 (edge of transition)

## Experiment Summary (recent)
| Exp | Reward | Gates | Flight Time | Key Change |
|-----|--------|-------|-------------|------------|
| 034 | 17.26 | 0 | 29.98s | PBRS + speed=0.7 (hover baseline) |
| 035 | 11.32 | 0 | 0.96s | survive=0 (crash) |
| 036 | 28.61 | 0 | 0.93s | survive=0.15 (crash, highest reward!) |
| 037 | 18.10 | 0 | 1.62s | survive=0.3 (crash edge) |
| **038** | **24.29** | **0** | **0.96s** | **ent=0.05 (crash, destroyed hover)** |

## In Progress
- None — awaiting orchestrator direction

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-038
- In progress: none
