# Status -- Last Updated 2026-03-21

## Last Completed
- **exp_038** -- ent_coef=0.05 (7x increase), survive=0.5, fine-tune from exp_034
- **Training:** Mean reward 24.29 ± 2.08, 8M steps, 4064s.
- **Benchmark L2:** 0/5 finishes, 0 gates, **avg 0.96s** (crashes, same as survive=0 experiments)
- **Finding:** Higher entropy DESTROYED hover (exp_034 hovered at 29.98s with same survive=0.5). Pushed deterministic mean from hover to crash, NOT to navigation. The wider distribution's mean is unstable.

## APPROACHES EXHAUSTED
1. **Reward tuning** (exp_026-037): survive, speed, PBRS, proximity sweeps — all hover-or-crash
2. **Entropy regularization** (exp_038): destroys hover, doesn't produce navigation

## STRATEGY PIVOT: Structural Change (Curriculum)
Short episodes (300 steps = 6s) bound hover reward accumulation. Navigate to gate 0 = 255 reward vs hover = 150. Navigation is 1.7x better in short episodes.

## In Progress
- **exp_039** -- max_episode_steps=300 (6s), survive=0.5, PBRS, speed=0.7
- First structural change — episode length, not reward coefficients
- Fine-tune from exp_034 (stable hover baseline)

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s

## Queue Status
- Completed: exp_022-038
- In progress: exp_039 (short episodes curriculum)
