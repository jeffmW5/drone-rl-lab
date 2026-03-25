# exp_048 — Short Episodes + Tight Logstd + Survive=0.15

## Hypothesis
Combine tight logstd (exp_046) with short episodes (200 steps) + survive=0.15.
exp_046: tight std → real navigation but crashes 0.5s short. exp_047: survive=0.15
→ hover trap with 1500-step episodes. Short episodes cap hover reward:
survive(0.15)×200=30. Navigation to gate (~100 steps): survive(15)+progress(~30)+gate_bonus(10)=55.
Navigation is 1.8x better than hover.

## Training
- **Timesteps**: 2,383,872 (time-budget stopped at 3600s)
- **Mean reward**: 18.881 ± 0.611
- **Reward curve**: Moderate — neither the high 29+ of exp_046 nor the flat 10.0 of exp_047

## Benchmark
- **Flight time**: ~0.54s (worse than exp_046's 1.3s)
- **Gates passed**: 0
- **Behavior**: Quick crash — short episodes trained a policy that expects truncation at 200 steps,
  producing aggressive maneuvers that crash in the longer benchmark sim.

## Result
**FAILURE** — Short episodes made it worse. The 200-step cap (4s) trained an overly
aggressive policy that doesn't generalize to longer horizons. exp_046 (1500-step, survive=0.05)
remains the best benchmark performer.

## Key Insight
Short episodes change the effective MDP — the policy learns to optimize for a 4-second
horizon, producing risky maneuvers that crash immediately in unbounded deployment.
The survive_coef balance must be solved within the original episode length.
