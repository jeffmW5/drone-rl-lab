# exp_047 — Tight Logstd + More Survive (0.15)

## Hypothesis
exp_046 tight logstd produced 1.3s flights but crashed short of gate. Increase
survive from 0.05 to 0.15 to extend flight. Tight logstd should prevent hover trap.

## Training
- **Mean reward**: 10.005 ± 0.047 (extremely tight — pure hover)
- **Timesteps**: 2,809,856 (time-budget stopped)
- **Reward curve**: Flat at 10.0 from iter 200 onward

## Benchmark
- **Flight time**: 0.76s, 0 gates

## Result
**FAILURE** — Hover trap. Tight logstd did NOT prevent it. With survive=0.15 and
1500-step episodes, hover gives 0.15×1500=225 total — far exceeds navigate+crash (~30).
The math makes hover overwhelmingly dominant regardless of std.

## Key Insight
Tight logstd alone cannot break hover trap when episode length × survive_coef
dominates gate_bonus. Fix: short episodes to cap hover reward.
