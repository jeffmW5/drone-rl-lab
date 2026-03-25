# exp_046 — Tight Logstd Clamp (max_logstd=-1.0, std≤0.37)

## Hypothesis
Same as exp_044/045 but with max_logstd=-1.0 (std ≤ 0.37). At std=0.37,
actions stay within ±0.74 of the mean (2σ), forcing precise policy learning.

## Training
- **Timesteps**: 4,345,856 (time-budget stopped at 3600s)
- **Mean reward**: 29.202 ± 0.447
- **Peak reward**: ~39.3 (NEW ALL-TIME HIGH)
- **Reward curve**: Ramp 9→15 (iter 0-130), then jump to 36-39 (iter 300-500)

## Learned Logstd
All actions clamped to effective std=0.37 during training.

## Benchmark (mid-air sim, 5 runs deterministic)
- **Flight time**: 1.22-1.62s (avg ~1.36s)
- **Gates passed**: 0/5 runs
- **Finished**: No
- **Behavior**: Consistent ~1.3s flights toward gate, crashes just before arrival.
  Much more consistent than exp_044 (0.7-1.4s) and exp_045 (0.7-2.2s).

## Result
**FAILURE at benchmark** but most consistent behavior yet. The tight clamp produces
a meaningful deterministic mean that flies toward the gate for ~1.3s before crashing.
The drone needs ~0.5s more flight time to reach the gate (0.75m at ~0.6 m/s).

## Key Insight
Tight logstd works — the mean learned actual navigation. But survive=0.05 doesn't
provide enough flight stability. With std=0.37 preventing hover trap, it should be
safe to increase survive_coef to extend flight time.
