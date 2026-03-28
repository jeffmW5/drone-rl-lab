# exp_061 — Stochastic Deployment of exp_060 Model

## Hypothesis
The stochastic training policy navigates (28.02 reward) but the deterministic mean crashes at 0.66s. Deploying with stochastic sampling (sample from learned distribution instead of using mean) should recover the navigation behavior seen in training.

## Method
- **Model:** exp_060 checkpoint (body_frame_obs + soft_collision + bilateral_progress)
- **Change:** Set `DRONE_RL_STOCHASTIC=true` — controller samples from Normal(mean, std) instead of using mean
- **No retraining** — inference-only change
- **Paper basis:** Dynamic Entropy Tuning (2512.18336) — stochastic policies generalize where deterministic crash

## Config
No new training config. Benchmark env vars:
```
DRONE_RL_CKPT_PATH=results/exp_060_combined/model.ckpt
DRONE_RL_BODY_FRAME_OBS=true
DRONE_RL_STOCHASTIC=true
```

## Results

### Benchmark (level2_midair, 5 runs)
| Run | Flight Time (s) | Gates | Finished |
|:---:|:---:|:---:|:---:|
| 1 | 0.78 | 0 | No |
| 2 | 2.14 | 0 | No |
| 3 | 2.24 | 0 | No |
| 4 | 1.88 | 0 | No |
| 5 | 1.30 | 0 | No |

**Average: 1.67s flight, 0 gates, 0/5 finish**

### Comparison with deterministic (exp_060)
| Mode | Avg Flight (s) | Avg Gates |
|------|:---:|:---:|
| Deterministic (exp_060) | 0.66 | 0 |
| Stochastic (exp_061) | 1.67 | 0 |

## Analysis

- **2.5x longer flights** with stochastic sampling (1.67s vs 0.66s)
- Still 0 gates — stochastic noise alone doesn't solve navigation
- The learned std is too wide (full temperature=1.0) — action noise prevents precise control
- Supports the hypothesis that the mean is unstable, but full stochastic is too noisy
- **Next:** exp_062 — temperature scaling to find sweet spot between deterministic crash and stochastic noise

## Verdict
**PARTIAL** — stochastic deployment extends flight time 2.5x but doesn't achieve gate passage. Temperature scaling needed.
