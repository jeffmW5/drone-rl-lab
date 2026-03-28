# exp_062 — Temperature-Scaled Deployment

## Hypothesis
Full stochastic (T=1.0) may be too noisy for deployment. Scaling temperature
(action = mean + T * std * noise) should find a sweet spot between the
deterministic mean crash and full stochastic noise.

## Method
- **Model:** exp_060 checkpoint (body_frame_obs + soft_collision + bilateral_progress)
- **Change:** Added `DRONE_RL_NOISE_SCALE` env var to attitude_rl_race.py. When set,
  computes action = mean + noise_scale * std * N(0,1) instead of pure stochastic/deterministic.
- **Sweep:** T = 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0
- **Paper basis:** Standard RL practice + entropy annealing theory (2405.20250)

## Results

### Coarse sweep (5 runs each)
| T | Avg Flight (s) | Gates | Best Run |
|:---:|:---:|:---:|:---:|
| 0.1 | 0.70 | 0/5 | 0.78s |
| 0.3 | 0.90 | **1/5** | 1.50s, 1 gate |
| 0.5 | 0.61 | 0/5 | 0.78s |
| 1.0 | 1.13 | 0/5 | 2.02s |

### Fine sweep (10 runs each)
| T | Avg Flight (s) | Gates | Best Run |
|:---:|:---:|:---:|:---:|
| 0.2 | 0.82 | **1/10** | 1.32s, 1 gate |
| 0.25 | 0.92 | 0/10 | 1.88s |
| 0.3 | 0.75 | 0/10 | 2.00s |
| 0.35 | 0.81 | 0/10 | 1.48s |
| 0.4 | 0.89 | 0/10 | 1.26s |

### Comparison
| Mode | Avg Flight (s) | Gates |
|------|:---:|:---:|
| Deterministic (exp_060) | 0.66 | 0/5 |
| Full stochastic (exp_061) | 1.67 | 0/5 |
| Best temperature T=0.3 | 0.90 | 1/15 |
| Best temperature T=0.2 | 0.82 | 1/10 |

## Analysis

- **2 gate passages in 70 total runs** across all temperatures — essentially noise
- Temperature scaling slightly extends flight time vs deterministic but far less than full stochastic
- No clear sweet spot — all temperatures produce 0 or 1 gate out of 5-10 runs
- The policy mean has NOT learned reliable gate navigation, so no amount of
  deployment-time noise injection can fix it
- **Conclusion:** The problem is in training, not deployment. Need to fix how the
  mean converges during training (entropy annealing, longer training, or architecture)

## Verdict
**FAILURE** — Temperature scaling does not solve the deployment gap. The mean
policy itself is inadequate. Need training-time fixes (exp_064 entropy annealing
or exp_063 extended training).
