# exp_045 — Logstd Clamp (max_logstd=0.5)

## Hypothesis
Same as exp_044 but with max_logstd=0.5 (std ≤ 1.65) to prevent std blowup.
exp_044's pitch std exploded to 90, making the mean meaningless. Clamping forces
the policy to learn a meaningful mean action.

## Config
- Same as exp_044: survive=0.05, alt=0.5, progress=50, gate_bonus=10, speed=0.5
- **NEW**: max_logstd=0.5 (clamps actor_logstd in forward pass)
- budget_seconds: 3600 (1 hour rerun)

## Code Change
Added `max_logstd` parameter to Agent class in `train_rl.py`:
- `agent.max_logstd = 0.5` set after Agent creation
- In `get_action_and_value`: `action_logstd = torch.clamp(action_logstd, max=self.max_logstd)`

## Training
- **Timesteps**: 4,681,728 / 20,000,000 (time-budget stopped at 3602s)
- **Mean reward**: 26.518 ± 0.962
- **Peak reward**: ~37 (matching exp_044)
- **Reward curve**: Same as exp_044 — ramp to 30+, plateau at 26-35

## Learned Logstd
| Action | Stored logstd | Effective std (clamped) | exp_044 std |
|--------|--------------|------------------------|------------|
| Roll | 0.035 | 1.04 (not clamped) | 0.92 |
| Pitch | 1.276 | **1.65 (clamped at 0.5)** | **90.5** |
| Yaw | 2.500 | 1.65 (clamped, zeroed) | 489 |
| Thrust | -0.481 | 0.62 (not clamped) | 0.47 |

## Benchmark (mid-air sim)
- **Deterministic (5 runs)**: 0.7-2.2s, 0 gates
- **Stochastic scale=0.5 (3 runs)**: 0.5-1.2s, 0 gates
- **Stochastic scale=0.3 (3 runs)**: 1.0-1.8s, 0 gates
- **Stochastic scale=0.1 (3 runs)**: 1.3-1.5s, 0 gates

## Result
**FAILURE** — Slightly better flight times than exp_044 (up to 2.2s vs 1.4s) but
still 0 gates across all deployment modes. Clamping logstd to 0.5 (std=1.65) is
still too wide for the 2×64 network to learn a precise mean.

## Key Insight
max_logstd=0.5 is too permissive. std=1.65 on [-1,1] actions means the mean only
needs to point in the right direction within ±1.65 — not precise enough for flight
control. Options: (1) tighter clamp (max_logstd=-1, std≤0.37), (2) larger network
for more expressive mean, (3) different algorithm (SAC with auto-tuned entropy).
