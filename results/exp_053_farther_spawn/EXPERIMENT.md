# exp_053 — Farther Spawn (spawn_offset=1.5)

## Hypothesis
Training-benchmark domain gap: training spawns 0.75m from gate mid-air, benchmark starts
2.06m from gate 0 at ground level. Doubling spawn_offset to 1.5m trains longer flights,
closer to benchmark distance.

## Config
- spawn_offset: 1.5 (2x from exp_046's 0.75)
- spawn_pos_noise: 0.3 (2x from 0.15)
- spawn_vel_noise: 0.5 (from 0.3)
- All other params same as exp_046

## Training
- **Mean reward:** ~29 (final, declined from peak)
- **Peak reward:** ~39.71 (iter 520)
- **Steps:** ~4M / 20M (time-budget limited at 3601s)
- **GPU:** RTX 4090 on RunPod

### Training Trajectory
- Smooth climb: 10 → 22 → 30 → 39.7 (peak at iter 520)
- Late decline: 39.7 → 28 (v_loss spikes, similar to exp_049)
- No hover trap at any point

## Benchmark (Level 2)
| Run | Time (s) | Gates | Finished |
|-----|----------|-------|----------|
| 1 | 0.64 | 0 | No |
| 2 | 0.72 | 0 | No |
| 3 | 0.78 | 0 | No |
| 4 | 0.62 | 0 | No |
| 5 | 0.72 | 0 | No |
| **Avg** | **0.70** | **0** | **0%** |

## Result
**FAILURE** — 0 gates, 0.70s avg flight. WORSE than exp_046's 1.3s.

## Analysis
1. **Farther horizontal spawn makes things WORSE:** Increasing spawn_offset from 0.75 to
   1.5 made the training harder (fewer rewards per episode) without fixing the actual
   domain gap. The benchmark crashed faster (0.70s vs 1.3s).

2. **The domain gap is VERTICAL, not horizontal:** The benchmark starts at z=0.01 (ground
   level) while training spawns at z≈0.7 (gate altitude). Increasing horizontal distance
   doesn't help because the policy still never trains on ground-level starts. The altitude
   domain gap is the critical factor.

3. **exp_054 (race start) is the real fix:** Must set random_gate_ratio=0.0 to train
   from the actual race start position ([-1.5, 0.75, 0.01]), eliminating both the
   altitude and distance domain gaps simultaneously.
