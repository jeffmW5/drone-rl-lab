# exp_057 — Body-Frame Gate Observations

## Hypothesis
exp_056 showed bilateral progress creates correct gate direction, but progress_coef=50 is too
aggressive (dive crashes at 0.64s). Root cause: policy must combine world-frame gate vectors with
quaternion to determine gate direction — hard for 2-layer MLP. With body_frame_obs=true, "gate is
0.5m forward" is directly in the obs vector. Lower progress_coef=20 (from 50) since body-frame obs
provides directional info that progress_coef was brute-forcing. Keep bilateral progress.

## Config
- Base: exp_056 (bilateral progress, tight logstd, survive=0.05, random gate starts)
- Changes: `body_frame_obs: true` (obs: 55D instead of 57D), `progress_coef: 20` (from 50)
- File: `configs/exp_057_body_frame_obs.yaml`

## Training Results
- **Mean reward: 9.78** (flat throughout, never exceeded ~11.5)
- Steps: 1,671,168 / 20,000,000 (budget-limited at 3601s)
- Wall time: 3601s on RTX 3090 (secure cloud, ~$0.39/hr)
- GPU cost: ~$0.39

### Reward curve
- iter 1-40: 10.6 → 11.6 (initial learning)
- iter 40-100: 11.6 → 8.9 (dip, v_loss spike at 149)
- iter 100-410: 8.9 → 10.0 (slow recovery, flat plateau)

## Benchmark Results (level2_midair)
| Run | Flight time (s) | Gates | Notes |
|-----|----------------|-------|-------|
| 1 | 0.52 | 0 | Crash |
| 2 | 0.56 | 0 | Crash |
| 3 | 0.52 | 0 | Crash |
| 4 | 0.94 | 1 | Gate passage then crash |
| 5 | 0.62 | 0 | Crash |

**Average: 0.63s, 0.2 gates, 0 finishes**

## Diagnosis
1. **progress_coef=20 was too low** — exp_056 with progress_coef=50 hit 28.92 mean reward and
   showed clear gate direction; exp_057 with progress_coef=20 only reached 9.78 and showed minimal
   navigation learning. The body-frame obs didn't compensate for the weaker progress signal.
2. **Body-frame obs alone isn't enough** — The obs transformation is correct (55D: gate positions
   and normals in drone body frame) but the progress reward still needs to be strong enough to
   break the hover/drift equilibrium.
3. **Benchmark worse than exp_056** — 0.63s vs 0.64s, similar crash pattern but even less directed.
   The weak progress signal meant less aggressive approach behavior.
4. **One gate passage** — Run 4 passed a gate (0.94s), suggesting the body-frame obs does help
   with gate approach when the policy happens to be near one, but the weak reward signal doesn't
   consistently drive toward gates.

## Key Takeaway
Body-frame obs needs to be combined with strong progress_coef (50, not 20). The obs transformation
alone doesn't replace the need for a strong navigation reward. Next: exp_060 combines body_frame_obs
+ progress_coef=50 + soft_collision.
