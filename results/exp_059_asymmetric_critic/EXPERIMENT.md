# exp_059 — Asymmetric Actor-Critic

## Hypothesis
Recent racing papers commonly use an asymmetric actor-critic: the actor trains on deployable observations, while the critic gets privileged state for better value estimates. Here that means the critic sees 28D of extra gate pose information while the actor still uses the standard 57D observation. If the value function improves, PPO should get cleaner advantages and push the policy toward gates more reliably than exp_056.

## Config
- Base: exp_056 bilateral progress reward
- Change: `asymmetric_critic: true`
- Actor obs: 57D
- Critic obs: 85D total (57D actor obs + 28D privileged state)
- File: `configs/exp_059_asymmetric_critic.yaml`

## Training Results
- **Mean reward: 32.502 +/- 1.149**
- Steps: 1,974,272 / 20,000,000 (budget-limited at 3600s)
- Wall time: 3603.3s on RTX 3090
- Pod: RunPod RTX 3090, pod `674pqvwr97pjhq`

### Reward curve
- iter 1-100: 11.75 -> 11.85
- iter 100-200: 11.85 -> 16.96
- iter 200-300: 16.96 -> 25.06
- iter 300-400: 25.06 -> 27.98
- iter 400-480: 27.98 -> 33.88

Peak logged reward during training: **33.88** at iter 480.

## Benchmark Results (level2_midair)
| Run | Flight time (s) | Gates | Finished |
|-----|-----------------|-------|----------|
| 1 | 0.52 | 0 | No |
| 2 | 0.52 | 0 | No |
| 3 | 0.84 | 0 | No |
| 4 | 1.38 | 0 | No |
| 5 | 0.70 | 0 | No |

**Average: 0.79s flight, 0 gates, 0/5 finishes**

## Comparison vs exp_056
| Metric | exp_056 bilateral progress | exp_059 asymmetric critic |
|--------|-----------------------------|---------------------------|
| mean_reward | 28.92 | **32.50** |
| peak logged reward | 40.96 | 33.88 |
| timesteps | 3,706,880 | 1,974,272 |
| wall time | 3601s | 3603s |

## What this tells us
1. **The asymmetric critic improved 1-hour efficiency.** exp_059 reached a higher final mean reward than exp_056 within the same 1-hour budget despite training on fewer timesteps.
2. **Learning was steadier early.** Reward climbed gradually through the hour instead of the large spike-then-decay pattern seen in exp_056.
3. **Benchmark still failed.** Even with better training reward, the matched mid-air benchmark remained at 0 gates and short flights.
4. **Value learning is still noisy.** v_loss remained volatile, sometimes spiking above 100, so the critic is better but not fully stable.
5. **This improves training, not deployment.** The asymmetric critic looks like a good training-side optimization, but it does not solve the deterministic deployment gap on its own.

## Notes
- This run was executed on a fresh RunPod RTX 3090 pod using the updated pod bootstrap.
- Startup timing logs showed first env/model setup by ~7s, first reset by ~14s, first rollout step completion by ~26s, and first training iteration by ~29s.
- The original run exposed a constructor bug in `AsymmetricAgent` (`std` passed into `nn.Linear`), which was fixed before the successful run.

## Suggested next experiment
Keep the asymmetric critic and focus on stabilizing the value function under racing rewards. The cleanest next step is likely a critic-side stabilization change rather than removing privileged inputs, since the current result already beats exp_056 on the same wall-clock budget.
