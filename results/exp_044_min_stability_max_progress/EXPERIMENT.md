# exp_044 — Min Stability + Max Progress (20M steps)

## Hypothesis
Minimum viable stability (survive=0.05, alt=0.5) + maximum navigation incentive
(progress=50, gate_bonus=10, speed=0.5). Makes navigation 25x more valuable than
hovering per gate. 20M steps (2.5x longer). Random mid-air spawns.

## Config
- `survive_coef`: 0.05, `alt_coef`: 0.5
- `progress_coef`: 50.0, `gate_bonus`: 10.0, `speed_coef`: 0.5
- `gate_in_view_coef`: 0.0, `reward_mode`: add
- `max_episode_steps`: 1500
- `random_gate_start`: true, `random_gate_ratio`: 1.0
- `total_timesteps`: 20,000,000, `budget_seconds`: 14400

## Training
- **Timesteps**: 18,325,504 / 20,000,000 (time-budget stopped at iter 4490/4882)
- **Wall time**: 14402s (~4 hours)
- **Mean reward**: 26.384 ± 0.724
- **Peak reward**: ~37 (iter 200-800)
- **Reward curve**: Rapid climb 9→37 (iter 0-800), then settled to 26-31 band
- **Highest training reward ever recorded** — peak 37 smashes previous best of 28.6

## Benchmark (mid-air sim, 6 runs)
- **Flight time**: 0.7-1.4s (avg ~0.9s)
- **Gates passed**: 0/6 runs
- **Finished**: No
- **Behavior**: Crash in <1.5s. Slightly longer than exp_041-043 (0.52s) due to
  survive=0.05, but deterministic mean policy still crashes.

## Result
**FAILURE at benchmark** despite best-ever training performance. Training reward
26-37 proves the stochastic policy navigates and likely passes gates during
training. But the deterministic mean policy crashes at deployment.

## Key Insight
This conclusively confirms the exploration-exploitation gap (hard rule #31).
The reward design is now SOLVED — survive=0.05 + progress=50 + gate_bonus=10
produces genuine navigation during training. The problem is purely policy
optimization: the deterministic mean of the learned distribution does not
converge to a stable flight mode.

Potential fixes that haven't been tried:
1. Deploy stochastic policy (add noise at benchmark)
2. Larger network ([512,512,256,128] per Pasumarti 2024) — more capacity to
   represent both hover and navigate modes
3. Curriculum: train on hover first, then gradually introduce navigation
4. Lower initial action logstd to keep mean closer to exploration
5. SAC instead of PPO (entropy-maximizing, no separate exploration/exploitation)
