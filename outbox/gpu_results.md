# GPU Training Results — RunPod RTX 3090

**Date:** 2026-03-15
**Pod:** nwcko9i3b6tz65 (RTX 3090, 24GB VRAM)

## Setup Notes
- Had to install `jax[cuda12]` (not included in `pip install -e ".[sim,rl]"`)
- Fixed 3 GPU compatibility bugs in `train_racing.py`:
  1. JAX GPU tensors can't be converted with `np.array()` — need `np.asarray()`
  2. PyTorch CUDA tensors need `.cpu()` before `.numpy()`
  3. JAX env.step() expects JAX GPU arrays, not numpy — added `_action_for_env()`

## Results

| Exp | Level | Steps | Envs | n_obs | Final Reward | Wall Time | Status |
|-----|-------|-------|------|-------|:------------:|:---------:|--------|
| **014** | L0 | 1.5M | 1024 | 2 | **7.29** | 206s | Validates n_obs=2 works with GPU compute |
| **015** | L2 | 3M | 1024 | 2 | **7.53** | 413s | First L2 training — reward still climbing |
| **016** | L2 | 10M | 1024 | 2 | **7.71** | 1466s | Extended L2 — converged, periodic dips |

## Analysis

### exp_014: n_obs=2 GPU validation
- **Validates Hard Rule #1**: n_obs=2 needs 1024 envs, not 64. With GPU + 1024 envs, reward reaches 7.29 (same level as exp_010's 7.36 with n_obs=0).
- Completed all 1.5M steps in 206s (vs exp_013's failure at 297k/500k in 600s on CPU).

### exp_015: First Level 2 training (3M steps)
- Reward climbed steadily: -45 → 7.53 over 3M steps.
- Peak reward ~7.73 around 2.4M steps, with periodic dips (~7.2) and recoveries.
- Reward was still improving at 3M — justified running exp_016.

### exp_016: Extended Level 2 training (10M steps)
- Final reward: **7.71** (converged around ~7.72-7.74 peak).
- Same periodic dip pattern every ~800k steps (reward drops ~0.3 then recovers).
- Reward plateaued after ~6M steps — additional 4M steps didn't improve significantly.
- **The periodic dips suggest v_loss instability** (values jump from 0.001 to 0.5+), possibly from occasional catastrophic trajectory resets.

### Comparison to previous experiments
| Exp | Level | Steps | Reward | Key difference |
|-----|-------|-------|--------|---------------|
| exp_010 | L0 | 500k (CPU) | 7.36 | Baseline, n_obs=0 |
| exp_013 | L0 | 297k (CPU) | 5.02 | n_obs=2, undertrained |
| exp_014 | L0 | 1.5M (GPU) | 7.29 | n_obs=2, GPU — validates compute was the issue |
| exp_015 | L2 | 3M (GPU) | 7.53 | First L2 training |
| exp_016 | L2 | 10M (GPU) | **7.71** | Best model, L2 converged |

### Key observation
The training reward (7.71) is **higher** on Level 2 than Level 0 (7.29). This seems paradoxical since L2 is harder. Possible explanations:
1. The reward function may be easier to maximize on randomized tracks (shorter paths?)
2. 10M steps vs 1.5M steps — more training, higher reward regardless of level
3. The reward may not correlate with actual lap completion — need to benchmark with sim

## Next Steps
1. **Benchmark exp_015 and exp_016 models on Level 2 sim** — the critical test. Training reward != lap completion.
2. **Download model checkpoints from pod** — git push from pod failed (no credentials). Models are at:
   - `/root/drone-rl-lab/results/exp_015_gpu_level2/model.ckpt`
   - `/root/drone-rl-lab/results/exp_016_gpu_level2_long/model.ckpt`
   SCP also blocked by RunPod proxy. Need to either set up git credentials or use `rsync` with a custom SSH port.
3. **Create attitude_rl_exp016.py** controller using the exp_016 checkpoint and test on Level 2 with 5 runs.

## Pod Status
Pod is still running. Models are on the pod but not yet transferred to local. Do NOT shut down the pod until models are retrieved.
