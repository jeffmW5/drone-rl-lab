# RunPod Throughput Findings -- 2026-04-19

Measured on the same RunPod RTX 3090 pod used for `exp_071_obs_normalization`.

## Inputs

- Config: `configs/exp_071_obs_normalization.yaml`
- Command:

```bash
cd /root/lsy_drone_racing
export PYTHONPATH=/root/lsy_drone_racing:/root/drone-rl-lab
export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax/compilation_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
/root/.pixi/bin/pixi run -e gpu python /root/drone-rl-lab/scripts/benchmark_racing_throughput.py \
    /root/drone-rl-lab/configs/exp_071_obs_normalization.yaml \
    --num-envs 512 --rollout-steps 4 --update-epochs 1
```

## Measured Result

```json
{
  "env_build_s": 5.7401,
  "first_reset_s": 2.9878,
  "env_step_s": 2.5487,
  "env_samples_per_s": 803.54,
  "policy_forward_s": 0.1773,
  "policy_samples_per_s": 11549.66,
  "ppo_update_s": 0.2836,
  "ppo_update_samples_per_s": 14443.32,
  "rollout_total_s": 2.7281,
  "rollout_samples_per_s": 750.71,
  "tensor_convert_s": 0.0021,
  "num_envs": 512,
  "device": "cuda",
  "jax_device": "gpu"
}
```

## Interpretation

- Environment stepping is the dominant cost.
- Policy forward is much faster than rollout.
- PPO update is also much faster than rollout.
- Tensor conversion is negligible in this measurement.
- Low GPU utilization during training is expected under this stack because the
  GPU spends substantial time waiting for env-side work.

## Practical Conclusion

If the goal is faster wall-clock training, the next wins will not come from
micro-optimizing the policy network update path. The next wins must come from:

1. reducing env-step / simulator cost
2. restructuring the rollout architecture to keep the accelerator busier
3. reducing per-iteration host-side overhead without shrinking useful batch size
