#!/bin/bash
# =============================================================================
# Drone RL Lab — Lightweight JAX Warmup
# =============================================================================
# Front-loads JAX GPU startup and a single env reset/step so the first visible
# training run spends less time in opaque compile work.
#
# Usage:
#   bash scripts/warmup_jax.sh
#   bash scripts/warmup_jax.sh configs/exp_NNN.yaml
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-}"

if ! command -v drone-rl-gpu-python >/dev/null 2>&1; then
    echo "[warmup] ERROR: drone-rl-gpu-python not found. Run scripts/setup_runpod.sh first."
    exit 1
fi

if [ -n "${CONFIG_PATH}" ] && [ ! -f "${CONFIG_PATH}" ]; then
    echo "[warmup] ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

echo "[warmup] Repo: ${REPO_DIR}"
echo "[warmup] Config: ${CONFIG_PATH:-<generic>}"

PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
drone-rl-gpu-python - "${CONFIG_PATH}" <<'PY'
import os
import sys
import time

start = time.perf_counter()


def log(message: str) -> None:
    elapsed = time.perf_counter() - start
    print(f"[warmup +{elapsed:6.1f}s] {message}", flush=True)


def run_generic_jax_warmup() -> None:
    import jax
    import jax.numpy as jnp

    log(f"JAX cache dir: {os.environ.get('JAX_COMPILATION_CACHE_DIR', '<unset>')}")
    log(f"JAX devices: {jax.devices()}")

    @jax.jit
    def fused_kernel(x, y):
        return jnp.tanh(x @ y + 0.1).sum(axis=1)

    x = jnp.ones((512, 512), dtype=jnp.float32)
    y = jnp.ones((512, 512), dtype=jnp.float32)
    log("Compiling generic JAX matmul warmup...")
    fused_kernel(x, y).block_until_ready()
    log("Generic JAX warmup complete.")


def run_env_specific_warmup(config_path: str) -> None:
    import numpy as np
    import torch
    import yaml

    from train_racing import _action_for_env, _to_np, build_args
    from lsy_drone_racing.control.train_rl import Agent, make_envs

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config.get("backend", "hover") != "racing":
        log(f"Config backend={config.get('backend')} is not racing; skipping env warmup.")
        return

    racing_cfg = config["racing"]
    args = build_args(racing_cfg)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    jax_device = args.jax_device if args.cuda else "cpu"
    jax_on_gpu = jax_device == "gpu"
    level = racing_cfg.get("level", "level0")
    env_type = racing_cfg.get("env_type", "trajectory")

    reward_coefs = {
        "n_obs": racing_cfg.get("n_obs", 2),
        "rpy_coef": racing_cfg.get("rpy_coef", 0.06),
        "d_act_th_coef": racing_cfg.get("d_act_th_coef", 0.4),
        "d_act_xy_coef": racing_cfg.get("d_act_xy_coef", 1.0),
        "act_coef": racing_cfg.get("act_coef", 0.02),
        "gate_aware": racing_cfg.get("gate_aware", False),
    }

    log(
        f"Creating env-specific warmup "
        f"(env_type={env_type}, jax_device={jax_device}, torch_device={device})..."
    )

    if env_type == "race":
        from lsy_drone_racing.control.train_race import make_race_envs

        race_coefs = {
            **reward_coefs,
            "gate_bonus": racing_cfg.get("gate_bonus", 5.0),
            "proximity_coef": racing_cfg.get("proximity_coef", 2.0),
            "speed_coef": racing_cfg.get("speed_coef", 0.1),
            "max_episode_steps": racing_cfg.get("max_episode_steps", 1500),
            "oob_coef": racing_cfg.get("oob_coef", 0.0),
            "z_low": racing_cfg.get("z_low", 0.0),
            "z_high": racing_cfg.get("z_high", 2.0),
            "alt_coef": racing_cfg.get("alt_coef", 0.0),
            "survive_coef": racing_cfg.get("survive_coef", 0.0),
            "vz_coef": racing_cfg.get("vz_coef", 0.0),
            "vz_threshold": racing_cfg.get("vz_threshold", 0.5),
            "random_gate_start": racing_cfg.get("random_gate_start", False),
            "random_gate_ratio": racing_cfg.get("random_gate_ratio", 1.0),
            "spawn_offset": racing_cfg.get("spawn_offset", 0.75),
            "spawn_pos_noise": racing_cfg.get("spawn_pos_noise", 0.15),
            "spawn_vel_noise": racing_cfg.get("spawn_vel_noise", 0.3),
            "bilateral_progress": racing_cfg.get("bilateral_progress", False),
            "body_frame_obs": racing_cfg.get("body_frame_obs", False),
            "soft_collision": racing_cfg.get("soft_collision", False),
            "soft_collision_penalty": racing_cfg.get("soft_collision_penalty", 5.0),
            "soft_collision_steps": racing_cfg.get("soft_collision_steps", 5000000),
            "asymmetric_critic": racing_cfg.get("asymmetric_critic", False),
        }
        race_config = racing_cfg.get("race_config", f"{level}_attitude.toml")
        envs = make_race_envs(
            config=race_config,
            num_envs=1,
            jax_device=jax_device,
            torch_device=device,
            coefs=race_coefs,
        )
    else:
        envs = make_envs(
            config=f"{level}.toml",
            num_envs=1,
            jax_device=jax_device,
            torch_device=device,
            coefs=reward_coefs,
        )

    try:
        obs, _ = envs.reset()
        obs_tensor = torch.tensor(_to_np(obs), dtype=torch.float32).to(device)
        agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        log("Running single env step warmup...")
        envs.step(_action_for_env(action, jax_on_gpu))
        log("Env-specific warmup complete.")
    finally:
        envs.close()


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else ""
    run_generic_jax_warmup()
    if config_path:
        run_env_specific_warmup(config_path)
    else:
        log("No config provided; skipped env-specific warmup.")


if __name__ == "__main__":
    main()
PY
