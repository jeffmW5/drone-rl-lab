#!/usr/bin/env python3
"""Benchmark throughput of the racing training stack.

Measures:
- environment reset / warmup time
- rollout throughput using the real policy + env.step path
- policy forward throughput
- PPO update throughput on collected rollout data

This is intended to answer whether the main bottleneck is environment stepping,
policy inference, or PPO optimization before attempting a trainer swap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jp
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LSY_ROOT = Path("/media/lsy_drone_racing")
if str(LSY_ROOT) not in sys.path:
    sys.path.insert(0, str(LSY_ROOT))

from lsy_drone_racing.control.train_rl import ActionPenalty, Agent, Args, FlattenJaxObservation


def _to_np(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, jax.Array):
        return np.asarray(value)
    return np.asarray(value)


def _action_for_env(action: torch.Tensor, jax_on_gpu: bool):
    action_np = action.detach().cpu().numpy()
    if jax_on_gpu:
        return jax.device_put(jp.asarray(action_np), device=jax.devices("gpu")[0])
    return action_np


def build_args(racing_cfg: dict):
    valid_keys = {field.name for field in fields(Args)}
    args = Args.create(**{k: v for k, v in racing_cfg.items() if k in valid_keys})
    return argparse.Namespace(**asdict(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to racing experiment YAML config")
    parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
        help="Number of env steps to measure in the rollout benchmark",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=3,
        help="How many PPO update epochs to run in the benchmark section",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch/NumPy seed used for repeatability",
    )
    parser.add_argument(
        "--rollout-backend",
        choices=("torch", "jax"),
        default="torch",
        help="Use the current Torch policy path or an experimental JAX-only rollout path",
    )
    return parser.parse_args()


def load_experiment(config_path: Path) -> tuple[dict, dict]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if config.get("backend") != "racing":
        raise ValueError(f"Expected backend=racing, got {config.get('backend')!r}")

    return config, config["racing"]


def build_env(racing_cfg: dict, device: torch.device, jax_device: str):
    reward_coefs = {
        "n_obs": racing_cfg.get("n_obs", 2),
        "rpy_coef": racing_cfg.get("rpy_coef", 0.06),
        "d_act_th_coef": racing_cfg.get("d_act_th_coef", 0.4),
        "d_act_xy_coef": racing_cfg.get("d_act_xy_coef", 1.0),
        "act_coef": racing_cfg.get("act_coef", 0.02),
        "gate_aware": racing_cfg.get("gate_aware", False),
    }

    env_type = racing_cfg.get("env_type", "trajectory")
    level = racing_cfg.get("level", "level0")
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
            "progress_coef": racing_cfg.get("progress_coef", 0.0),
            "gate_in_view_coef": racing_cfg.get("gate_in_view_coef", 0.0),
            "reward_mode": racing_cfg.get("reward_mode", "add"),
            "spawn_offset": racing_cfg.get("spawn_offset", 0.75),
            "spawn_pos_noise": racing_cfg.get("spawn_pos_noise", 0.15),
            "spawn_vel_noise": racing_cfg.get("spawn_vel_noise", 0.3),
            "bilateral_progress": racing_cfg.get("bilateral_progress", False),
            "body_frame_obs": racing_cfg.get("body_frame_obs", False),
            "soft_collision": racing_cfg.get("soft_collision", False),
            "soft_collision_penalty": racing_cfg.get("soft_collision_penalty", 5.0),
            "soft_collision_steps": racing_cfg.get("soft_collision_steps", 5_000_000),
            "asymmetric_critic": racing_cfg.get("asymmetric_critic", False),
        }
        race_config = racing_cfg.get("race_config", f"{level}_attitude.toml")
        envs = make_race_envs(
            config=race_config,
            num_envs=racing_cfg["num_envs"],
            jax_device=jax_device,
            torch_device=device,
            coefs=race_coefs,
        )
        return envs, env_type

    from lsy_drone_racing.control.train_rl import make_envs

    envs = make_envs(
        config=f"{level}.toml",
        num_envs=racing_cfg["num_envs"],
        jax_device=jax_device,
        torch_device=device,
        coefs=reward_coefs,
    )
    return envs, env_type


def build_race_env_jax(racing_cfg: dict, jax_device: str):
    from lsy_drone_racing.control.train_race import (
        AppendPrivilegedObs,
        NormalizeRaceActions,
        RaceRewardAndObs,
        RaceStackObs,
    )
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
    from lsy_drone_racing.utils import load_config

    reward_coefs = {
        "n_obs": racing_cfg.get("n_obs", 2),
        "rpy_coef": racing_cfg.get("rpy_coef", 0.06),
        "d_act_th_coef": racing_cfg.get("d_act_th_coef", 0.4),
        "d_act_xy_coef": racing_cfg.get("d_act_xy_coef", 1.0),
        "act_coef": racing_cfg.get("act_coef", 0.02),
        "gate_aware": racing_cfg.get("gate_aware", False),
    }
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
        "progress_coef": racing_cfg.get("progress_coef", 0.0),
        "gate_in_view_coef": racing_cfg.get("gate_in_view_coef", 0.0),
        "reward_mode": racing_cfg.get("reward_mode", "add"),
        "spawn_offset": racing_cfg.get("spawn_offset", 0.75),
        "spawn_pos_noise": racing_cfg.get("spawn_pos_noise", 0.15),
        "spawn_vel_noise": racing_cfg.get("spawn_vel_noise", 0.3),
        "bilateral_progress": racing_cfg.get("bilateral_progress", False),
        "body_frame_obs": racing_cfg.get("body_frame_obs", False),
        "soft_collision": racing_cfg.get("soft_collision", False),
        "soft_collision_penalty": racing_cfg.get("soft_collision_penalty", 5.0),
        "soft_collision_steps": racing_cfg.get("soft_collision_steps", 5_000_000),
        "asymmetric_critic": racing_cfg.get("asymmetric_critic", False),
    }

    level = racing_cfg.get("level", "level0")
    race_config = racing_cfg.get("race_config", f"{level}_attitude.toml")
    cfg = load_config(LSY_ROOT / "config" / race_config)
    n_gates = len(cfg.env.track.gates)
    control_mode = cfg.env.get("control_mode", "attitude")

    env = VecDroneRaceEnv(
        num_envs=racing_cfg["num_envs"],
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=control_mode,
        disturbances=cfg.env.get("disturbances", None),
        randomizations=cfg.env.get("randomizations", None),
        max_episode_steps=race_coefs.get("max_episode_steps", 1500),
        device=jax_device,
    )
    reward_wrapper_kwargs = {
        "gate_bonus": race_coefs.get("gate_bonus", 5.0),
        "proximity_coef": race_coefs.get("proximity_coef", 2.0),
        "speed_coef": race_coefs.get("speed_coef", 0.1),
        "rpy_coef": race_coefs.get("rpy_coef", 0.06),
        "oob_coef": race_coefs.get("oob_coef", 0.0),
        "z_low": race_coefs.get("z_low", 0.0),
        "z_high": race_coefs.get("z_high", 2.0),
        "alt_coef": race_coefs.get("alt_coef", 0.0),
        "survive_coef": race_coefs.get("survive_coef", 0.0),
        "vz_coef": race_coefs.get("vz_coef", 0.0),
        "vz_threshold": race_coefs.get("vz_threshold", 0.5),
        "random_gate_start": race_coefs.get("random_gate_start", False),
        "random_gate_ratio": race_coefs.get("random_gate_ratio", 1.0),
        "progress_coef": race_coefs.get("progress_coef", 0.0),
        "gate_in_view_coef": race_coefs.get("gate_in_view_coef", 0.0),
        "reward_mode": race_coefs.get("reward_mode", "add"),
        "spawn_offset": race_coefs.get("spawn_offset", 0.75),
        "spawn_pos_noise": race_coefs.get("spawn_pos_noise", 0.15),
        "spawn_vel_noise": race_coefs.get("spawn_vel_noise", 0.3),
        "bilateral_progress": race_coefs.get("bilateral_progress", False),
        "body_frame_obs": race_coefs.get("body_frame_obs", False),
        "soft_collision": race_coefs.get("soft_collision", False),
        "soft_collision_penalty": race_coefs.get("soft_collision_penalty", 5.0),
        "soft_collision_steps": race_coefs.get("soft_collision_steps", 5_000_000),
        "asymmetric_critic": race_coefs.get("asymmetric_critic", False),
    }

    env = NormalizeRaceActions(env)
    env = RaceRewardAndObs(env, n_gates=n_gates, **reward_wrapper_kwargs)
    reward_wrapper = env
    env = RaceStackObs(env, n_obs=race_coefs.get("n_obs", 2))
    env = ActionPenalty(
        env,
        act_coef=race_coefs.get("act_coef", 0.02),
        d_act_th_coef=race_coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=race_coefs.get("d_act_xy_coef", 1.0),
    )
    if race_coefs.get("asymmetric_critic", False):
        env = AppendPrivilegedObs(env, reward_wrapper)
    env = FlattenJaxObservation(env)
    return env, "race"


def build_agent(racing_cfg: dict, envs, device: torch.device):
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    hidden_size = racing_cfg.get("hidden_size", 64)

    if racing_cfg.get("asymmetric_critic", False):
        from lsy_drone_racing.control.train_rl import AsymmetricAgent

        actor_obs_dim = getattr(envs, "actor_obs_dim", obs_shape[0])
        agent = AsymmetricAgent(obs_shape, act_shape, actor_obs_dim, hidden_size=hidden_size).to(
            device
        )
    else:
        agent = Agent(obs_shape, act_shape, hidden_size=hidden_size).to(device)

    if racing_cfg.get("max_logstd") is not None:
        agent.max_logstd = racing_cfg["max_logstd"]

    return agent


def _torch_linear_params(layer: torch.nn.Linear) -> tuple[jax.Array, jax.Array]:
    return (
        jp.asarray(layer.weight.detach().cpu().numpy().T),
        jp.asarray(layer.bias.detach().cpu().numpy()),
    )


def build_jax_actor_params(agent: Agent) -> dict[str, jax.Array]:
    layers = [layer for layer in agent.actor_mean if isinstance(layer, torch.nn.Linear)]
    if len(layers) != 3:
        raise ValueError(f"Expected 3 linear layers in actor_mean, got {len(layers)}")
    w1, b1 = _torch_linear_params(layers[0])
    w2, b2 = _torch_linear_params(layers[1])
    w3, b3 = _torch_linear_params(layers[2])
    logstd = jp.asarray(agent.actor_logstd.detach().cpu().numpy())
    max_logstd = getattr(agent, "max_logstd", None)
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
        "logstd": logstd,
        "max_logstd": None if max_logstd is None else jp.asarray(max_logstd, dtype=jp.float32),
    }


@jax.jit
def jax_actor_mean(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    x = jp.tanh(obs @ params["w1"] + params["b1"])
    x = jp.tanh(x @ params["w2"] + params["b2"])
    return jp.tanh(x @ params["w3"] + params["b3"])


@partial(jax.jit, static_argnames=("deterministic",))
def jax_actor_action(
    params: dict[str, jax.Array], obs: jax.Array, key: jax.Array, deterministic: bool = False
) -> tuple[jax.Array, jax.Array]:
    mean = jax_actor_mean(params, obs)
    if deterministic:
        return mean, key
    logstd = params["logstd"]
    if params["max_logstd"] is not None:
        logstd = jp.minimum(logstd, params["max_logstd"])
    std = jp.exp(jp.broadcast_to(logstd, mean.shape))
    key, sample_key = jax.random.split(key)
    action = mean + std * jax.random.normal(sample_key, mean.shape, dtype=mean.dtype)
    return action, key


def benchmark_jax_rollout(
    config: dict, racing_cfg: dict, args: argparse.Namespace, runtime_args: argparse.Namespace
) -> dict:
    if racing_cfg.get("env_type", "trajectory") != "race":
        raise ValueError("JAX rollout backend currently only supports env_type='race'")

    jax_device = runtime_args.jax_device if runtime_args.cuda else "cpu"
    if jax_device == "gpu":
        try:
            jax.devices("gpu")
        except RuntimeError:
            jax_device = "cpu"

    t0 = time.perf_counter()
    envs, env_type = build_race_env_jax(racing_cfg, jax_device=jax_device)
    env_build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    next_obs, _ = envs.reset(seed=args.seed)
    if jax_device == "gpu":
        next_obs = jax.block_until_ready(next_obs)
    reset_s = time.perf_counter() - t1
    next_obs = jp.asarray(next_obs)

    torch_agent = build_agent(racing_cfg, envs, torch.device("cpu"))
    actor_params = build_jax_actor_params(torch_agent)
    key = jax.random.PRNGKey(args.seed)

    forward_time = 0.0
    env_step_time = 0.0
    rollout_steps = args.rollout_steps
    rollout_samples = rollout_steps * runtime_args.num_envs

    for _ in range(rollout_steps):
        start = time.perf_counter()
        action, key = jax_actor_action(actor_params, next_obs, key)
        action = jax.block_until_ready(action)
        forward_time += time.perf_counter() - start

        start = time.perf_counter()
        obs, reward, terminated, truncated, _info = envs.step(action)
        jax.block_until_ready(obs)
        jax.block_until_ready(reward)
        jax.block_until_ready(terminated)
        jax.block_until_ready(truncated)
        env_step_time += time.perf_counter() - start
        next_obs = jp.asarray(obs)

    envs.close()

    total_rollout_time = forward_time + env_step_time
    return {
        "config": config["name"],
        "env_type": env_type,
        "device": "jax",
        "jax_device": jax_device,
        "rollout_backend": "jax",
        "num_envs": runtime_args.num_envs,
        "num_steps": runtime_args.num_steps,
        "batch_size": runtime_args.batch_size,
        "hidden_size": racing_cfg.get("hidden_size", 64),
        "env_build_s": round(env_build_s, 4),
        "first_reset_s": round(reset_s, 4),
        "rollout_steps": rollout_steps,
        "rollout_samples": rollout_samples,
        "rollout_total_s": round(total_rollout_time, 4),
        "rollout_samples_per_s": round(rollout_samples / total_rollout_time, 2),
        "policy_forward_s": round(forward_time, 4),
        "policy_samples_per_s": round(rollout_samples / max(forward_time, 1e-9), 2),
        "env_step_s": round(env_step_time, 4),
        "env_samples_per_s": round(rollout_samples / max(env_step_time, 1e-9), 2),
        "tensor_convert_s": 0.0,
        "ppo_update_epochs": 0,
        "ppo_update_s": 0.0,
        "ppo_update_samples": 0,
        "ppo_update_samples_per_s": 0.0,
    }


def benchmark(config: dict, racing_cfg: dict, args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    runtime_args = build_args(racing_cfg)
    if args.num_envs is not None:
        runtime_args.num_envs = args.num_envs
        runtime_args.batch_size = int(runtime_args.num_envs * runtime_args.num_steps)
        runtime_args.minibatch_size = int(runtime_args.batch_size // runtime_args.num_minibatches)
        racing_cfg = dict(racing_cfg)
        racing_cfg["num_envs"] = args.num_envs

    if args.rollout_backend == "jax":
        return benchmark_jax_rollout(config, racing_cfg, args, runtime_args)

    device = torch.device("cuda" if runtime_args.cuda and torch.cuda.is_available() else "cpu")
    jax_device = runtime_args.jax_device if runtime_args.cuda else "cpu"
    if jax_device == "gpu":
        try:
            jax.devices("gpu")
        except RuntimeError:
            jax_device = "cpu"
    jax_on_gpu = jax_device == "gpu"

    t0 = time.perf_counter()
    envs, env_type = build_env(racing_cfg, device=device, jax_device=jax_device)
    env_build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    next_obs, _ = envs.reset(seed=args.seed)
    reset_s = time.perf_counter() - t1
    next_obs = torch.tensor(_to_np(next_obs), dtype=torch.float32, device=device)
    next_done = torch.zeros(racing_cfg["num_envs"], device=device)

    agent = build_agent(racing_cfg, envs, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=runtime_args.learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape

    obs_buf = torch.zeros((runtime_args.num_steps, runtime_args.num_envs) + obs_shape, device=device)
    actions_buf = torch.zeros(
        (runtime_args.num_steps, runtime_args.num_envs) + act_shape, device=device
    )
    logprobs_buf = torch.zeros((runtime_args.num_steps, runtime_args.num_envs), device=device)
    rewards_buf = torch.zeros((runtime_args.num_steps, runtime_args.num_envs), device=device)
    terminated_buf = torch.zeros((runtime_args.num_steps, runtime_args.num_envs), device=device)
    values_buf = torch.zeros((runtime_args.num_steps, runtime_args.num_envs), device=device)

    forward_time = 0.0
    env_step_time = 0.0
    tensor_time = 0.0
    rollout_steps = args.rollout_steps
    rollout_samples = rollout_steps * runtime_args.num_envs

    for i in range(rollout_steps):
        buf_idx = i % runtime_args.num_steps
        obs_buf[buf_idx] = next_obs

        start = time.perf_counter()
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time += time.perf_counter() - start

        actions_buf[buf_idx] = action
        logprobs_buf[buf_idx] = logprob
        values_buf[buf_idx] = value.flatten()

        start = time.perf_counter()
        obs, reward, terminated, truncated, _info = envs.step(_action_for_env(action, jax_on_gpu))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        env_step_time += time.perf_counter() - start

        start = time.perf_counter()
        rewards_buf[buf_idx] = torch.tensor(_to_np(reward), dtype=torch.float32, device=device)
        terminated_buf[buf_idx] = torch.tensor(
            _to_np(terminated).astype(np.float32), dtype=torch.float32, device=device
        )
        next_obs = torch.tensor(_to_np(obs), dtype=torch.float32, device=device)
        next_done = torch.tensor(
            np.logical_or(_to_np(terminated), _to_np(truncated)).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        tensor_time += time.perf_counter() - start

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = 0
        for t in reversed(range(runtime_args.num_steps)):
            if t == runtime_args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - terminated_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + runtime_args.gamma * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = (
                delta + runtime_args.gamma * runtime_args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values_buf

    b_obs = obs_buf.reshape((-1,) + obs_shape)
    b_logprobs = logprobs_buf.reshape(-1)
    b_actions = actions_buf.reshape((-1,) + act_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values_buf.reshape(-1)
    b_inds = np.arange(runtime_args.batch_size)

    update_time = 0.0
    update_samples = runtime_args.batch_size * args.update_epochs
    for _epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        epoch_start = time.perf_counter()
        for start in range(0, runtime_args.batch_size, runtime_args.minibatch_size):
            end = start + runtime_args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            if runtime_args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - runtime_args.clip_coef, 1 + runtime_args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            if runtime_args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -runtime_args.clip_coef, runtime_args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - runtime_args.ent_coef * entropy_loss + v_loss * runtime_args.vf_coef

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), runtime_args.max_grad_norm)
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_time += time.perf_counter() - epoch_start

    envs.close()

    total_rollout_time = forward_time + env_step_time + tensor_time
    summary = {
        "config": config["name"],
        "env_type": env_type,
        "device": str(device),
        "jax_device": jax_device,
        "num_envs": runtime_args.num_envs,
        "num_steps": runtime_args.num_steps,
        "batch_size": runtime_args.batch_size,
        "hidden_size": racing_cfg.get("hidden_size", 64),
        "rollout_backend": "torch",
        "env_build_s": round(env_build_s, 4),
        "first_reset_s": round(reset_s, 4),
        "rollout_steps": rollout_steps,
        "rollout_samples": rollout_samples,
        "rollout_total_s": round(total_rollout_time, 4),
        "rollout_samples_per_s": round(rollout_samples / total_rollout_time, 2),
        "policy_forward_s": round(forward_time, 4),
        "policy_samples_per_s": round(rollout_samples / max(forward_time, 1e-9), 2),
        "env_step_s": round(env_step_time, 4),
        "env_samples_per_s": round(rollout_samples / max(env_step_time, 1e-9), 2),
        "tensor_convert_s": round(tensor_time, 4),
        "ppo_update_epochs": args.update_epochs,
        "ppo_update_s": round(update_time, 4),
        "ppo_update_samples": update_samples,
        "ppo_update_samples_per_s": round(update_samples / max(update_time, 1e-9), 2),
    }
    return summary


def main() -> None:
    args = parse_args()
    config, racing_cfg = load_experiment(args.config)
    summary = benchmark(config, racing_cfg, args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
