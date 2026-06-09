#!/usr/bin/env python3
"""Short CUDA smoke test for the AI-GP environment and PPO model."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.model import ActorCritic
from train_ai_gp import _build_env_config, load_config


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke_ai_gp.py configs/ai_gp_NNN.yaml")

    config = load_config(sys.argv[1])
    section = copy.deepcopy(config["ai_gp"])
    section["num_envs"] = min(int(section.get("num_envs", 4096)), 256)
    section["device"] = "cuda"
    section["allow_cpu"] = False
    section.setdefault("env", {})
    section["env"]["num_envs"] = section["num_envs"]
    section["env"]["device"] = "cuda"
    section["env"]["max_episode_steps"] = min(
        int(section["env"].get("max_episode_steps", 500)), 50
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    env = AIGPVectorEnv(_build_env_config(section))
    model = ActorCritic(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        actor_observation_dim=env.actor_observation_dim,
        hidden_sizes=tuple(section.get("hidden_sizes", [128, 128])),
    ).to(env.device)
    observation, _ = env.reset()
    total_reward = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for _ in range(16):
            action, _, _, _ = model.get_action_and_value(observation)
            observation, reward, _, _, _ = env.step(action)
            total_reward += reward

    _, logprob, entropy, value = model.get_action_and_value(observation.detach())
    loss = (
        -logprob.mean()
        - 0.001 * entropy.mean()
        + value.square().mean()
    )
    loss.backward()

    tensors = (observation, total_reward, loss)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise SystemExit("non-finite tensor detected")
    print(
        "smoke_ok "
        f"device={torch.cuda.get_device_name(0)} "
        f"envs={env.num_envs} "
        f"obs={tuple(observation.shape)} "
        f"mean_reward={float(total_reward.mean()):.3f}"
    )


if __name__ == "__main__":
    main()
