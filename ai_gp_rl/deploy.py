"""Checkpoint loader for shadow-mode and calibrated live deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .contract import ACTOR_OBS_DIM, LivePolicyFeatures, build_actor_observation
from .model import ActorCritic


class AIGPPolicyRunner:
    """Loads a training checkpoint and emits normalized deterministic actions."""

    def __init__(self, model: ActorCritic, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device
        self.previous_action = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> "AIGPPolicyRunner":
        target_device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=target_device, weights_only=False
        )
        metadata = checkpoint["metadata"]
        model = ActorCritic(
            observation_dim=int(metadata["observation_dim"]),
            action_dim=int(metadata["action_dim"]),
            actor_observation_dim=int(metadata["actor_observation_dim"]),
            hidden_sizes=tuple(metadata["hidden_sizes"]),
        ).to(target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if model.actor_observation_dim != ACTOR_OBS_DIM:
            raise ValueError(
                "checkpoint actor observation contract does not match this runtime"
            )
        return cls(model, target_device)

    @torch.no_grad()
    def act(self, features: LivePolicyFeatures) -> tuple[float, float, float, float]:
        observation = torch.tensor(
            build_actor_observation(features),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        action = self.model.deterministic_action(observation).squeeze(0).cpu().tolist()
        self.previous_action = tuple(float(value) for value in action)
        return self.previous_action

    @torch.no_grad()
    def act_from_observation(self, observation: Sequence[float]) -> tuple[float, ...]:
        if len(observation) != ACTOR_OBS_DIM:
            raise ValueError(
                f"expected {ACTOR_OBS_DIM} actor features, received {len(observation)}"
            )
        tensor = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return tuple(
            float(value)
            for value in self.model.deterministic_action(tensor).squeeze(0).cpu().tolist()
        )
