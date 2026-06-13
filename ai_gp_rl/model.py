"""Torch actor-critic used by AI-GP PPO training and deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions.normal import Normal


def _layer_init(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, output_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((_layer_init(nn.Linear(previous, width), 2**0.5), nn.LeakyReLU(0.2)))
        previous = width
    layers.append(_layer_init(nn.Linear(previous, output_dim), output_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Asymmetric PPO model.

    The actor receives only deployable features. The critic receives the actor
    features plus privileged state appended by the surrogate environment.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_observation_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor_observation_dim = actor_observation_dim
        self.hidden_sizes = tuple(hidden_sizes)

        self.actor_mean = _mlp(actor_observation_dim, hidden_sizes, action_dim, 0.01)
        self.critic = _mlp(observation_dim, hidden_sizes, 1, 1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_observation = observation[..., : self.actor_observation_dim]
        action_mean = self.actor_mean(actor_observation)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(-1),
            distribution.entropy().sum(-1),
            self.critic(observation),
        )

    @torch.no_grad()
    def deterministic_action(self, actor_observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(actor_observation[..., : self.actor_observation_dim]))

    def actor_step(
        self,
        actor_observation: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del hidden_state
        return self.actor_mean(
            actor_observation[..., : self.actor_observation_dim]
        ), None


class RecurrentStudentPolicy(nn.Module):
    """GRU student used only with deployable per-frame observations."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_observation_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("recurrent student requires at least one hidden size")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor_observation_dim = actor_observation_dim
        self.hidden_sizes = tuple(hidden_sizes)
        recurrent_width = self.hidden_sizes[0]
        self.input_encoder = nn.Sequential(
            _layer_init(nn.Linear(actor_observation_dim, recurrent_width), 2**0.5),
            nn.LeakyReLU(0.2),
        )
        self.recurrent = nn.GRUCell(recurrent_width, recurrent_width)
        head_layers: list[nn.Module] = []
        previous = recurrent_width
        for width in self.hidden_sizes[1:]:
            head_layers.extend(
                (
                    _layer_init(nn.Linear(previous, width), 2**0.5),
                    nn.LeakyReLU(0.2),
                )
            )
            previous = width
        head_layers.append(_layer_init(nn.Linear(previous, action_dim), 0.01))
        self.action_head = nn.Sequential(*head_layers)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.hidden_sizes[0]),
            device=device,
            dtype=next(self.parameters()).dtype,
        )

    def forward_step(
        self,
        actor_observation: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.input_encoder(
            actor_observation[..., : self.actor_observation_dim]
        )
        next_hidden = self.recurrent(encoded, hidden_state)
        return self.action_head(next_hidden), next_hidden

    def actor_step(
        self,
        actor_observation: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            raise ValueError("recurrent student requires hidden state")
        return self.forward_step(actor_observation, hidden_state)


def build_policy_from_metadata(
    metadata: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> ActorCritic | RecurrentStudentPolicy:
    architecture = str(metadata.get("policy_architecture", "mlp"))
    model_kwargs = {
        "observation_dim": int(metadata["observation_dim"]),
        "action_dim": int(metadata["action_dim"]),
        "actor_observation_dim": int(metadata["actor_observation_dim"]),
        "hidden_sizes": tuple(metadata["hidden_sizes"]),
    }
    if architecture == "mlp":
        model: ActorCritic | RecurrentStudentPolicy = ActorCritic(**model_kwargs)
    elif architecture == "gru":
        model = RecurrentStudentPolicy(**model_kwargs)
    else:
        raise ValueError(f"unsupported policy architecture: {architecture}")
    return model.to(device)
