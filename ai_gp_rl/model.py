"""Torch actor-critic used by AI-GP PPO training and deployment."""

from __future__ import annotations

from collections.abc import Sequence

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
