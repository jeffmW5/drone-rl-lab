"""Deployable normalized-action shaping shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionGovernorConfig:
    slew_limits: tuple[float, float, float, float] = (0.08, 0.12, 0.12, 0.15)
    upward_brake_start_mps: float = 1.5
    upward_brake_gain: float = 0.30


def govern_action(
    requested_action: torch.Tensor,
    previous_executed_action: torch.Tensor,
    live_base_observation: torch.Tensor,
    config: ActionGovernorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape normalized actions using only deployable observation fields."""

    body_velocity_mps = live_base_observation[:, :3] * 8.0
    gravity_body = live_base_observation[:, 3:6]
    upward_speed_mps = -(body_velocity_mps * gravity_body).sum(dim=1)

    shaped = requested_action.clone()
    upward_excess = (
        upward_speed_mps - config.upward_brake_start_mps
    ).clamp(min=0.0)
    shaped[:, 0] -= config.upward_brake_gain * upward_excess
    shaped = shaped.clamp(-1.0, 1.0)

    limits = torch.tensor(
        config.slew_limits,
        device=shaped.device,
        dtype=shaped.dtype,
    )
    executed = previous_executed_action + (
        shaped - previous_executed_action
    ).clamp(-limits, limits)
    executed = executed.clamp(-1.0, 1.0)
    intervention = (executed - requested_action).abs().amax(dim=1)
    return executed, intervention
