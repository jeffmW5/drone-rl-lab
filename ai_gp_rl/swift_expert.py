"""Swift-style full-course geometric teacher for AI-GP gate flight."""

from __future__ import annotations

from math import pi
from typing import Protocol

import torch

from .contract import ACTION_DIM


class SwiftTeacherEnv(Protocol):
    """Subset of AIGPVectorEnv used by the geometric teacher."""

    num_envs: int
    device: torch.device
    position: torch.Tensor
    velocity: torch.Tensor
    attitude: torch.Tensor
    gate_index: torch.Tensor
    env_gate_positions: torch.Tensor
    gate_normals: torch.Tensor
    gate_lateral: torch.Tensor
    gate_vertical: torch.Tensor
    command_rate_limits: torch.Tensor
    rate_response_gain: torch.Tensor
    mass_scale: torch.Tensor
    thrust_scale: torch.Tensor
    config: object


def geometric_gate_teacher_action(env: SwiftTeacherEnv) -> torch.Tensor:
    """Return normalized actions for fast full-course active-gate flight.

    The teacher is a privileged geometric policy: it observes exact active-gate
    geometry and state, regulates cross-track/altitude error in the gate frame,
    and maps desired acceleration to thrust plus body-rate commands. It is used
    to bootstrap a neural teacher/student with behavior cloning and DAgger-style
    rollouts before PPO fine-tuning.
    """

    env_ids = torch.arange(env.num_envs, device=env.device)
    gate_index = env.gate_index
    gate_position = env.env_gate_positions[env_ids, gate_index]
    normal = env.gate_normals[gate_index]
    lateral = env.gate_lateral[gate_index]
    vertical = env.gate_vertical[gate_index]

    offset = env.position - gate_position
    plane_offset = (offset * normal).sum(1)
    lateral_offset = (offset * lateral).sum(1)
    vertical_offset = (offset * vertical).sum(1)

    forward_speed = (env.velocity * normal).sum(1)
    lateral_speed = (env.velocity * lateral).sum(1)
    vertical_speed = (env.velocity * vertical).sum(1)

    distance_to_plane = (-plane_offset).clamp(0.0, 28.0)
    cross_track_error = lateral_offset.abs() + vertical_offset.abs()
    speed_scale = (1.0 / (1.0 + 0.16 * cross_track_error)).clamp(0.45, 1.0)
    target_forward_speed = (4.0 + 0.50 * distance_to_plane).clamp(3.5, 13.5)
    target_forward_speed = target_forward_speed * speed_scale
    target_lateral_speed = (-1.8 * lateral_offset).clamp(-4.5, 4.5)
    target_vertical_speed = (-1.6 * vertical_offset).clamp(-4.0, 4.0)

    forward_accel = (target_forward_speed - forward_speed).clamp(-7.0, 7.0)
    lateral_accel = (2.0 * (target_lateral_speed - lateral_speed)).clamp(-8.0, 8.0)
    vertical_accel = (2.2 * (target_vertical_speed - vertical_speed)).clamp(-8.0, 8.0)
    support_accel = (9.81 + vertical_accel).clamp(4.0, 19.0)

    base_pitch = torch.as_tensor(
        getattr(env.config, "base_pitch_offset_rad", 0.0),
        device=env.device,
        dtype=env.position.dtype,
    )
    desired_roll = torch.atan2(-lateral_accel, support_accel).clamp(-0.65, 0.65)
    desired_pitch = (
        base_pitch - torch.atan2(forward_accel, support_accel)
    ).clamp(-0.85, 0.75)
    desired_yaw = torch.atan2(normal[:, 1], normal[:, 0])

    attitude_error = torch.stack(
        (
            desired_roll - env.attitude[:, 0],
            desired_pitch - env.attitude[:, 1],
            _wrap_pi(desired_yaw - env.attitude[:, 2]),
        ),
        dim=1,
    )
    desired_rate = torch.stack(
        (
            2.0 * attitude_error[:, 0],
            1.8 * attitude_error[:, 1],
            1.6 * attitude_error[:, 2],
        ),
        dim=1,
    )
    max_response_rate = env.command_rate_limits * env.rate_response_gain
    rate_action = desired_rate / max_response_rate.clamp(min=1e-4)

    horizontal_accel = torch.sqrt(
        forward_accel.square() + lateral_accel.square()
    )
    collective_accel = torch.sqrt(
        support_accel.square() + horizontal_accel.square()
    )
    collective_action = _normalized_collective_action(env, collective_accel)

    action = torch.cat((collective_action.unsqueeze(1), rate_action), dim=1)
    if action.shape != (env.num_envs, ACTION_DIM):
        raise RuntimeError("Swift teacher action contract mismatch")
    return action.clamp(-0.98, 0.98)


def raw_geometric_gate_teacher_action(env: SwiftTeacherEnv) -> torch.Tensor:
    """Return raw actions suitable for ``AIGPVectorEnv.step``."""

    return torch.atanh(
        geometric_gate_teacher_action(env).clamp(-0.999, 0.999)
    )


def _normalized_collective_action(
    env: SwiftTeacherEnv,
    collective_accel: torch.Tensor,
) -> torch.Tensor:
    dynamics_model = getattr(env.config, "dynamics_model", "legacy_collective")
    if dynamics_model == "measured_ai_gp_v1":
        adjusted = collective_accel * env.mass_scale / env.thrust_scale.clamp(min=1e-4)
        thrust_command = (
            adjusted - float(getattr(env.config, "thrust_acceleration_bias_mps2"))
        ) / max(float(getattr(env.config, "thrust_acceleration_gain_mps2")), 1e-4)
        center = float(getattr(env.config, "thrust_command_center"))
        up_span = float(getattr(env.config, "thrust_command_span_up"))
        down_span = float(getattr(env.config, "thrust_command_span_down"))
        span = torch.where(
            thrust_command >= center,
            torch.full_like(thrust_command, up_span),
            torch.full_like(thrust_command, down_span),
        )
        return (thrust_command - center) / span.clamp(min=1e-4)

    adjusted_g = collective_accel * env.mass_scale / env.thrust_scale.clamp(min=1e-4)
    return (adjusted_g / 9.81 - 1.0) / max(
        float(getattr(env.config, "collective_range_g", 0.8)),
        1e-4,
    )


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + pi, 2.0 * pi) - pi
