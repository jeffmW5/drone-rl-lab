"""Hybrid policy/teacher helpers for AI-GP training and evaluation."""

from __future__ import annotations

import torch

from .env import AIGPVectorEnv


def parse_gate_indices(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip().lower() in {"", "all", "*"}:
        return None
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def teacher_takeover_mask(
    env: AIGPVectorEnv,
    *,
    mode: str,
    takeover_distance_m: float,
    lateral_threshold_m: float = 0.40,
    vertical_threshold_m: float = 0.40,
    gate_indices: set[int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    distance_to_plane = (-plane_offset).clamp(min=0.0)

    if gate_indices is None:
        gate_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        gate_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for gate in gate_indices:
            gate_mask |= gate_index == gate

    approaching = plane_offset < 0.0
    near_gate = approaching & (distance_to_plane <= takeover_distance_m)
    off_center = (
        (lateral_offset.abs() >= lateral_threshold_m)
        | (vertical_offset.abs() >= vertical_threshold_m)
    )

    if mode == "policy":
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif mode == "teacher":
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif mode == "near_gate_teacher":
        mask = near_gate
    elif mode == "off_center_near_gate_teacher":
        mask = near_gate & off_center
    else:
        raise ValueError(f"unsupported hybrid mode: {mode}")

    return mask & gate_mask, {
        "plane_offset_m": plane_offset,
        "lateral_offset_m": lateral_offset,
        "vertical_offset_m": vertical_offset,
        "distance_to_plane_m": distance_to_plane,
    }
