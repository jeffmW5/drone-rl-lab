#!/usr/bin/env python3
"""Evaluate the geometric Swift teacher in the AI-GP surrogate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.swift_expert import raw_geometric_gate_teacher_action
from scripts.evaluate_ai_gp_checkpoint import _as_list, _percentile
from train_ai_gp import _build_env_config, load_config


def parse_env_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"environment override must be key=value: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("environment override key cannot be empty")
        overrides[key] = yaml.safe_load(raw)
    return overrides


@torch.no_grad()
def evaluate_teacher(
    config_path: Path,
    *,
    output_path: Path,
    episode_count: int,
    trajectory_count: int,
    randomization: bool,
    seed: int | None,
    env_overrides: dict[str, Any],
) -> dict[str, Any]:
    config = load_config(config_path)
    section = config["ai_gp"]
    env_config = _build_env_config(section, evaluation=True)
    env_config.randomization = randomization
    if seed is not None:
        env_config.seed = seed
    for key, value in env_overrides.items():
        if key not in env_config.__dataclass_fields__:
            raise ValueError(f"unknown AI-GP environment override: {key}")
        setattr(env_config, key, value)
    if env_config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA teacher evaluation requested without CUDA")

    env = AIGPVectorEnv(env_config)
    env.set_curriculum(1.0)
    _, _ = env.reset()
    max_altitude = env.position[:, 2].clone()
    max_abs_vertical_speed = env.velocity[:, 2].abs().clone()
    upward_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    longest_upward_run = torch.zeros_like(upward_run)
    episode_missed_gate = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    episode_steps = torch.zeros_like(upward_run)

    episodes: list[dict[str, Any]] = []
    gate_events: list[dict[str, Any]] = []
    tracked_envs = set(range(min(trajectory_count, env.num_envs)))
    active_trajectories = set(tracked_envs)
    trajectories: dict[str, dict[str, Any]] = {
        str(env_id): {"env_id": env_id, "samples": [], "gate_passages": []}
        for env_id in tracked_envs
    }
    pending_gate_events: dict[int, list[dict[str, Any]]] = {
        env_id: [] for env_id in range(env.num_envs)
    }

    while len(episodes) < episode_count:
        _, _, _, _, info = env.step(raw_geometric_gate_teacher_action(env))
        episode_missed_gate |= info["missed_gate"]
        episode_steps += 1
        position = info["position"]
        velocity = info["velocity"]
        max_altitude = torch.maximum(max_altitude, position[:, 2])
        max_abs_vertical_speed = torch.maximum(
            max_abs_vertical_speed, velocity[:, 2].abs()
        )
        sustained_upward = (velocity[:, 2] > 2.0) & (position[:, 2] > 1.5)
        upward_run = torch.where(
            sustained_upward, upward_run + 1, torch.zeros_like(upward_run)
        )
        longest_upward_run = torch.maximum(longest_upward_run, upward_run)

        for env_id in sorted(active_trajectories):
            sample = {
                "step": int(episode_steps[env_id]),
                "time_s": float(episode_steps[env_id]) * env_config.dt,
                "position_m": _as_list(position[env_id]),
                "velocity_mps": _as_list(velocity[env_id]),
                "active_gate_index": int(info["active_gate_index"][env_id]),
                "gates_passed": int(info["gates_passed"][env_id]),
                "passed_gate": bool(info["passed_gate"][env_id]),
                "missed_gate": bool(info["missed_gate"][env_id]),
                "collision": bool(info["collision"][env_id]),
                "out_of_bounds": bool(info["out_of_bounds"][env_id]),
                "done": bool(info["done"][env_id]),
            }
            trajectories[str(env_id)]["samples"].append(sample)

        for env_id in torch.where(info["passed_gate"])[0].detach().cpu().tolist():
            event = {
                "env_id": env_id,
                "gate_index": int(info["active_gate_index"][env_id]),
                "step": int(episode_steps[env_id]),
                "time_s": float(episode_steps[env_id]) * env_config.dt,
                "position_m": _as_list(position[env_id]),
                "lateral_offset_m": float(info["gate_lateral_offset"][env_id]),
                "vertical_offset_m": float(info["gate_vertical_offset"][env_id]),
                "plane_offset_m": float(info["gate_plane_offset"][env_id]),
                "lateral_margin_m": env_config.gate_half_width_m
                - abs(float(info["gate_lateral_offset"][env_id])),
                "vertical_margin_m": env_config.gate_half_height_m
                - abs(float(info["gate_vertical_offset"][env_id])),
            }
            pending_gate_events[env_id].append(event)
            if env_id in active_trajectories:
                trajectories[str(env_id)]["gate_passages"].append(event)

        done_ids = torch.where(info["done"])[0].detach().cpu().tolist()
        accepted_done_ids = done_ids[: episode_count - len(episodes)]
        for env_id in accepted_done_ids:
            episode_index = len(episodes)
            for event in pending_gate_events[env_id]:
                event["episode_index"] = episode_index
                gate_events.append(event)
            max_z = float(max_altitude[env_id])
            max_vz = float(max_abs_vertical_speed[env_id])
            longest_upward_s = float(longest_upward_run[env_id]) * env_config.dt
            episodes.append(
                {
                    "episode_index": episode_index,
                    "env_id": env_id,
                    "steps": int(episode_steps[env_id]),
                    "return": float(info["episode_return"][env_id]),
                    "gates_passed": int(info["gates_passed"][env_id]),
                    "success": bool(info["success"][env_id]),
                    "collision": bool(info["collision"][env_id]),
                    "out_of_bounds": bool(info["out_of_bounds"][env_id]),
                    "missed_gate": bool(episode_missed_gate[env_id]),
                    "max_altitude_m": max_z,
                    "max_abs_vertical_speed_mps": max_vz,
                    "vertical_runaway": bool(
                        max_z >= 0.95 * env_config.max_altitude_m
                        or max_vz >= env_config.vertical_runaway_speed_mps
                        or longest_upward_s >= 1.0
                    ),
                    "final_position_m": _as_list(position[env_id]),
                    "final_velocity_mps": _as_list(velocity[env_id]),
                }
            )
            active_trajectories.discard(env_id)

        if done_ids:
            reset_ids = torch.tensor(done_ids, device=env.device)
            for env_id in done_ids:
                pending_gate_events[env_id] = []
            max_altitude[reset_ids] = env.position[reset_ids, 2]
            max_abs_vertical_speed[reset_ids] = env.velocity[reset_ids, 2].abs()
            upward_run[reset_ids] = 0
            longest_upward_run[reset_ids] = 0
            episode_missed_gate[reset_ids] = False
            episode_steps[reset_ids] = 0

    gate_counts = [episode["gates_passed"] for episode in episodes]
    crossing_margins = [
        min(event["lateral_margin_m"], event["vertical_margin_m"])
        for event in gate_events
    ]
    summary = {
        "gate0_passage_rate": sum(count >= 1 for count in gate_counts)
        / len(episodes),
        "mean_gates": sum(gate_counts) / len(episodes),
        "success_rate": sum(episode["success"] for episode in episodes)
        / len(episodes),
        "collision_rate": sum(episode["collision"] for episode in episodes)
        / len(episodes),
        "out_of_bounds_rate": sum(episode["out_of_bounds"] for episode in episodes)
        / len(episodes),
        "missed_gate_rate": sum(episode["missed_gate"] for episode in episodes)
        / len(episodes),
        "vertical_runaway_rate": sum(episode["vertical_runaway"] for episode in episodes)
        / len(episodes),
        "gate_crossing_count": len(gate_events),
        "gate_crossing_min_margin_m": min(crossing_margins)
        if crossing_margins
        else float("nan"),
        "gate_crossing_p05_margin_m": _percentile(crossing_margins, 0.05),
    }
    result = {
        "experiment": config["name"],
        "policy": "geometric_gate_teacher_v1",
        "environment_randomization": randomization,
        "evaluation_seed": env_config.seed,
        "env_overrides": env_overrides,
        "episodes": len(episodes),
        "deterministic_summary": summary,
        "episode_summaries": episodes,
        "gate_passages": gate_events,
        "trajectories": list(trajectories.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved={output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--randomization", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--env-override",
        action="append",
        default=[],
        help="Environment override as key=value, parsed as YAML.",
    )
    args = parser.parse_args()
    evaluate_teacher(
        args.config,
        output_path=args.output,
        episode_count=args.episodes,
        trajectory_count=args.trajectories,
        randomization=args.randomization,
        seed=args.seed,
        env_overrides=parse_env_overrides(args.env_override),
    )


if __name__ == "__main__":
    main()
