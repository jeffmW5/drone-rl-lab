#!/usr/bin/env python3
"""Deterministic checkpoint evaluation with time-series flight telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.model import ActorCritic
from train_ai_gp import _build_env_config, load_config


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _as_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


@torch.no_grad()
def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    *,
    episode_count: int,
    trajectory_count: int,
    randomization: bool,
    seed: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    section = config["ai_gp"]
    env_config = _build_env_config(section, evaluation=True)
    env_config.randomization = randomization
    if seed is not None:
        env_config.seed = seed
    if env_config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint evaluation requested without CUDA")

    env = AIGPVectorEnv(env_config)
    env.set_curriculum(1.0)
    checkpoint = torch.load(
        checkpoint_path, map_location=env.device, weights_only=False
    )
    hidden_sizes = tuple(
        checkpoint.get("metadata", {}).get(
            "hidden_sizes", section.get("hidden_sizes", [128, 128])
        )
    )
    model = ActorCritic(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        actor_observation_dim=env.actor_observation_dim,
        hidden_sizes=hidden_sizes,
    ).to(env.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    observation, _ = env.reset()
    max_altitude = env.position[:, 2].clone()
    min_altitude = env.position[:, 2].clone()
    max_abs_vertical_speed = env.velocity[:, 2].abs().clone()
    upward_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    longest_upward_run = torch.zeros_like(upward_run)
    episode_steps = torch.zeros_like(upward_run)

    episodes: list[dict[str, Any]] = []
    gate_events: list[dict[str, Any]] = []
    pending_gate_events: dict[int, list[dict[str, Any]]] = {
        env_id: [] for env_id in range(env.num_envs)
    }
    tracked_envs = set(range(min(trajectory_count, env.num_envs)))
    active_trajectories = set(tracked_envs)
    trajectories: dict[str, dict[str, Any]] = {
        str(env_id): {"env_id": env_id, "samples": [], "gate_passages": []}
        for env_id in tracked_envs
    }

    while len(episodes) < episode_count:
        action, _, _, _ = model.get_action_and_value(
            observation, deterministic=True
        )
        observation, _, _, _, info = env.step(action)
        episode_steps += 1

        position = info["position"]
        velocity = info["velocity"]
        max_altitude = torch.maximum(max_altitude, position[:, 2])
        min_altitude = torch.minimum(min_altitude, position[:, 2])
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
                "attitude_rad": _as_list(info["attitude"][env_id]),
                "applied_action": _as_list(info["applied_action"][env_id]),
                "active_gate_index": int(info["active_gate_index"][env_id]),
                "gates_passed": int(info["gates_passed"][env_id]),
                "passed_gate": bool(info["passed_gate"][env_id]),
                "collision": bool(info["collision"][env_id]),
                "out_of_bounds": bool(info["out_of_bounds"][env_id]),
                "done": bool(info["done"][env_id]),
            }
            trajectories[str(env_id)]["samples"].append(sample)

        passed_ids = torch.where(info["passed_gate"])[0].detach().cpu().tolist()
        for env_id in passed_ids:
            gate_index = int(info["active_gate_index"][env_id])
            lateral_offset = float(info["gate_lateral_offset"][env_id])
            vertical_offset = float(info["gate_vertical_offset"][env_id])
            plane_offset = float(info["gate_plane_offset"][env_id])
            event = {
                "env_id": env_id,
                "gate_index": gate_index,
                "step": int(episode_steps[env_id]),
                "time_s": float(episode_steps[env_id]) * env_config.dt,
                "position_m": _as_list(position[env_id]),
                "lateral_offset_m": lateral_offset,
                "vertical_offset_m": vertical_offset,
                "plane_offset_m": plane_offset,
                "lateral_margin_m": env_config.gate_half_width_m
                - abs(lateral_offset),
                "vertical_margin_m": env_config.gate_half_height_m
                - abs(vertical_offset),
            }
            pending_gate_events[env_id].append(event)
            if env_id in active_trajectories:
                trajectories[str(env_id)]["gate_passages"].append(event)

        done_ids = torch.where(info["done"])[0].detach().cpu().tolist()
        remaining = episode_count - len(episodes)
        accepted_done_ids = done_ids[:remaining]
        for env_id in accepted_done_ids:
            max_z = float(max_altitude[env_id])
            max_vz = float(max_abs_vertical_speed[env_id])
            longest_upward_s = (
                float(longest_upward_run[env_id]) * env_config.dt
            )
            episode_index = len(episodes)
            for event in pending_gate_events[env_id]:
                event["episode_index"] = episode_index
                gate_events.append(event)
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
                    "distance_reduction_m": float(
                        info["distance_reduction"][env_id]
                    ),
                    "min_altitude_m": float(min_altitude[env_id]),
                    "max_altitude_m": max_z,
                    "max_abs_vertical_speed_mps": max_vz,
                    "longest_sustained_upward_s": longest_upward_s,
                    "vertical_runaway": bool(
                        max_z >= 0.95 * env_config.max_altitude_m
                        or longest_upward_s >= 1.0
                    ),
                    "final_position_m": _as_list(position[env_id]),
                    "final_velocity_mps": _as_list(velocity[env_id]),
                }
            )
            active_trajectories.discard(env_id)

        if done_ids:
            for env_id in done_ids:
                pending_gate_events[env_id] = []
            reset_ids = torch.tensor(done_ids, device=env.device)
            max_altitude[reset_ids] = env.position[reset_ids, 2]
            min_altitude[reset_ids] = env.position[reset_ids, 2]
            max_abs_vertical_speed[reset_ids] = env.velocity[reset_ids, 2].abs()
            upward_run[reset_ids] = 0
            longest_upward_run[reset_ids] = 0
            episode_steps[reset_ids] = 0

    gate_counts = [episode["gates_passed"] for episode in episodes]
    max_altitudes = [episode["max_altitude_m"] for episode in episodes]
    max_vertical_speeds = [
        episode["max_abs_vertical_speed_mps"] for episode in episodes
    ]
    crossing_margins = [
        min(event["lateral_margin_m"], event["vertical_margin_m"])
        for event in gate_events
    ]
    return {
        "experiment": config["name"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "environment_randomization": randomization,
        "evaluation_seed": env_config.seed,
        "episodes": len(episodes),
        "deterministic_summary": {
            "gate0_passage_rate": sum(count >= 1 for count in gate_counts)
            / len(episodes),
            "mean_gates": sum(gate_counts) / len(episodes),
            "success_rate": sum(episode["success"] for episode in episodes)
            / len(episodes),
            "collision_rate": sum(episode["collision"] for episode in episodes)
            / len(episodes),
            "out_of_bounds_rate": sum(
                episode["out_of_bounds"] for episode in episodes
            )
            / len(episodes),
            "vertical_runaway_rate": sum(
                episode["vertical_runaway"] for episode in episodes
            )
            / len(episodes),
            "max_altitude_m_mean": sum(max_altitudes) / len(episodes),
            "max_altitude_m_p95": _percentile(max_altitudes, 0.95),
            "max_altitude_m_max": max(max_altitudes),
            "max_abs_vertical_speed_mps_p95": _percentile(
                max_vertical_speeds, 0.95
            ),
            "max_abs_vertical_speed_mps_max": max(max_vertical_speeds),
            "gate_crossing_count": len(gate_events),
            "gate_crossing_min_margin_m": min(crossing_margins)
            if crossing_margins
            else None,
        },
        "episode_summaries": episodes,
        "gate_passages": gate_events,
        "trajectories": list(trajectories.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--randomization", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    report = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        episode_count=args.episodes,
        trajectory_count=args.trajectories,
        randomization=args.randomization,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["deterministic_summary"], indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
