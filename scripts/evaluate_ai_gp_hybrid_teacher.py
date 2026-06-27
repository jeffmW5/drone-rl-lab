#!/usr/bin/env python3
"""Evaluate policy/teacher hybrids in the AI-GP surrogate."""

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

from ai_gp_rl.action_governor import ActionGovernorConfig, govern_action
from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.hybrid_teacher import parse_gate_indices, teacher_takeover_mask
from ai_gp_rl.model import RecurrentStudentPolicy, build_policy_from_metadata
from ai_gp_rl.swift_expert import geometric_gate_teacher_action
from scripts.evaluate_ai_gp_checkpoint import _as_list, _percentile
from train_ai_gp import _build_env_config, load_config


@torch.no_grad()
def evaluate_hybrid(
    config_path: Path,
    checkpoint_path: Path,
    *,
    output_path: Path,
    episode_count: int,
    trajectory_count: int,
    randomization: bool,
    seed: int | None,
    mode: str,
    takeover_distance_m: float,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
    gate_indices: set[int] | None,
    device_override: str | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    section = config["ai_gp"]
    env_config = _build_env_config(section, evaluation=True)
    env_config.randomization = randomization
    if seed is not None:
        env_config.seed = seed
    if device_override is not None:
        env_config.device = device_override
    if env_config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA hybrid evaluation requested without CUDA")

    env = AIGPVectorEnv(env_config)
    env.set_curriculum(1.0)
    checkpoint = torch.load(
        checkpoint_path, map_location=env.device, weights_only=False
    )
    metadata = checkpoint.get("metadata", {})
    model = build_policy_from_metadata(metadata, device=env.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    governor_section = metadata.get("action_governor")
    governor_config = (
        ActionGovernorConfig(
            slew_limits=tuple(
                float(value) for value in governor_section["slew_limits"]
            ),
            upward_brake_start_mps=float(
                governor_section["upward_brake_start_mps"]
            ),
            upward_brake_gain=float(governor_section["upward_brake_gain"]),
        )
        if governor_section
        else None
    )

    observation, _ = env.reset()
    hidden_state = (
        model.initial_state(env.num_envs, device=env.device)
        if isinstance(model, RecurrentStudentPolicy)
        else None
    )
    max_altitude = env.position[:, 2].clone()
    min_altitude = env.position[:, 2].clone()
    max_abs_vertical_speed = env.velocity[:, 2].abs().clone()
    upward_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    longest_upward_run = torch.zeros_like(upward_run)
    episode_missed_gate = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    episode_teacher_samples = torch.zeros_like(upward_run)
    episode_steps = torch.zeros_like(upward_run)

    intervention_samples = 0
    total_samples = 0
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
        policy_action, next_hidden_state = model.actor_step(
            observation[:, : env.actor_observation_dim], hidden_state
        )
        teacher_action = torch.atanh(
            geometric_gate_teacher_action(env).clamp(-0.999, 0.999)
        )
        teacher_mask, gate_frame = teacher_takeover_mask(
            env,
            mode=mode,
            takeover_distance_m=takeover_distance_m,
            lateral_threshold_m=lateral_threshold_m,
            vertical_threshold_m=vertical_threshold_m,
            gate_indices=gate_indices,
        )
        action = torch.where(teacher_mask.unsqueeze(1), teacher_action, policy_action)
        if governor_config is not None:
            normalized_action, _ = govern_action(
                torch.tanh(action),
                env.previous_action,
                env.live_observation_history[:, -1],
                governor_config,
            )
            action = torch.atanh(normalized_action.clamp(-0.999, 0.999))

        observation, _, _, _, info = env.step(action)
        episode_missed_gate |= info["missed_gate"]
        episode_teacher_samples += teacher_mask.long()
        intervention_samples += int(teacher_mask.sum())
        total_samples += env.num_envs
        if next_hidden_state is not None:
            hidden_state = next_hidden_state * (~info["done"]).unsqueeze(1)
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
                "teacher_takeover": bool(teacher_mask[env_id]),
                "teacher_takeover_samples": int(episode_teacher_samples[env_id]),
                "active_gate_index": int(info["active_gate_index"][env_id]),
                "gates_passed": int(info["gates_passed"][env_id]),
                "pre_step_gate_frame": {
                    "plane_offset_m": float(gate_frame["plane_offset_m"][env_id]),
                    "lateral_offset_m": float(
                        gate_frame["lateral_offset_m"][env_id]
                    ),
                    "vertical_offset_m": float(
                        gate_frame["vertical_offset_m"][env_id]
                    ),
                    "distance_to_plane_m": float(
                        gate_frame["distance_to_plane_m"][env_id]
                    ),
                },
                "passed_gate": bool(info["passed_gate"][env_id]),
                "missed_gate": bool(info["missed_gate"][env_id]),
                "collision": bool(info["collision"][env_id]),
                "out_of_bounds": bool(info["out_of_bounds"][env_id]),
                "done": bool(info["done"][env_id]),
            }
            trajectories[str(env_id)]["samples"].append(sample)

        passed_ids = torch.where(info["passed_gate"])[0].detach().cpu().tolist()
        for env_id in passed_ids:
            lateral_offset = float(info["gate_lateral_offset"][env_id])
            vertical_offset = float(info["gate_vertical_offset"][env_id])
            event = {
                "env_id": env_id,
                "gate_index": int(info["active_gate_index"][env_id]),
                "step": int(episode_steps[env_id]),
                "time_s": float(episode_steps[env_id]) * env_config.dt,
                "position_m": _as_list(position[env_id]),
                "lateral_offset_m": lateral_offset,
                "vertical_offset_m": vertical_offset,
                "plane_offset_m": float(info["gate_plane_offset"][env_id]),
                "lateral_margin_m": env_config.gate_half_width_m
                - abs(lateral_offset),
                "vertical_margin_m": env_config.gate_half_height_m
                - abs(vertical_offset),
            }
            pending_gate_events[env_id].append(event)
            if env_id in active_trajectories:
                trajectories[str(env_id)]["gate_passages"].append(event)

        done_ids = torch.where(info["done"])[0].detach().cpu().tolist()
        accepted_done_ids = done_ids[: episode_count - len(episodes)]
        for env_id in accepted_done_ids:
            max_z = float(max_altitude[env_id])
            max_vz = float(max_abs_vertical_speed[env_id])
            longest_upward_s = float(longest_upward_run[env_id]) * env_config.dt
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
                    "missed_gate": bool(episode_missed_gate[env_id]),
                    "teacher_takeover_samples": int(episode_teacher_samples[env_id]),
                    "teacher_takeover_fraction": float(
                        episode_teacher_samples[env_id]
                        / episode_steps[env_id].clamp(min=1)
                    ),
                    "distance_reduction_m": float(
                        info["distance_reduction"][env_id]
                    ),
                    "min_altitude_m": float(min_altitude[env_id]),
                    "max_altitude_m": max_z,
                    "max_abs_vertical_speed_mps": max_vz,
                    "longest_sustained_upward_s": longest_upward_s,
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
            for env_id in done_ids:
                pending_gate_events[env_id] = []
            reset_ids = torch.tensor(done_ids, device=env.device)
            max_altitude[reset_ids] = env.position[reset_ids, 2]
            min_altitude[reset_ids] = env.position[reset_ids, 2]
            max_abs_vertical_speed[reset_ids] = env.velocity[reset_ids, 2].abs()
            upward_run[reset_ids] = 0
            longest_upward_run[reset_ids] = 0
            episode_missed_gate[reset_ids] = False
            episode_teacher_samples[reset_ids] = 0
            episode_steps[reset_ids] = 0

    gate_counts = [episode["gates_passed"] for episode in episodes]
    max_altitudes = [episode["max_altitude_m"] for episode in episodes]
    max_vertical_speeds = [
        episode["max_abs_vertical_speed_mps"] for episode in episodes
    ]
    teacher_fractions = [
        episode["teacher_takeover_fraction"] for episode in episodes
    ]
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
        "max_altitude_m_mean": sum(max_altitudes) / len(episodes),
        "max_altitude_m_p95": _percentile(max_altitudes, 0.95),
        "max_altitude_m_max": max(max_altitudes),
        "max_abs_vertical_speed_mps_p95": _percentile(max_vertical_speeds, 0.95),
        "max_abs_vertical_speed_mps_max": max(max_vertical_speeds),
        "gate_crossing_count": len(gate_events),
        "gate_crossing_min_margin_m": min(crossing_margins)
        if crossing_margins
        else None,
        "teacher_intervention_rate": intervention_samples / max(total_samples, 1),
        "teacher_takeover_episode_fraction_mean": sum(teacher_fractions)
        / len(teacher_fractions),
    }
    result = {
        "experiment": config["name"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "environment_randomization": randomization,
        "evaluation_seed": env_config.seed,
        "hybrid_mode": mode,
        "hybrid_parameters": {
            "takeover_distance_m": takeover_distance_m,
            "lateral_threshold_m": lateral_threshold_m,
            "vertical_threshold_m": vertical_threshold_m,
            "gate_indices": sorted(gate_indices) if gate_indices is not None else "all",
        },
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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "policy",
            "teacher",
            "near_gate_teacher",
            "off_center_near_gate_teacher",
        ),
        default="near_gate_teacher",
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--randomization", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--takeover-distance-m", type=float, default=10.0)
    parser.add_argument("--lateral-threshold-m", type=float, default=0.40)
    parser.add_argument("--vertical-threshold-m", type=float, default=0.40)
    parser.add_argument(
        "--gate-indices",
        default="all",
        help="Comma-separated active gate indices to allow teacher takeover, or all.",
    )
    args = parser.parse_args()

    evaluate_hybrid(
        args.config,
        args.checkpoint,
        output_path=args.output,
        episode_count=args.episodes,
        trajectory_count=args.trajectories,
        randomization=args.randomization,
        seed=args.seed,
        mode=args.mode,
        takeover_distance_m=args.takeover_distance_m,
        lateral_threshold_m=args.lateral_threshold_m,
        vertical_threshold_m=args.vertical_threshold_m,
        gate_indices=parse_gate_indices(args.gate_indices),
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
