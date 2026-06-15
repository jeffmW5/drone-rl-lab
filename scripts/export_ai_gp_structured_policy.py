#!/usr/bin/env python3
"""Export a structured-state AI-GP policy as simulator-ready JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.contract import (
    ACTION_NAMES,
    RATE_SCALES_RADPS,
    STRUCTURED_TEACHER_FEATURE_NAMES,
    STRUCTURED_TEACHER_OBS_DIM,
    VELOCITY_SCALE_MPS,
)
from ai_gp_rl.model import ActorCritic, build_policy_from_metadata
from ai_gp_rl.track import (
    AI_GP_GATE_SIZE_M,
    AI_GP_TRACK_GATES_NED,
    AI_GP_TRACK_NAME,
    ai_gp_track_altitude_offset_m,
    ai_gp_track_surrogate_positions,
)


NOMINAL_PROMOTION_STATUS = "structured_sim_nominal_passed_randomized_partial"
POLICY_ROLE = "structured_state_sim_teacher"
OBSERVATION_CONTRACT = "structured_teacher_v2"


def export_policy(
    checkpoint_path: Path,
    nominal_validation_report_path: Path,
    output_path: Path,
    *,
    randomized_validation_report_paths: tuple[Path, ...] = (),
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    _validate_metadata(metadata)

    nominal_validation = _load_validation_report(
        nominal_validation_report_path,
        checkpoint_name=checkpoint_path.name,
    )
    _validate_nominal_promotion(nominal_validation["summary"])
    randomized_validations = [
        _load_validation_report(path, checkpoint_name=checkpoint_path.name)
        for path in randomized_validation_report_paths
    ]

    model = build_policy_from_metadata(metadata, device="cpu")
    if not isinstance(model, ActorCritic):
        raise ValueError("structured AI-GP sim export currently supports MLP policies")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actor_features = tuple(metadata["actor_features"])
    action_calibration = metadata["action_calibration"]
    artifact = {
        "schema_version": 1,
        "policy_role": POLICY_ROLE,
        "policy_architecture": "mlp",
        "validation_status": NOMINAL_PROMOTION_STATUS,
        "requires_structured_state": True,
        "not_live_vision_policy": True,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_global_step": int(checkpoint.get("global_step", 0)),
        "training_method": metadata.get("training_method"),
        "expert": metadata.get("expert"),
        "observation_contract": OBSERVATION_CONTRACT,
        "actor_observation_dim": len(actor_features),
        "actor_features": list(actor_features),
        "observation_normalization": _observation_normalization(),
        "action_names": list(ACTION_NAMES),
        "output_activation": "tanh",
        "action_semantics": metadata.get("action_semantics"),
        "action_calibration": action_calibration,
        "action_command_map": _action_command_map(action_calibration),
        "track": _track_payload(),
        "validation": {
            "promotion_gate": "nominal telemetry only; randomized summaries are context",
            "nominal": nominal_validation,
            "randomized": randomized_validations,
        },
        "runtime_notes": [
            "Compute the 26 actor features in the exact listed order.",
            "Use the active gate index from race status and clamp the next gate to the final gate.",
            "Use the measured AI-GP gate map and upright inferred gate normals embedded here.",
            "This artifact is for structured simulator state; it is not a camera-only live policy.",
            "Use best_policy.pt, not final_policy.pt; the final checkpoint regressed after the best eval.",
        ],
        "hidden_activation": {
            "name": "leaky_relu",
            "negative_slope": 0.2,
        },
        "layers": _extract_mlp_layers(model),
        "test_vectors": _build_test_vectors(
            model,
            feature_count=len(actor_features),
            action_calibration=action_calibration,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"saved={output_path}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    actor_features = tuple(metadata.get("actor_features", ()))
    if actor_features != STRUCTURED_TEACHER_FEATURE_NAMES:
        raise ValueError("checkpoint does not use the structured_teacher_v2 actor contract")
    if int(metadata.get("actor_observation_dim", -1)) != STRUCTURED_TEACHER_OBS_DIM:
        raise ValueError("checkpoint actor feature count does not match structured_teacher_v2")
    if int(metadata.get("action_dim", -1)) != len(ACTION_NAMES):
        raise ValueError("checkpoint action dimension does not match AI-GP action contract")
    if tuple(metadata.get("action_names", ACTION_NAMES)) != ACTION_NAMES:
        raise ValueError("checkpoint action names do not match AI-GP action contract")
    observation_contract = metadata.get("observation_contract", OBSERVATION_CONTRACT)
    if observation_contract != OBSERVATION_CONTRACT:
        raise ValueError(f"unsupported observation contract: {observation_contract}")
    if metadata.get("policy_architecture", "mlp") != "mlp":
        raise ValueError("structured AI-GP sim export currently supports MLP policies")
    action_calibration = metadata.get("action_calibration")
    if not isinstance(action_calibration, dict):
        raise ValueError("checkpoint is missing measured action calibration")
    if action_calibration.get("contract") != "measured_ai_gp_v1":
        raise ValueError("checkpoint action calibration is not measured_ai_gp_v1")
    required_calibration_keys = (
        "thrust_command_center",
        "thrust_span_up",
        "thrust_span_down",
        "max_roll_rate_radps",
        "max_pitch_rate_radps",
        "max_yaw_rate_radps",
    )
    missing = [key for key in required_calibration_keys if key not in action_calibration]
    if missing:
        raise ValueError(f"checkpoint action calibration is missing: {missing}")


def _load_validation_report(path: Path, *, checkpoint_name: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("checkpoint") != checkpoint_name:
        raise ValueError(
            f"validation report {path} does not match checkpoint {checkpoint_name}"
        )
    summary = report.get("deterministic_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"validation report {path} is missing deterministic_summary")
    summary_keys = (
        "gate0_passage_rate",
        "mean_gates",
        "success_rate",
        "collision_rate",
        "out_of_bounds_rate",
        "missed_gate_rate",
        "vertical_runaway_rate",
        "max_altitude_m_mean",
        "max_altitude_m_p95",
        "max_altitude_m_max",
        "max_abs_vertical_speed_mps_p95",
        "max_abs_vertical_speed_mps_max",
        "gate_crossing_count",
        "gate_crossing_min_margin_m",
    )
    return {
        "path": str(path),
        "experiment": report.get("experiment"),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_global_step": report.get("checkpoint_global_step"),
        "environment_randomization": bool(report.get("environment_randomization", False)),
        "evaluation_seed": report.get("evaluation_seed"),
        "episodes": int(report.get("episodes", 0)),
        "summary": {
            key: float(summary[key])
            for key in summary_keys
            if key in summary
        },
    }


def _validate_nominal_promotion(summary: dict[str, float]) -> None:
    required = (
        "gate0_passage_rate",
        "mean_gates",
        "success_rate",
        "collision_rate",
        "out_of_bounds_rate",
        "missed_gate_rate",
        "vertical_runaway_rate",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"nominal validation summary is missing: {missing}")
    failures = []
    if summary["gate0_passage_rate"] < 0.99:
        failures.append("gate0_passage_rate < 0.99")
    if summary["mean_gates"] < 5.95:
        failures.append("mean_gates < 5.95")
    if summary["success_rate"] < 0.95:
        failures.append("success_rate < 0.95")
    for key in (
        "collision_rate",
        "out_of_bounds_rate",
        "missed_gate_rate",
        "vertical_runaway_rate",
    ):
        if summary[key] > 0.0:
            failures.append(f"{key} > 0.0")
    if failures:
        raise ValueError("nominal validation failed: " + ", ".join(failures))


def _extract_mlp_layers(model: ActorCritic) -> list[dict[str, list[list[float]] | list[float]]]:
    layers = []
    for module in model.actor_mean:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().tolist(),
                    "bias": module.bias.detach().tolist(),
                }
            )
    if len(layers) != len(model.hidden_sizes) + 1:
        raise ValueError("unexpected actor MLP layer structure")
    return layers


def _build_test_vectors(
    model: ActorCritic,
    *,
    feature_count: int,
    action_calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(20260615)
    test_inputs = torch.rand((4, feature_count), generator=generator) * 2.0 - 1.0
    with torch.no_grad():
        expected_actions = torch.tanh(model.actor_mean(test_inputs)).tolist()
    return [
        {
            "observation": observation,
            "expected_action": action,
            "expected_command": _map_action_to_command(action, action_calibration),
        }
        for observation, action in zip(test_inputs.tolist(), expected_actions)
    ]


def _map_action_to_command(
    action: list[float],
    action_calibration: dict[str, Any],
) -> dict[str, float]:
    collective, roll, pitch, yaw = (_clamp(float(value), -1.0, 1.0) for value in action)
    center = float(action_calibration["thrust_command_center"])
    thrust_span = (
        float(action_calibration["thrust_span_up"])
        if collective >= 0.0
        else float(action_calibration["thrust_span_down"])
    )
    return {
        "thrust_normalized": center + collective * thrust_span,
        "roll_rate_radps": roll * float(action_calibration["max_roll_rate_radps"]),
        "pitch_rate_radps": pitch * float(action_calibration["max_pitch_rate_radps"]),
        "yaw_rate_radps": yaw * float(action_calibration["max_yaw_rate_radps"]),
    }


def _action_command_map(action_calibration: dict[str, Any]) -> dict[str, str | float]:
    return {
        "collective_offset": (
            "thrust_command_center + action[0] * thrust_span_up when action[0] >= 0, "
            "else thrust_command_center + action[0] * thrust_span_down"
        ),
        "roll_rate_radps": "action[1] * max_roll_rate_radps",
        "pitch_rate_radps": "action[2] * max_pitch_rate_radps; negative pitch is forward",
        "yaw_rate_radps": "action[3] * max_yaw_rate_radps",
        "thrust_command_center": float(action_calibration["thrust_command_center"]),
        "thrust_span_up": float(action_calibration["thrust_span_up"]),
        "thrust_span_down": float(action_calibration["thrust_span_down"]),
        "max_roll_rate_radps": float(action_calibration["max_roll_rate_radps"]),
        "max_pitch_rate_radps": float(action_calibration["max_pitch_rate_radps"]),
        "max_yaw_rate_radps": float(action_calibration["max_yaw_rate_radps"]),
    }


def _observation_normalization() -> list[dict[str, Any]]:
    return [
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[0:3]),
            "source": "body_frame(active_gate_center_surrogate_flu_m - vehicle_position_surrogate_flu_m)",
            "scale": 30.0,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[3:6]),
            "source": "body_frame(active_gate_normal_surrogate_flu_unit)",
            "scale": 1.0,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[6:9]),
            "source": "body_frame(next_gate_center_surrogate_flu_m - vehicle_position_surrogate_flu_m)",
            "scale": 30.0,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[9:12]),
            "source": "body_frame(next_gate_normal_surrogate_flu_unit)",
            "scale": 1.0,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[12:15]),
            "source": "body_frame(vehicle_velocity_surrogate_flu_mps)",
            "scale": VELOCITY_SCALE_MPS,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[15:18]),
            "source": "body_frame(gravity_unit_surrogate_flu)",
            "scale": 1.0,
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[18:21]),
            "source": "body angular rates rad/s",
            "scale": list(RATE_SCALES_RADPS),
        },
        {
            "features": list(STRUCTURED_TEACHER_FEATURE_NAMES[21:25]),
            "source": "previous tanh-normalized policy action",
            "scale": 1.0,
        },
        {
            "features": [STRUCTURED_TEACHER_FEATURE_NAMES[25]],
            "source": "active_gate_index / max(gate_count - 1, 1)",
            "scale": 1.0,
        },
    ]


def _track_payload() -> dict[str, Any]:
    gates_surrogate = ai_gp_track_surrogate_positions()
    gate_normals = _gate_normals(gates_surrogate)
    return {
        "track_name": AI_GP_TRACK_NAME,
        "gate_count": len(AI_GP_TRACK_GATES_NED),
        "gate_size_m": {
            "width": AI_GP_GATE_SIZE_M,
            "height": AI_GP_GATE_SIZE_M,
        },
        "coordinate_contract": {
            "ai_gp_telemetry": "NED: x=north, y=east, z=down",
            "training_surrogate": "FLU: x=forward along decreasing north, y=east/left convention, z=up",
            "ned_to_surrogate": "surrogate = (-north, east, altitude_offset_m - down)",
            "gate_normals": "inferred from previous horizontal course segment and forced upright",
        },
        "surrogate_altitude_offset_m": ai_gp_track_altitude_offset_m(),
        "gates_ned": [
            {
                "index": index,
                "center_ned_m": list(center),
                "width_m": AI_GP_GATE_SIZE_M,
                "height_m": AI_GP_GATE_SIZE_M,
            }
            for index, center in enumerate(AI_GP_TRACK_GATES_NED)
        ],
        "gates_surrogate_flu": [
            {
                "index": index,
                "center_m": list(center),
                "normal_unit": list(gate_normals[index]),
            }
            for index, center in enumerate(gates_surrogate)
        ],
    }


def _gate_normals(
    gates: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    previous = ((gates[0][0] - 4.0, gates[0][1], gates[0][2]), *gates[:-1])
    normals = []
    for gate, prior in zip(gates, previous):
        vector = (gate[0] - prior[0], gate[1] - prior[1], 0.0)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm < 1e-6:
            raise ValueError("cannot infer gate normal from duplicate gate centers")
        normals.append(tuple(value / norm for value in vector))
    return tuple(normals)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("nominal_validation_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--randomized-validation-report",
        action="append",
        default=[],
        type=Path,
        help="Optional randomized telemetry report to include as context.",
    )
    args = parser.parse_args()
    export_policy(
        args.checkpoint,
        args.nominal_validation_report,
        args.output,
        randomized_validation_report_paths=tuple(args.randomized_validation_report),
    )


if __name__ == "__main__":
    main()
