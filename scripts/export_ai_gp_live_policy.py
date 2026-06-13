#!/usr/bin/env python3
"""Export a telemetry-validated live policy as dependency-light JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.contract import (
    ACTION_NAMES,
    ACTOR_FEATURE_NAMES,
    TEMPORAL_BASE_FEATURE_NAMES,
    temporal_feature_names,
)
from ai_gp_rl.model import ActorCritic


def export_policy(
    checkpoint_path: Path,
    validation_report_path: Path,
    output_path: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if metadata.get("policy_role") != "distilled_live_student":
        raise ValueError("only distilled_live_student checkpoints may be exported")
    actor_features = tuple(metadata.get("actor_features", ()))
    observation_contract = metadata.get("observation_contract", "live_features_v1")
    base_actor_features = tuple(
        metadata.get("base_actor_features", ACTOR_FEATURE_NAMES)
    )
    history_length = int(metadata.get("history_length", 1))
    if observation_contract == "live_features_v1":
        if (
            actor_features != ACTOR_FEATURE_NAMES
            or base_actor_features != ACTOR_FEATURE_NAMES
            or history_length != 1
        ):
            raise ValueError("checkpoint does not use the 18D live actor contract")
    elif observation_contract == "temporal_live_v1":
        if (
            base_actor_features != TEMPORAL_BASE_FEATURE_NAMES
            or actor_features != temporal_feature_names(history_length)
        ):
            raise ValueError("checkpoint temporal live contract does not match")
    else:
        raise ValueError(f"unsupported observation contract: {observation_contract}")
    if int(metadata["actor_observation_dim"]) != len(actor_features):
        raise ValueError("checkpoint actor feature count does not match model input")
    validation_report = json.loads(
        validation_report_path.read_text(encoding="utf-8")
    )
    if validation_report.get("checkpoint") != checkpoint_path.name:
        raise ValueError("validation report does not match checkpoint")
    summary = validation_report["deterministic_summary"]
    validation_passed = (
        float(summary["gate0_passage_rate"]) >= 0.95
        and float(summary["mean_gates"]) >= 2.0
        and float(summary["collision_rate"]) < 0.20
        and float(summary["out_of_bounds_rate"]) < 0.20
        and float(summary["vertical_runaway_rate"]) == 0.0
    )
    if not validation_passed:
        raise ValueError("checkpoint failed deterministic telemetry promotion")

    model = ActorCritic(
        observation_dim=int(metadata["observation_dim"]),
        action_dim=int(metadata["action_dim"]),
        actor_observation_dim=int(metadata["actor_observation_dim"]),
        hidden_sizes=tuple(metadata["hidden_sizes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    layers = []
    for module in model.actor_mean:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().tolist(),
                    "bias": module.bias.detach().tolist(),
                }
            )

    generator = torch.Generator().manual_seed(20260609)
    test_inputs = torch.rand((4, len(actor_features)), generator=generator) * 2 - 1
    with torch.no_grad():
        expected_actions = torch.tanh(model.actor_mean(test_inputs)).tolist()
    artifact = {
        "schema_version": 1,
        "policy_role": "distilled_live_student",
        "validation_status": "surrogate_passed_pending_windows_simulator",
        "surrogate_validation_summary": summary,
        "source_checkpoint": checkpoint_path.name,
        "source_global_step": int(checkpoint.get("global_step", 0)),
        "source_teacher_checkpoint": metadata["source_teacher_checkpoint"],
        "observation_contract": observation_contract,
        "base_actor_features": list(base_actor_features),
        "history_length": history_length,
        "actor_features": list(actor_features),
        "action_names": list(ACTION_NAMES),
        "hidden_activation": {"name": "leaky_relu", "negative_slope": 0.2},
        "output_activation": "tanh",
        "layers": layers,
        "test_vectors": [
            {"observation": observation, "expected_action": action}
            for observation, action in zip(test_inputs.tolist(), expected_actions)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact), encoding="utf-8")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("validation_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_policy(args.checkpoint, args.validation_report, args.output)


if __name__ == "__main__":
    main()
