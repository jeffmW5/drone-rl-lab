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
    CORNER_BASE_FEATURE_NAMES,
    MOTION_FEATURE_NAMES,
    TEMPORAL_BASE_FEATURE_NAMES,
    corner_temporal_feature_names,
    temporal_feature_names,
)
from ai_gp_rl.model import (
    ActorCritic,
    RecurrentStudentPolicy,
    build_policy_from_metadata,
)


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
    elif observation_contract == "corner_temporal_live_v1":
        if (
            base_actor_features != CORNER_BASE_FEATURE_NAMES
            or actor_features != corner_temporal_feature_names(history_length)
        ):
            raise ValueError("checkpoint corner temporal contract does not match")
    elif observation_contract == "recurrent_live_v1":
        if (
            base_actor_features != TEMPORAL_BASE_FEATURE_NAMES
            or actor_features != TEMPORAL_BASE_FEATURE_NAMES
            or history_length != 1
        ):
            raise ValueError("checkpoint recurrent live contract does not match")
    elif observation_contract == "corner_recurrent_live_v1":
        if (
            base_actor_features != CORNER_BASE_FEATURE_NAMES
            or actor_features != CORNER_BASE_FEATURE_NAMES
            or history_length != 1
        ):
            raise ValueError("checkpoint corner recurrent contract does not match")
    elif observation_contract == "motion_live_v1":
        if (
            base_actor_features != MOTION_FEATURE_NAMES
            or actor_features != MOTION_FEATURE_NAMES
            or history_length != 1
        ):
            raise ValueError("checkpoint motion live contract does not match")
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

    model = build_policy_from_metadata(metadata, device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    generator = torch.Generator().manual_seed(20260609)
    policy_architecture = str(metadata.get("policy_architecture", "mlp"))
    artifact = {
        "schema_version": 2 if policy_architecture == "gru" else 1,
        "policy_role": "distilled_live_student",
        "policy_architecture": policy_architecture,
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
        "output_activation": "tanh",
        "action_governor": metadata.get("action_governor"),
    }
    if isinstance(model, ActorCritic):
        layers = []
        for module in model.actor_mean:
            if isinstance(module, nn.Linear):
                layers.append(
                    {
                        "weight": module.weight.detach().tolist(),
                        "bias": module.bias.detach().tolist(),
                    }
                )
        test_inputs = (
            torch.rand((4, len(actor_features)), generator=generator) * 2 - 1
        )
        with torch.no_grad():
            expected_actions = torch.tanh(model.actor_mean(test_inputs)).tolist()
        artifact.update(
            {
                "hidden_activation": {
                    "name": "leaky_relu",
                    "negative_slope": 0.2,
                },
                "layers": layers,
                "test_vectors": [
                    {"observation": observation, "expected_action": action}
                    for observation, action in zip(
                        test_inputs.tolist(), expected_actions
                    )
                ],
            }
        )
    elif isinstance(model, RecurrentStudentPolicy):
        input_linear = model.input_encoder[0]
        action_layers = [
            module for module in model.action_head if isinstance(module, nn.Linear)
        ]
        test_inputs = (
            torch.rand((6, len(actor_features)), generator=generator) * 2 - 1
        )
        reset_before = [True, False, False, True, False, False]
        hidden = model.initial_state(1, device="cpu")
        expected_steps = []
        with torch.no_grad():
            for observation, reset in zip(test_inputs, reset_before):
                if reset:
                    hidden.zero_()
                raw_action, hidden = model.forward_step(
                    observation.unsqueeze(0), hidden
                )
                expected_steps.append(
                    {
                        "observation": observation.tolist(),
                        "reset_before": reset,
                        "expected_action": torch.tanh(raw_action)[0].tolist(),
                        "expected_hidden": hidden[0].tolist(),
                    }
                )
        artifact.update(
            {
                "hidden_activation": {
                    "name": "leaky_relu",
                    "negative_slope": 0.2,
                },
                "recurrent_hidden_size": model.hidden_sizes[0],
                "input_encoder": {
                    "weight": input_linear.weight.detach().tolist(),
                    "bias": input_linear.bias.detach().tolist(),
                },
                "gru_cell": {
                    "gate_order": ["reset", "update", "new"],
                    "weight_ih": model.recurrent.weight_ih.detach().tolist(),
                    "weight_hh": model.recurrent.weight_hh.detach().tolist(),
                    "bias_ih": model.recurrent.bias_ih.detach().tolist(),
                    "bias_hh": model.recurrent.bias_hh.detach().tolist(),
                },
                "action_layers": [
                    {
                        "weight": layer.weight.detach().tolist(),
                        "bias": layer.bias.detach().tolist(),
                    }
                    for layer in action_layers
                ],
                "test_sequence": expected_steps,
            }
        )
    else:
        raise ValueError("unsupported student model type")
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
