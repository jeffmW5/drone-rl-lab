#!/usr/bin/env python3
"""Export a telemetry-validated 18D live policy as dependency-light JSON."""

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

from ai_gp_rl.contract import ACTOR_FEATURE_NAMES, ACTION_NAMES
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
    if tuple(metadata.get("actor_features", ())) != ACTOR_FEATURE_NAMES:
        raise ValueError("checkpoint does not use the 18D live actor contract")
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
    test_inputs = torch.rand((4, len(ACTOR_FEATURE_NAMES)), generator=generator) * 2 - 1
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
        "actor_features": list(ACTOR_FEATURE_NAMES),
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
