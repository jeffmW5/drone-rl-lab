#!/usr/bin/env python3
"""Attach a deployable action governor to a live-student checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def configure_governor(
    source_path: Path,
    output_path: Path,
    *,
    slew_limits: tuple[float, float, float, float],
    upward_brake_start_mps: float,
    upward_brake_gain: float,
    lineage_checkpoint_path: Path | None = None,
) -> None:
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("policy_role") != "distilled_live_student":
        raise ValueError("action governors require a distilled live student")
    if len(slew_limits) != 4 or any(value <= 0.0 for value in slew_limits):
        raise ValueError("action governor requires four positive slew limits")
    if upward_brake_start_mps < 0.0 or upward_brake_gain < 0.0:
        raise ValueError("action governor braking parameters must be non-negative")

    if lineage_checkpoint_path is not None:
        lineage = torch.load(
            lineage_checkpoint_path, map_location="cpu", weights_only=False
        )
        lineage_metadata = lineage.get("metadata", {})
        for key in ("source_teacher_checkpoint", "source_teacher_global_step"):
            if metadata.get(key) is None and lineage_metadata.get(key) is not None:
                metadata[key] = lineage_metadata[key]

    metadata["action_governor"] = {
        "slew_limits": list(slew_limits),
        "upward_brake_start_mps": upward_brake_start_mps,
        "upward_brake_gain": upward_brake_gain,
    }
    metadata["source_ungoverned_checkpoint"] = source_path.name
    checkpoint["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--slew-limits",
        type=float,
        nargs=4,
        default=(0.15, 0.25, 0.25, 0.35),
    )
    parser.add_argument("--upward-brake-start-mps", type=float, default=1.0)
    parser.add_argument("--upward-brake-gain", type=float, default=0.10)
    parser.add_argument("--lineage-checkpoint", type=Path)
    args = parser.parse_args()
    configure_governor(
        args.source,
        args.output,
        slew_limits=tuple(args.slew_limits),
        upward_brake_start_mps=args.upward_brake_start_mps,
        upward_brake_gain=args.upward_brake_gain,
        lineage_checkpoint_path=args.lineage_checkpoint,
    )


if __name__ == "__main__":
    main()
