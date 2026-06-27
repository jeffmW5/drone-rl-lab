#!/usr/bin/env python3
"""Extract pre-failure hard cases from AI-GP telemetry evaluations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.track import ai_gp_track_surrogate_positions


def extract_hard_cases(
    telemetry_paths: list[Path],
    output_path: Path,
    *,
    window_s: float,
) -> dict[str, Any]:
    gates = ai_gp_track_surrogate_positions()
    normals = _gate_normals(gates)
    lateral, vertical = _gate_bases(normals)
    hard_cases: list[dict[str, Any]] = []
    tracked_trajectory_count = 0
    failed_tracked_trajectory_count = 0

    for telemetry_path in telemetry_paths:
        report = json.loads(telemetry_path.read_text(encoding="utf-8"))
        tracked_trajectory_count += len(report.get("trajectories", ()))
        for trajectory in report.get("trajectories", ()):
            samples = trajectory.get("samples", [])
            if not samples:
                continue
            final = samples[-1]
            failure_type = _failure_type(final)
            if failure_type is None:
                continue
            failed_tracked_trajectory_count += 1
            final_time_s = float(final["time_s"])
            for sample in samples:
                time_to_done_s = final_time_s - float(sample["time_s"])
                if time_to_done_s > window_s:
                    continue
                gate_index = min(
                    max(int(sample.get("active_gate_index", 0)), 0),
                    len(gates) - 1,
                )
                position = tuple(float(value) for value in sample["position_m"])
                velocity = tuple(float(value) for value in sample["velocity_mps"])
                gate_frame = _gate_frame_state(
                    position,
                    velocity,
                    gates[gate_index],
                    normals[gate_index],
                    lateral[gate_index],
                    vertical[gate_index],
                )
                hard_cases.append(
                    {
                        "source_report": str(telemetry_path),
                        "evaluation_seed": report.get("evaluation_seed"),
                        "environment_randomization": report.get(
                            "environment_randomization"
                        ),
                        "trajectory_env_id": trajectory.get("env_id"),
                        "failure_type": failure_type,
                        "time_to_done_s": time_to_done_s,
                        "sample": sample,
                        "gate_frame": {
                            "gate_index": gate_index,
                            **gate_frame,
                        },
                    }
                )

    summary = _summarize(hard_cases)
    result = {
        "source_reports": [str(path) for path in telemetry_paths],
        "tracked_trajectory_count": tracked_trajectory_count,
        "failed_tracked_trajectory_count": failed_tracked_trajectory_count,
        "window_s": window_s,
        "hard_case_count": len(hard_cases),
        "summary": summary,
        "hard_cases": hard_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "hard_cases"}, indent=2))
    print(f"saved={output_path}")
    return result


def _failure_type(sample: dict[str, Any]) -> str | None:
    if bool(sample.get("missed_gate")):
        return "missed_gate"
    if bool(sample.get("collision")):
        return "collision"
    if bool(sample.get("out_of_bounds")):
        return "out_of_bounds"
    if bool(sample.get("done")) and int(sample.get("gates_passed", 0)) < 6:
        return "incomplete_done"
    return None


def _gate_frame_state(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    gate: tuple[float, float, float],
    normal: tuple[float, float, float],
    lateral: tuple[float, float, float],
    vertical: tuple[float, float, float],
) -> dict[str, float]:
    offset = tuple(position[index] - gate[index] for index in range(3))
    return {
        "plane_offset_m": _dot(offset, normal),
        "lateral_offset_m": _dot(offset, lateral),
        "vertical_offset_m": _dot(offset, vertical),
        "forward_speed_mps": _dot(velocity, normal),
        "lateral_speed_mps": _dot(velocity, lateral),
        "vertical_speed_mps": _dot(velocity, vertical),
        "distance_to_plane_m": max(-_dot(offset, normal), 0.0),
    }


def _summarize(hard_cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_failure = Counter(case["failure_type"] for case in hard_cases)
    by_gate = Counter(case["gate_frame"]["gate_index"] for case in hard_cases)
    per_gate: dict[str, dict[str, Any]] = {}
    for gate_index in sorted(by_gate):
        cases = [
            case for case in hard_cases if case["gate_frame"]["gate_index"] == gate_index
        ]
        per_gate[str(gate_index)] = {
            "count": len(cases),
            "plane_offset_m": _stats(
                [case["gate_frame"]["plane_offset_m"] for case in cases]
            ),
            "lateral_offset_m": _stats(
                [case["gate_frame"]["lateral_offset_m"] for case in cases]
            ),
            "vertical_offset_m": _stats(
                [case["gate_frame"]["vertical_offset_m"] for case in cases]
            ),
            "forward_speed_mps": _stats(
                [case["gate_frame"]["forward_speed_mps"] for case in cases]
            ),
            "lateral_speed_mps": _stats(
                [case["gate_frame"]["lateral_speed_mps"] for case in cases]
            ),
            "vertical_speed_mps": _stats(
                [case["gate_frame"]["vertical_speed_mps"] for case in cases]
            ),
        }
    return {
        "by_failure_type": dict(sorted(by_failure.items())),
        "by_gate_index": {str(key): by_gate[key] for key in sorted(by_gate)},
        "per_gate": per_gate,
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "median": math.nan, "max": math.nan}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "max": ordered[-1],
    }


def _gate_normals(
    gates: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    previous = ((gates[0][0] - 4.0, gates[0][1], gates[0][2]), *gates[:-1])
    normals = []
    for gate, prior in zip(gates, previous):
        vector = (gate[0] - prior[0], gate[1] - prior[1], 0.0)
        norm = math.sqrt(_dot(vector, vector))
        if norm < 1e-6:
            raise ValueError("cannot infer gate normal from duplicate gate centers")
        normals.append(tuple(value / norm for value in vector))
    return tuple(normals)


def _gate_bases(
    normals: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    lateral = []
    vertical = []
    up = (0.0, 0.0, 1.0)
    for normal in normals:
        side = _cross(up, normal)
        side_norm = math.sqrt(_dot(side, side))
        if side_norm < 1e-6:
            side = (0.0, 1.0, 0.0)
        else:
            side = tuple(value / side_norm for value in side)
        up_axis = _cross(normal, side)
        up_norm = math.sqrt(_dot(up_axis, up_axis))
        vertical.append(tuple(value / up_norm for value in up_axis))
        lateral.append(side)
    return tuple(lateral), tuple(vertical)


def _dot(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("telemetry", nargs="+", type=Path)
    parser.add_argument("--window-s", type=float, default=2.0)
    args = parser.parse_args()
    extract_hard_cases(args.telemetry, args.output, window_s=args.window_s)


if __name__ == "__main__":
    main()
