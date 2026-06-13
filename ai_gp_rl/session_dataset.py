"""Convert recorded AI-GP sessions into deployable policy observations."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contract import (
    ACTOR_FEATURE_NAMES,
    DETECTION_AGE_SCALE_S,
    LivePolicyFeatures,
    TEMPORAL_BASE_FEATURE_NAMES,
    TemporalLivePolicyFeatures,
    build_actor_observation,
    build_temporal_base_observation,
)


@dataclass(frozen=True)
class SessionDatasetSummary:
    session_id: str
    row_count: int
    skipped_no_telemetry: int
    full_detection_rows: int
    persisted_frame_count: int
    mean_telemetry_age_s: float | None
    max_telemetry_age_s: float | None
    output_path: str


def export_session_dataset(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    max_telemetry_age_s: float = 0.10,
) -> SessionDatasetSummary:
    session_path = Path(session_dir)
    detections = _read_jsonl(session_path / "detections.jsonl")
    telemetry = [
        record
        for record in _read_jsonl(session_path / "telemetry.jsonl")
        if _has_policy_telemetry(record)
    ]
    telemetry.sort(key=lambda record: float(record["monotonic_time_s"]))
    telemetry_times = [float(record["monotonic_time_s"]) for record in telemetry]
    output = Path(output_path) if output_path is not None else session_path / "rl_features.jsonl"

    rows: list[dict[str, Any]] = []
    skipped_no_telemetry = 0
    full_detection_rows = 0
    telemetry_ages: list[float] = []
    last_detection: dict[str, Any] | None = None
    last_detection_time: float | None = None

    for detection_record in detections:
        timestamp = float(detection_record["monotonic_time_s"])
        telemetry_index = bisect.bisect_right(telemetry_times, timestamp) - 1
        if telemetry_index < 0:
            skipped_no_telemetry += 1
            continue
        telemetry_record = telemetry[telemetry_index]
        telemetry_age = timestamp - telemetry_times[telemetry_index]
        if telemetry_age > max_telemetry_age_s:
            skipped_no_telemetry += 1
            continue

        candidates = detection_record.get("detections") or []
        selected_detection = candidates[0] if candidates else None
        if selected_detection is not None:
            last_detection = selected_detection
            last_detection_time = timestamp
            full_detection_rows += 1

        gate_center, gate_size, gate_area = _gate_features(last_detection)
        gate_age = (
            DETECTION_AGE_SCALE_S
            if last_detection_time is None
            else min(timestamp - last_detection_time, DETECTION_AGE_SCALE_S)
        )
        confidence = (
            float(selected_detection.get("confidence", 0.0))
            if selected_detection is not None
            else 0.0
        )
        body_velocity, gravity_body = _telemetry_body_features(telemetry_record)
        angular_rate = telemetry_record["angular_rate_radps"]
        features = LivePolicyFeatures(
            body_velocity_mps=body_velocity,
            gravity_body=gravity_body,
            angular_rate_radps=(
                float(angular_rate["x"]),
                float(angular_rate["y"]),
                float(angular_rate["z"]),
            ),
            gate_center_normalized=gate_center,
            gate_area_normalized=gate_area,
            gate_confidence=confidence,
            gate_age_s=gate_age,
            previous_action=(0.0, 0.0, 0.0, 0.0),
        )
        observation = build_actor_observation(features)
        temporal_base_observation = build_temporal_base_observation(
            TemporalLivePolicyFeatures(
                body_velocity_mps=body_velocity,
                gravity_body=gravity_body,
                angular_rate_radps=(
                    float(angular_rate["x"]),
                    float(angular_rate["y"]),
                    float(angular_rate["z"]),
                ),
                gate_center_normalized=gate_center,
                gate_size_normalized=gate_size,
                gate_area_normalized=gate_area,
                gate_confidence=confidence,
                gate_age_s=gate_age,
                previous_action=(0.0, 0.0, 0.0, 0.0),
            )
        )
        rows.append(
            {
                "frame_id": str(detection_record["frame_id"]),
                "monotonic_time_s": timestamp,
                "telemetry_time_s": telemetry_times[telemetry_index],
                "telemetry_age_s": telemetry_age,
                "detection_count": int(detection_record.get("detection_count", len(candidates))),
                "actor_features": dict(zip(ACTOR_FEATURE_NAMES, observation)),
                "actor_observation": observation,
                "temporal_base_features": dict(
                    zip(TEMPORAL_BASE_FEATURE_NAMES, temporal_base_observation)
                ),
                "temporal_base_observation": temporal_base_observation,
                "source": {
                    "telemetry": telemetry_record,
                    "detection": selected_detection,
                },
                "coordinate_contract": "telemetry NED/FRD to policy FLU v1; angular rates retain adapter signs",
            }
        )
        telemetry_ages.append(telemetry_age)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = SessionDatasetSummary(
        session_id=session_path.name,
        row_count=len(rows),
        skipped_no_telemetry=skipped_no_telemetry,
        full_detection_rows=full_detection_rows,
        persisted_frame_count=len(list((session_path / "frames").glob("*")))
        if (session_path / "frames").exists()
        else 0,
        mean_telemetry_age_s=(
            sum(telemetry_ages) / len(telemetry_ages) if telemetry_ages else None
        ),
        max_telemetry_age_s=max(telemetry_ages) if telemetry_ages else None,
        output_path=str(output),
    )
    (session_path / "rl_features_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _has_policy_telemetry(record: dict[str, Any]) -> bool:
    return all(
        isinstance(record.get(field), dict)
        for field in ("velocity_mps", "attitude_rad", "angular_rate_radps")
    )


def _gate_features(
    detection: dict[str, Any] | None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    if detection is None:
        return (0.0, 0.0), (0.0, 0.0), 0.0
    bbox = detection["bbox"]
    center = bbox["center"]
    width = float(bbox["width"])
    height = float(bbox["height"])
    return (
        (
            (float(center["x"]) - 0.5) * 2.0,
            (float(center["y"]) - 0.5) * 2.0,
        ),
        (width, height),
        width * height,
    )


def _telemetry_body_features(
    telemetry: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    attitude = telemetry["attitude_rad"]
    roll = float(attitude["roll_rad"])
    pitch = float(attitude["pitch_rad"])
    yaw = float(attitude["yaw_rad"])
    rotation_body_frd_to_ned = _rotation_matrix(roll, pitch, yaw)

    velocity = telemetry["velocity_mps"]
    velocity_ned = (
        float(velocity["x"]),
        float(velocity["y"]),
        float(velocity["z"]),
    )
    velocity_frd = _transpose_matvec(rotation_body_frd_to_ned, velocity_ned)
    gravity_frd = _transpose_matvec(rotation_body_frd_to_ned, (0.0, 0.0, 1.0))
    return (
        (velocity_frd[0], -velocity_frd[1], -velocity_frd[2]),
        (gravity_frd[0], -gravity_frd[1], -gravity_frd[2]),
    )


def _rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _transpose_matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
