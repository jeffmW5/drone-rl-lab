#!/usr/bin/env python3
"""Run gate detection offline, then export synchronized PPO observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STACK_ROOT = REPO_ROOT / "tmp" / "ai-grand-prix-stack-remote"
for path in (REPO_ROOT, STACK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapter.vision import VisionFrame, VisionStorageKind
from ai_gp_rl.session_dataset import export_session_dataset
from perception.highlighted_gate_detector import HighlightedGateDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    parser.add_argument("--max-telemetry-age", type=float, default=0.10)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    detector = HighlightedGateDetector()
    records = _deduplicate_frames(_read_jsonl(session_dir / "vision.jsonl"))
    detection_path = session_dir / "detections.jsonl"

    with detection_path.open("w", encoding="utf-8") as handle:
        for record in records:
            frame_path = Path(str(record["storage_ref"]))
            if not frame_path.is_absolute():
                frame_path = session_dir / frame_path
            frame = VisionFrame(
                frame_id=str(record["frame_id"]),
                monotonic_time_s=float(record["monotonic_time_s"]),
                width_px=int(record["width_px"]),
                height_px=int(record["height_px"]),
                pixel_format=str(record["pixel_format"]),
                source=str(record["source"]),
                storage_kind=VisionStorageKind.PATH,
                storage_ref=str(frame_path),
            )
            detections = detector.detect(frame)
            payload = {
                "monotonic_time_s": frame.monotonic_time_s,
                "frame_id": frame.frame_id,
                "detection_count": len(detections),
                "detections": [_detection_record(detection) for detection in detections],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary = export_session_dataset(
        session_dir, max_telemetry_age_s=args.max_telemetry_age
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


def _detection_record(detection) -> dict[str, object]:
    return {
        "confidence": detection.confidence,
        "bbox": {
            "center": {
                "x": detection.bbox.center.x,
                "y": detection.bbox.center.y,
            },
            "width": detection.bbox.width,
            "height": detection.bbox.height,
        },
        "sequence_hint": detection.sequence_hint,
        "gate_label": detection.gate_label,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _deduplicate_frames(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for record in records:
        frame_id = str(record["frame_id"])
        if frame_id not in unique:
            unique[frame_id] = record
    return list(unique.values())


if __name__ == "__main__":
    main()
