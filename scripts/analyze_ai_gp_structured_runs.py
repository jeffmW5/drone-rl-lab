from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def analyze_session(session_dir: Path) -> dict[str, Any]:
    summary_path = session_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in {session_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = _read_jsonl(session_dir / "events.jsonl")
    policy_rows = _read_jsonl(session_dir / "policy.jsonl")

    collisions = [event for event in events if event.get("event_type") == "collision"]
    first_collision = collisions[0] if collisions else None
    collision_time = (
        float(first_collision["monotonic_time_s"]) if first_collision else None
    )
    last_policy = _last_before(policy_rows, collision_time)
    race_before = _race_before(events, collision_time)
    gate_plane = last_policy.get("gate_plane") if last_policy else None
    command = last_policy.get("mapped_command") if last_policy else None
    raw_command = last_policy.get("raw_mapped_command") if last_policy else None
    position = last_policy.get("position_m") if last_policy else None
    velocity = last_policy.get("velocity_mps") if last_policy else None

    return {
        "session": session_dir.name,
        "abort_reason": summary.get("abort_reason"),
        "max_gate": int(summary.get("max_active_gate_index", 0)),
        "gate0_passed": bool(summary.get("gate0_passed", False)),
        "race_finished": bool(summary.get("race_finished", False)),
        "collision_count": int(summary.get("collision_count", 0)),
        "collision_impact": _round(
            _nested(first_collision, "fields", "impact") if first_collision else None
        ),
        "active_gate_at_collision": _nested(
            race_before,
            "fields",
            "active_gate_index",
        ),
        "policy_steps": int(summary.get("policy_steps", 0)),
        "command_count": int(summary.get("command_count", 0)),
        "max_altitude_rise_m": _round(summary.get("max_altitude_rise_m")),
        "max_upward_speed_mps": _round(summary.get("max_upward_speed_mps")),
        "thrust_multiplier": summary.get("thrust_multiplier"),
        "roll_rate_multiplier": summary.get("roll_rate_multiplier"),
        "pitch_rate_multiplier": summary.get("pitch_rate_multiplier"),
        "yaw_rate_multiplier": summary.get("yaw_rate_multiplier"),
        "use_sim_gate_normals": bool(summary.get("use_sim_gate_normals", False)),
        "signed_distance_m": _round(_nested(gate_plane, "signed_distance_m")),
        "lateral_offset_m": _round(_nested(gate_plane, "lateral_offset_m")),
        "vertical_offset_m": _round(_nested(gate_plane, "vertical_offset_m")),
        "rectangular_margin_m": _round(_nested(gate_plane, "rectangular_margin_m")),
        "thrust_normalized": _round(_nested(command, "thrust_normalized")),
        "roll_rate_radps": _round(_nested(command, "roll_rate_radps")),
        "pitch_rate_radps": _round(_nested(command, "pitch_rate_radps")),
        "yaw_rate_radps": _round(_nested(command, "yaw_rate_radps")),
        "raw_thrust_normalized": _round(_nested(raw_command, "thrust_normalized")),
        "raw_roll_rate_radps": _round(_nested(raw_command, "roll_rate_radps")),
        "raw_pitch_rate_radps": _round(_nested(raw_command, "pitch_rate_radps")),
        "raw_yaw_rate_radps": _round(_nested(raw_command, "yaw_rate_radps")),
        "position_x": _round(_nested(position, "x")),
        "position_y": _round(_nested(position, "y")),
        "position_z": _round(_nested(position, "z")),
        "speed_mps": _round(_speed(velocity)),
        "observation": _float_list(last_policy.get("observation") if last_policy else None),
        "observation_features": last_policy.get("observation_features") if last_policy else None,
        "normalized_action": last_policy.get("normalized_action") if last_policy else None,
        "policy_log_time_s": _round(
            last_policy.get("monotonic_time_s") if last_policy else None
        ),
    }


def rank_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row["race_finished"]),
            int(row["max_gate"]),
            -int(row["collision_count"]),
            int(row["policy_steps"]),
        ),
        reverse=True,
    )


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    best = rank_sessions(rows)[0]
    return {
        "run_count": len(rows),
        "best_session": best["session"],
        "best_max_gate": best["max_gate"],
        "race_finish_count": sum(bool(row["race_finished"]) for row in rows),
        "gate0_pass_count": sum(bool(row["gate0_passed"]) for row in rows),
        "collision_run_count": sum(int(row["collision_count"]) > 0 for row in rows),
        "mean_max_gate": _round(
            sum(int(row["max_gate"]) for row in rows) / max(len(rows), 1)
        ),
        "mean_collision_count": _round(
            sum(int(row["collision_count"]) for row in rows) / max(len(rows), 1)
        ),
        "mean_policy_steps": _round(
            sum(int(row["policy_steps"]) for row in rows) / max(len(rows), 1)
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _last_before(
    rows: list[dict[str, Any]],
    monotonic_time_s: float | None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    if monotonic_time_s is None:
        return rows[-1]
    before = [
        row
        for row in rows
        if float(row.get("monotonic_time_s", 0.0)) <= monotonic_time_s
    ]
    return before[-1] if before else rows[0]


def _race_before(
    events: list[dict[str, Any]],
    monotonic_time_s: float | None,
) -> dict[str, Any] | None:
    race = [event for event in events if event.get("event_type") == "race.status"]
    return _last_before(race, monotonic_time_s)


def _nested(payload: Any, *keys: str) -> Any:
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _speed(vector: Any) -> float | None:
    if not isinstance(vector, dict):
        return None
    try:
        return (
            float(vector["x"]) ** 2
            + float(vector["y"]) ** 2
            + float(vector["z"]) ** 2
        ) ** 0.5
    except KeyError:
        return None


def _round(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    session_dirs: list[Path] = []
    for path in args.sessions:
        if path.is_dir() and (path / "summary.json").exists():
            session_dirs.append(path)
        elif path.is_dir():
            session_dirs.extend(
                child for child in path.iterdir() if (child / "summary.json").exists()
            )
    rows = rank_sessions([analyze_session(path) for path in session_dirs])
    output = {"summary": summarize_group(rows), "sessions": rows}
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
