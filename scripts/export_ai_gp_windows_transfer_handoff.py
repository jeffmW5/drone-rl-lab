from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_ai_gp_structured_runs import analyze_session, rank_sessions, summarize_group


SESSION_ROOT = REPO_ROOT / "tmp" / "ai-grand-prix-stack-remote" / "replay" / "sessions"
POLICY_PATH = REPO_ROOT / "exports" / "ai_gp" / "ai_gp_040_near_gate_teacher_structured_policy.json"


def export_handoff(
    run_id: str,
    output_path: Path,
    *,
    session_root: Path = SESSION_ROOT,
    max_cases: int = 40,
) -> dict[str, Any]:
    rows = []
    for session_dir in sorted(session_root.glob(f"{run_id}_c*")):
        summary_path = session_dir / "summary.json"
        if not summary_path.exists():
            continue
        row = analyze_session(session_dir)
        row["config"] = _config_name(session_dir.name)
        rows.append(row)
    if not rows:
        raise ValueError(f"no analyzed sessions found for run_id={run_id}")

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
    config_rankings = []
    for config, config_rows in by_config.items():
        best = rank_sessions(config_rows)[0]
        summary = summarize_group(config_rows)
        summary.update(
            {
                "config": config,
                "multipliers": {
                    "thrust": best["thrust_multiplier"],
                    "roll": best["roll_rate_multiplier"],
                    "pitch": best["pitch_rate_multiplier"],
                    "yaw": best["yaw_rate_multiplier"],
                },
                "best_collision_context": _collision_context(best),
            }
        )
        config_rankings.append(summary)
    config_rankings.sort(
        key=lambda item: (
            item["best_max_gate"],
            item["gate0_pass_count"],
            -item["collision_run_count"],
            item["mean_policy_steps"],
        ),
        reverse=True,
    )

    top_multipliers = config_rankings[0]["multipliers"]
    hard_cases = [
        _collision_context(row)
        for row in rank_sessions(rows)
        if int(row["collision_count"]) > 0
    ][:max_cases]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "windows_ai_gp_sim_runtime",
        "source_run_id": run_id,
        "source_sessions_glob": str(session_root / f"{run_id}_c*"),
        "policy_export": {
            "path": "exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json",
            "sha256": _sha256_from_policy_artifact(),
            "observation_contract": "structured_teacher_v2",
        },
        "best_runtime_baseline": {
            "thrust_multiplier": top_multipliers["thrust"],
            "roll_rate_multiplier": top_multipliers["roll"],
            "pitch_rate_multiplier": top_multipliers["pitch"],
            "yaw_rate_multiplier": top_multipliers["yaw"],
            "use_sim_gate_normals": False,
            "note": "Top 12-config Windows sweep baseline; reached active gate index 2 but did not clear gate 2.",
        },
        "windows_findings": [
            "Manual multiplier tuning is not sufficient; all tested configs top out around active gate index 2.",
            "Pitch-rate boosting degraded behavior; keep pitch multiplier at 1.00 for runtime baselines.",
            "Roll multiplier 2.00 is consistently better than 1.75 for reaching gate 2.",
            "Useful thrust band is about 1.10-1.12; 1.50 causes altitude aborts.",
            "Simulator gate normals from track.gate quaternions did not improve transfer in the tested y-axis mode.",
            "Most useful failures are gate-1/gate-2 crossing collisions at 3.8-5.6 m/s.",
        ],
        "training_objective": {
            "recommended_run_name": "ai_gp_041_windows_transfer_gate2_hardcase_30m",
            "start_from": "results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt",
            "goal": "Improve Windows AI-GP simulator transfer past gate 2 without sacrificing gate-0/gate-1 reliability.",
            "minimum_acceptance": {
                "windows_structured_gate0_pass_rate": ">= 0.90",
                "windows_structured_mean_max_gate": "> 2.0",
                "windows_structured_best_gate_index": ">= 3",
                "surrogate_nominal_success_rate": "do not regress materially from 040",
            },
        },
        "config_rankings": config_rankings,
        "hard_cases": hard_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _config_name(session_name: str) -> str:
    if session_name.endswith(tuple(f"_{index:02d}" for index in range(1, 100))):
        return session_name[:-3]
    return session_name


def _collision_context(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "session",
        "config",
        "abort_reason",
        "max_gate",
        "active_gate_at_collision",
        "collision_count",
        "collision_impact",
        "signed_distance_m",
        "lateral_offset_m",
        "vertical_offset_m",
        "rectangular_margin_m",
        "speed_mps",
        "position_x",
        "position_y",
        "position_z",
        "thrust_normalized",
        "roll_rate_radps",
        "pitch_rate_radps",
        "yaw_rate_radps",
        "raw_thrust_normalized",
        "raw_roll_rate_radps",
        "raw_pitch_rate_radps",
        "raw_yaw_rate_radps",
        "thrust_multiplier",
        "roll_rate_multiplier",
        "pitch_rate_multiplier",
        "yaw_rate_multiplier",
        "policy_steps",
        "policy_log_time_s",
        "observation",
        "observation_features",
        "normalized_action",
    )
    return {key: row.get(key) for key in keys if key in row}


def _sha256_from_policy_artifact() -> str | None:
    if not POLICY_PATH.exists():
        return None
    import hashlib

    return hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-cases", type=int, default=40)
    args = parser.parse_args()
    result = export_handoff(args.run_id, args.output, max_cases=args.max_cases)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "config_count": len(result["config_rankings"]),
                "hard_case_count": len(result["hard_cases"]),
                "top_config": result["config_rankings"][0]["config"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
