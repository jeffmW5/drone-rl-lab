from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from scripts.analyze_ai_gp_structured_runs import analyze_session, rank_sessions, summarize_group


SESSION_ROOT = LAB_ROOT / "tmp" / "ai-grand-prix-stack-remote" / "replay" / "sessions"
DEFAULT_POLICIES = (
    "040=exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json",
    "041=exports/ai_gp/ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json",
)
WINDOWS_RETEST_TARGET = {
    "gate0_pass_rate": ">= 0.90",
    "mean_max_gate": "> 2.0",
    "best_max_gate": ">= 3",
}


@dataclass(frozen=True)
class PolicySpec:
    label: str
    path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    thrust_multiplier: float
    roll_rate_multiplier: float
    pitch_rate_multiplier: float
    yaw_rate_multiplier: float
    duration_s: float
    control_rate_hz: float
    attempts: int


def parse_policy_specs(raw_specs: Sequence[str]) -> list[PolicySpec]:
    policies: list[PolicySpec] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError("policy specs must use LABEL=PATH")
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label or any(char.isspace() for char in label):
            raise ValueError(f"invalid policy label: {label!r}")
        policy_path = Path(path.strip())
        if not policy_path.is_absolute():
            policy_path = LAB_ROOT / policy_path
        policies.append(PolicySpec(label=label, path=policy_path))
    if not policies:
        raise ValueError("at least one policy is required")
    return policies


def build_policy_command(
    *,
    python: str,
    policy: PolicySpec,
    config_id: str,
    runtime: RuntimeConfig,
) -> list[str]:
    return [
        python,
        "-u",
        "-B",
        str(LAB_ROOT / "scripts" / "run_ai_gp_structured_windows.py"),
        "--policy",
        str(policy.path),
        "--attempts",
        str(runtime.attempts),
        "--duration",
        str(runtime.duration_s),
        "--target-gates",
        "0",
        "--allow-gate-plane-miss",
        "--control-rate-hz",
        str(runtime.control_rate_hz),
        "--thrust-multiplier",
        str(runtime.thrust_multiplier),
        "--roll-rate-multiplier",
        str(runtime.roll_rate_multiplier),
        "--pitch-rate-multiplier",
        str(runtime.pitch_rate_multiplier),
        "--yaw-rate-multiplier",
        str(runtime.yaw_rate_multiplier),
        "--run-id",
        config_id,
    ]


def collect_policy_rows(
    *,
    policy: PolicySpec,
    config_id: str,
    attempts: int,
    session_root: Path = SESSION_ROOT,
) -> list[dict]:
    rows = []
    for attempt in range(1, attempts + 1):
        session = session_root / f"{config_id}_{attempt:02d}"
        if not session.exists():
            continue
        row = analyze_session(session)
        row["policy_label"] = policy.label
        row["policy_path"] = str(policy.path)
        row["config"] = config_id
        rows.append(row)
    return rows


def summarize_policy_rows(policy: PolicySpec, rows: list[dict]) -> dict:
    summary = summarize_group(rows)
    run_count = int(summary["run_count"])
    summary.update(
        {
            "policy_label": policy.label,
            "policy_path": str(policy.path),
            "policy_export": policy_export_metadata(policy),
            "gate0_pass_rate": summary["gate0_pass_count"] / max(run_count, 1),
            "collision_run_rate": summary["collision_run_count"] / max(run_count, 1),
            "passes_windows_retest_target": (
                summary["gate0_pass_count"] / max(run_count, 1) >= 0.90
                and float(summary["mean_max_gate"]) > 2.0
                and int(summary["best_max_gate"]) >= 3
            ),
            "hard_cases": collision_contexts(rows, max_cases=10),
        }
    )
    return summary


def rank_policy_summaries(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda item: (
            int(item["race_finish_count"]),
            int(item["best_max_gate"]),
            float(item["mean_max_gate"]),
            float(item["gate0_pass_rate"]),
            -float(item["collision_run_rate"]),
            float(item["mean_policy_steps"]),
        ),
        reverse=True,
    )


def policy_export_metadata(policy: PolicySpec) -> dict:
    metadata = {
        "label": policy.label,
        "path": _repo_relative(policy.path),
        "sha256": _sha256(policy.path),
    }
    try:
        artifact = json.loads(policy.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return metadata
    validation = artifact.get("validation", {})
    randomized = validation.get("randomized", [])
    metadata.update(
        {
            "policy_role": artifact.get("policy_role"),
            "observation_contract": artifact.get("observation_contract"),
            "source_checkpoint": artifact.get("source_checkpoint"),
            "source_global_step": artifact.get("source_global_step"),
            "nominal_success_rate": _nested(
                validation,
                "nominal",
                "summary",
                "success_rate",
            ),
            "randomized_success_rates": [
                _nested(report, "summary", "success_rate")
                for report in randomized
                if _nested(report, "summary", "success_rate") is not None
            ],
        }
    )
    rates = metadata["randomized_success_rates"]
    if rates:
        metadata["randomized_average_success_rate"] = sum(rates) / len(rates)
    return metadata


def collision_contexts(rows: list[dict], *, max_cases: int) -> list[dict]:
    return [
        _collision_context(row)
        for row in rank_sessions(rows)
        if int(row.get("collision_count", 0)) > 0
    ][:max_cases]


def follow_up_recommendation(ranked_policies: list[dict]) -> dict:
    if not ranked_policies:
        return {
            "status": "no_result",
            "message": "No policy sessions were analyzed; rerun the Windows A/B test.",
        }
    winner = ranked_policies[0]
    if bool(winner.get("passes_windows_retest_target")):
        return {
            "status": "candidate_passed_windows_target",
            "policy_label": winner["policy_label"],
            "message": (
                "Promote only after reviewing trajectories and confirming the "
                "result is repeatable."
            ),
        }
    return {
        "status": "needs_more_transfer_work",
        "policy_label": winner["policy_label"],
        "message": (
            "No policy satisfied the Windows retest target; use the hard_cases "
            "from this summary for the next focused transfer run."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        default=None,
        help="Policy to test as LABEL=PATH. Defaults to tracked 040 and 041 exports.",
    )
    parser.add_argument("--attempts-per-policy", type=int, default=5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--thrust-multiplier", type=float, default=1.12)
    parser.add_argument("--roll-rate-multiplier", type=float, default=2.0)
    parser.add_argument("--pitch-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--yaw-rate-multiplier", type=float, default=2.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sleep-between-policies-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.attempts_per_policy < 1:
        raise ValueError("--attempts-per-policy must be positive")
    runtime = RuntimeConfig(
        thrust_multiplier=args.thrust_multiplier,
        roll_rate_multiplier=args.roll_rate_multiplier,
        pitch_rate_multiplier=args.pitch_rate_multiplier,
        yaw_rate_multiplier=args.yaw_rate_multiplier,
        duration_s=args.duration,
        control_rate_hz=args.control_rate_hz,
        attempts=args.attempts_per_policy,
    )
    for name, value in (
        ("duration", runtime.duration_s),
        ("control_rate_hz", runtime.control_rate_hz),
        ("thrust_multiplier", runtime.thrust_multiplier),
        ("roll_rate_multiplier", runtime.roll_rate_multiplier),
        ("pitch_rate_multiplier", runtime.pitch_rate_multiplier),
        ("yaw_rate_multiplier", runtime.yaw_rate_multiplier),
    ):
        if value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    policies = parse_policy_specs(args.policies or DEFAULT_POLICIES)
    missing = [str(policy.path) for policy in policies if not policy.path.exists()]
    if missing:
        raise FileNotFoundError(f"policy export not found: {missing}")

    run_id = args.run_id or "structured_ab_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    policy_summaries = []
    for policy in policies:
        config_id = (
            f"{run_id}_p{policy.label}_t{_tag(runtime.thrust_multiplier)}_"
            f"r{_tag(runtime.roll_rate_multiplier)}_"
            f"p{_tag(runtime.pitch_rate_multiplier)}_"
            f"y{_tag(runtime.yaw_rate_multiplier)}"
        )
        command = build_policy_command(
            python=args.python,
            policy=policy,
            config_id=config_id,
            runtime=runtime,
        )
        print(
            json.dumps(
                {
                    "event": "policy_start",
                    "policy": policy.label,
                    "config": config_id,
                    "policy_path": str(policy.path),
                }
            ),
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=LAB_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            print(
                json.dumps(
                    {
                        "event": "policy_failed",
                        "policy": policy.label,
                        "config": config_id,
                        "returncode": completed.returncode,
                        "output_tail": completed.stdout[-2000:],
                    }
                ),
                flush=True,
            )
            continue
        rows = collect_policy_rows(
            policy=policy,
            config_id=config_id,
            attempts=runtime.attempts,
        )
        if not rows:
            print(
                json.dumps(
                    {
                        "event": "policy_no_sessions",
                        "policy": policy.label,
                        "config": config_id,
                    }
                ),
                flush=True,
            )
            continue
        all_rows.extend(rows)
        summary = summarize_policy_rows(policy, rows)
        summary["config"] = config_id
        policy_summaries.append(summary)
        print(json.dumps({"event": "policy_done", **summary}), flush=True)
        if args.sleep_between_policies_s > 0.0:
            time.sleep(args.sleep_between_policies_s)

    ranked_policies = rank_policy_summaries(policy_summaries)
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "windows_ai_gp_policy_ab_runtime",
        "run_id": run_id,
        "source_sessions_glob": str(SESSION_ROOT / f"{run_id}_p*"),
        "runtime": {
            "attempts_per_policy": runtime.attempts,
            "duration_s": runtime.duration_s,
            "control_rate_hz": runtime.control_rate_hz,
            "thrust_multiplier": runtime.thrust_multiplier,
            "roll_rate_multiplier": runtime.roll_rate_multiplier,
            "pitch_rate_multiplier": runtime.pitch_rate_multiplier,
            "yaw_rate_multiplier": runtime.yaw_rate_multiplier,
        },
        "winner": ranked_policies[0]["policy_label"] if ranked_policies else None,
        "recommendation": follow_up_recommendation(ranked_policies),
        "policies": ranked_policies,
        "policy_exports": [policy_export_metadata(policy) for policy in policies],
        "best_sessions": rank_sessions(all_rows)[:10] if all_rows else [],
        "hard_cases": collision_contexts(all_rows, max_cases=40),
        "windows_retest_target": WINDOWS_RETEST_TARGET,
    }
    out_path = LAB_ROOT / "tmp" / f"{run_id}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "ab_done", "summary_path": str(out_path)}), flush=True)
    print(json.dumps(output, indent=2, sort_keys=True))


def _tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(LAB_ROOT))
    except ValueError:
        return str(path)


def _nested(payload: object, *keys: str) -> object | None:
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _collision_context(row: dict) -> dict:
    keys = (
        "session",
        "config",
        "policy_label",
        "policy_path",
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


if __name__ == "__main__":
    main()
