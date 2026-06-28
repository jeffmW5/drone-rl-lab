from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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
            "gate0_pass_rate": summary["gate0_pass_count"] / max(run_count, 1),
            "collision_run_rate": summary["collision_run_count"] / max(run_count, 1),
            "passes_windows_retest_target": (
                summary["gate0_pass_count"] / max(run_count, 1) >= 0.90
                and float(summary["mean_max_gate"]) > 2.0
                and int(summary["best_max_gate"]) >= 3
            ),
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
        "run_id": run_id,
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
        "policies": ranked_policies,
        "best_sessions": rank_sessions(all_rows)[:10] if all_rows else [],
        "windows_retest_target": {
            "gate0_pass_rate": ">= 0.90",
            "mean_max_gate": "> 2.0",
            "best_max_gate": ">= 3",
        },
    }
    out_path = LAB_ROOT / "tmp" / f"{run_id}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "ab_done", "summary_path": str(out_path)}), flush=True)
    print(json.dumps(output, indent=2, sort_keys=True))


def _tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "")


if __name__ == "__main__":
    main()
