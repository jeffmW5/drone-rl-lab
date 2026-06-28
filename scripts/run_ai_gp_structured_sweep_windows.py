from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from scripts.analyze_ai_gp_structured_runs import analyze_session, rank_sessions, summarize_group


SESSION_ROOT = LAB_ROOT / "tmp" / "ai-grand-prix-stack-remote" / "replay" / "sessions"


DEFAULT_SWEEP = (
    (1.08, 1.75, 1.0, 1.75),
    (1.08, 1.75, 1.0, 2.00),
    (1.08, 2.00, 1.0, 1.75),
    (1.08, 2.00, 1.0, 2.00),
    (1.10, 1.75, 1.0, 1.75),
    (1.10, 1.75, 1.0, 2.00),
    (1.10, 2.00, 1.0, 1.75),
    (1.10, 2.00, 1.0, 2.00),
    (1.12, 1.75, 1.0, 1.75),
    (1.12, 1.75, 1.0, 2.00),
    (1.12, 2.00, 1.0, 1.75),
    (1.12, 2.00, 1.0, 2.00),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts-per-config", type=int, default=5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sleep-between-configs-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.attempts_per_config < 1:
        raise ValueError("--attempts-per-config must be positive")
    run_id = args.run_id or "structured_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []
    config_summaries = []
    for index, (thrust, roll, pitch, yaw) in enumerate(DEFAULT_SWEEP, start=1):
        config_id = (
            f"{run_id}_c{index:02d}_t{_tag(thrust)}_r{_tag(roll)}_"
            f"p{_tag(pitch)}_y{_tag(yaw)}"
        )
        command = [
            args.python,
            "-u",
            "-B",
            str(LAB_ROOT / "scripts" / "run_ai_gp_structured_windows.py"),
            "--attempts",
            str(args.attempts_per_config),
            "--duration",
            str(args.duration),
            "--target-gates",
            "0",
            "--allow-gate-plane-miss",
            "--control-rate-hz",
            str(args.control_rate_hz),
            "--thrust-multiplier",
            str(thrust),
            "--roll-rate-multiplier",
            str(roll),
            "--pitch-rate-multiplier",
            str(pitch),
            "--yaw-rate-multiplier",
            str(yaw),
            "--run-id",
            config_id,
        ]
        print(json.dumps({"event": "config_start", "config": config_id}), flush=True)
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
                        "event": "config_failed",
                        "config": config_id,
                        "returncode": completed.returncode,
                        "output_tail": completed.stdout[-2000:],
                    }
                ),
                flush=True,
            )
            continue
        rows = []
        for attempt in range(1, args.attempts_per_config + 1):
            session = SESSION_ROOT / f"{config_id}_{attempt:02d}"
            if session.exists():
                row = analyze_session(session)
                row["config"] = config_id
                rows.append(row)
                all_rows.append(row)
        config_summary = summarize_group(rows)
        config_summary.update(
            {
                "config": config_id,
                "thrust": thrust,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }
        )
        config_summaries.append(config_summary)
        print(json.dumps({"event": "config_done", **config_summary}), flush=True)
        if args.sleep_between_configs_s > 0.0:
            time.sleep(args.sleep_between_configs_s)

    output = {
        "run_id": run_id,
        "best_sessions": rank_sessions(all_rows)[:10],
        "configs": sorted(
            config_summaries,
            key=lambda item: (
                item["best_max_gate"],
                item["race_finish_count"],
                -item["collision_run_count"],
                item["mean_policy_steps"],
            ),
            reverse=True,
        ),
    }
    out_path = LAB_ROOT / "tmp" / f"{run_id}_summary.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "sweep_done", "summary_path": str(out_path)}), flush=True)
    print(json.dumps(output, indent=2, sort_keys=True))


def _tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "")


if __name__ == "__main__":
    main()
