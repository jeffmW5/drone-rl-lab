from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_ai_gp_bounded_windows import EXECUTION_ROOT, STACK_ROOT, run


CONFIGURATIONS = (
    ("thrust50", 0.5, 0.0),
    ("thrust100", 1.0, 0.0),
    ("direction50", 0.0, 0.5),
    ("direction100", 0.0, 1.0),
    ("both50", 0.5, 0.5),
    ("both100", 1.0, 1.0),
)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"authority_matrix_{timestamp}"
    policy_path = (
        EXECUTION_ROOT
        / "policy"
        / "models"
        / "ai_gp_017_motion_safety_governed.json"
    )
    attempts: list[dict[str, object]] = []
    started_s = time.monotonic()

    for label, thrust_fraction, direction_fraction in CONFIGURATIONS:
        for repeat in range(1, 4):
            run_id = f"{prefix}_{label}_{repeat:02d}"
            summary = run(
                policy_path=policy_path,
                run_id=run_id,
                duration_s=30.0,
                target_gates=0,
                control_rate_hz=12.5,
                lateral_authority_scale=1.0,
                pitch_authority_scale=1.0,
                governor_slew_scale=4.0,
                gate_source="track_pose",
                post_control_rate_hz=50.0,
                post_max_roll_rate_radps=0.2,
                post_max_pitch_rate_radps=0.2,
                post_max_yaw_rate_radps=0.15,
                post_thrust_span_up=0.02,
                post_thrust_span_down=0.15,
                uniform_authority=False,
                launch_phase_s=2.0,
                authority_ramp_s=1.0,
                thrust_ungoverned_fraction=thrust_fraction,
                direction_ungoverned_fraction=direction_fraction,
            )
            record = {
                "configuration": label,
                "repeat": repeat,
                "thrust_ungoverned_fraction": thrust_fraction,
                "direction_ungoverned_fraction": direction_fraction,
                **summary,
            }
            attempts.append(record)
            print(
                f"{run_id}: gate={summary['max_active_gate_index']} "
                f"collision={summary['collision_count']} "
                f"abort={summary['abort_reason']}"
            )

    configurations = {}
    for label, thrust_fraction, direction_fraction in CONFIGURATIONS:
        selected = [
            attempt
            for attempt in attempts
            if attempt["configuration"] == label
        ]
        configurations[label] = {
            "thrust_ungoverned_fraction": thrust_fraction,
            "direction_ungoverned_fraction": direction_fraction,
            "attempt_count": len(selected),
            "gate0_pass_count": sum(
                bool(attempt["gate0_passed"]) for attempt in selected
            ),
            "race_finish_count": sum(
                bool(attempt["race_finished"]) for attempt in selected
            ),
            "collision_attempt_count": sum(
                int(attempt["collision_count"]) > 0 for attempt in selected
            ),
            "max_gate_index": max(
                int(attempt["max_active_gate_index"]) for attempt in selected
            ),
            "abort_reasons": [
                attempt["abort_reason"] for attempt in selected
            ],
        }

    report = {
        "run_prefix": prefix,
        "attempt_count": len(attempts),
        "elapsed_s": time.monotonic() - started_s,
        "launch_phase_s": 2.0,
        "authority_ramp_s": 1.0,
        "configurations": configurations,
        "attempts": attempts,
    }
    report_path = (
        STACK_ROOT / "replay" / "sessions" / f"{prefix}_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["configurations"], indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
