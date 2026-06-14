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


GAINS = (1.25, 1.50, 2.00)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"thrust_gain_matrix_{timestamp}"
    policy_path = (
        EXECUTION_ROOT
        / "policy"
        / "models"
        / "ai_gp_017_motion_safety_governed.json"
    )
    attempts: list[dict[str, object]] = []
    started_s = time.monotonic()

    for gain in GAINS:
        label = f"gain{int(round(gain * 100)):03d}"
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
                launch_phase_s=2.0,
                authority_ramp_s=1.0,
                post_thrust_gain=gain,
            )
            attempts.append(
                {
                    "gain": gain,
                    "repeat": repeat,
                    **summary,
                }
            )
            print(
                f"{run_id}: gate={summary['max_active_gate_index']} "
                f"collision={summary['collision_count']} "
                f"abort={summary['abort_reason']}"
            )

    groups = {}
    for gain in GAINS:
        selected = [item for item in attempts if item["gain"] == gain]
        groups[str(gain)] = {
            "attempt_count": len(selected),
            "gate0_pass_count": sum(
                bool(item["gate0_passed"]) for item in selected
            ),
            "race_finish_count": sum(
                bool(item["race_finished"]) for item in selected
            ),
            "collision_attempt_count": sum(
                int(item["collision_count"]) > 0 for item in selected
            ),
            "max_gate_index": max(
                int(item["max_active_gate_index"]) for item in selected
            ),
            "abort_reasons": [item["abort_reason"] for item in selected],
        }

    report = {
        "run_prefix": prefix,
        "attempt_count": len(attempts),
        "elapsed_s": time.monotonic() - started_s,
        "launch_phase_s": 2.0,
        "authority_ramp_s": 1.0,
        "direction_ungoverned_fraction": 0.0,
        "groups": groups,
        "attempts": attempts,
    }
    report_path = (
        STACK_ROOT / "replay" / "sessions" / f"{prefix}_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(groups, indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
