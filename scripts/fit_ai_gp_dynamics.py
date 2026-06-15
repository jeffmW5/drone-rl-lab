#!/usr/bin/env python3
"""Fit and validate the measured AI-GP surrogate dynamics profile."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.dynamics_fit import build_fit_report, write_fit_report


TRANSLATION_TRAINING_IDS = (
    "cal_high_launch_20260609_190404_t030000",
    "cal_high_launch_20260609_190404_t035000",
    "cal_high_launch_20260609_190404_t040000",
)
ROLL_TRAINING_IDS = tuple(
    f"cal_roll_sign_20260614_094920_{sign}_{trial:02d}_r030000"
    for sign in ("positive", "negative")
    for trial in range(1, 4)
)
PITCH_TRAINING_IDS = TRANSLATION_TRAINING_IDS
HELD_OUT_CALIBRATION_IDS = (
    "cal_high_launch_20260609_190602_t032500",
    "cal_high_launch_20260609_190650_t031250",
)
HELD_OUT_POLICY_IDS = (
    "full_policy_ramped_launch_batch14_01",
    "full_policy_ramped_launch_batch14_02",
    "full_policy_ramped_launch_batch14_03",
)


def default_sessions_root() -> Path:
    configured = os.environ.get("AI_GP_SESSIONS_ROOT")
    if configured:
        return Path(configured)
    return Path(
        "/media/drone-rl-lab/tmp/ai-grand-prix-stack-remote/"
        "replay/sessions"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=default_sessions_root(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "ai_gp_dynamics_fit_20260614"
        / "fit_report.json",
    )
    parser.add_argument("--fit-duration-s", type=float, default=3.0)
    args = parser.parse_args()

    report = build_fit_report(
        sessions_root=args.sessions_root,
        translation_training_ids=TRANSLATION_TRAINING_IDS,
        rate_roll_training_ids=ROLL_TRAINING_IDS,
        rate_pitch_training_ids=PITCH_TRAINING_IDS,
        held_out_calibration_ids=HELD_OUT_CALIBRATION_IDS,
        held_out_policy_ids=HELD_OUT_POLICY_IDS,
        fit_duration_s=args.fit_duration_s,
    )
    write_fit_report(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
