#!/usr/bin/env python3
"""Export an AI-GP replay session into synchronized PPO actor features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_gp_rl.session_dataset import export_session_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Path to one replay/sessions/<run_id> directory")
    parser.add_argument("--output", help="Optional output JSONL path")
    parser.add_argument("--max-telemetry-age", type=float, default=0.10)
    args = parser.parse_args()
    summary = export_session_dataset(
        args.session_dir,
        args.output,
        max_telemetry_age_s=args.max_telemetry_age,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
