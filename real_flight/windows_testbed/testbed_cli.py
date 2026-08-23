"""AI Deck WiFi Test Bed -- console front end.

Same four tests as the GUI, for headless use or when tkinter is unavailable.

  py -3 testbed_cli.py link
  py -3 testbed_cli.py packet --duration 60 --max-frames 0
  py -3 testbed_cli.py reconnect --attempts 12
  py -3 testbed_cli.py throughput --throughput-duration 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aideck_core as core
import aideck_tests as tests


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Deck WiFi test bed (console)")
    parser.add_argument("test", choices=sorted(tests.TESTS))
    parser.add_argument("--ip", action="append", default=None,
                        help="deck IP; repeat for multiple candidates")
    parser.add_argument("--port", type=int, default=core.DEFAULT_PORT)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=45.0,
                        help="packet test duration")
    parser.add_argument("--max-frames", type=int, default=2,
                        help="packet test frame cap; 0 = unlimited")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--throughput-duration", type=float, default=120.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--stall-gap", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=core.DEFAULT_OUT_DIR)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    cfg = tests.TestConfig(
        ips=args.ip or [core.DEFAULT_IPS[0], core.DEFAULT_IPS[1]],
        port=args.port,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        duration=args.duration,
        max_frames=args.max_frames,
        attempts=args.attempts,
        delay=args.delay,
        throughput_duration=args.throughput_duration,
        save_every=args.save_every,
        stall_gap_s=args.stall_gap,
    )

    label, func, prefix = tests.TESTS[args.test]
    run_dir = core.new_run_dir(args.out_dir, prefix)
    cancel = core.CancelToken()

    print(f"{label} -> {run_dir}")
    summary = None
    try:
        with core.Reporter(run_dir, prefix, on_line=print) as reporter:
            summary = func(cfg, run_dir, reporter, cancel)
    except KeyboardInterrupt:
        cancel.cancel()
        print("\nInterrupted.")

    if not args.no_zip:
        try:
            print(f"Zipped: {core.zip_run_dir(run_dir).name}")
        except OSError as exc:
            print(f"Zip failed: {exc!r}")

    print(f"Artifacts: {run_dir}")
    if summary is None:
        return 1
    if args.test == "link":
        return 0 if summary.get("ok") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
