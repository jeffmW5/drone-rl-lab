#!/usr/bin/env python3
"""
Drone RL Lab -- Structured Benchmark Runner
=============================================
Runs sim.py in the lsy_drone_racing pixi environment and parses output
into structured JSON results.

Usage:
    python scripts/benchmark.py --experiment exp_020_gpu_gate_traj_long --n_runs 5
    python scripts/benchmark.py -e exp_021_smooth_traj --level level2 --n_runs 10
    python scripts/benchmark.py -e exp_020_gpu_gate_traj_long --controller attitude_rl_generic.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

LSY_DIR = "/media/lsy_drone_racing"
CONTROL_DIR = os.path.join(LSY_DIR, "lsy_drone_racing", "control")
LAB_DIR = str(Path(__file__).resolve().parent.parent)

LEVEL_CONFIGS = {
    "level0": "level0_attitude.toml",
    "level1": "level1_attitude.toml",
    "level2": "level2_attitude.toml",
}


def find_controller(experiment_name: str) -> str | None:
    """Find the controller file for an experiment."""
    # Try: attitude_rl_{full_name}.py
    candidate = f"attitude_rl_{experiment_name}.py"
    if os.path.isfile(os.path.join(CONTROL_DIR, candidate)):
        return candidate

    # Try: attitude_rl_expNNN.py from exp_NNN_description
    parts = experiment_name.split("_")
    if len(parts) >= 2:
        short = f"attitude_rl_{parts[0]}{parts[1]}.py"
        if os.path.isfile(os.path.join(CONTROL_DIR, short)):
            return short

    # Try generic
    generic = "attitude_rl_generic.py"
    if os.path.isfile(os.path.join(CONTROL_DIR, generic)):
        return generic

    return None


def run_benchmark(level: str, controller: str, n_runs: int, timeout: int = 300) -> list[dict]:
    """Run sim.py via pixi and parse results."""
    config = LEVEL_CONFIGS.get(level)
    if not config:
        raise ValueError(f"Unknown level: {level}. Known: {list(LEVEL_CONFIGS.keys())}")

    env = os.environ.copy()
    env["SCIPY_ARRAY_API"] = "1"

    cmd = [
        "pixi", "run", "python", "scripts/sim.py",
        "--config", config,
        "--controller", controller,
        "--n_runs", str(n_runs),
    ]

    result = subprocess.run(
        cmd, cwd=LSY_DIR,
        capture_output=True, text=True, timeout=timeout,
        env=env,
    )

    output = result.stderr + "\n" + result.stdout

    # Parse: Flight time (s): X.XX\nFinished: True/False\nGates passed: N
    runs = []
    flight_times = re.findall(r"Flight time \(s\): ([\d.]+)", output)
    finishes = re.findall(r"Finished: (True|False)", output)
    gates = re.findall(r"Gates passed: (\d+)", output)

    for i in range(min(len(flight_times), len(finishes), len(gates))):
        runs.append({
            "time": float(flight_times[i]),
            "finished": finishes[i] == "True",
            "gates": int(gates[i]),
        })

    return runs


def benchmark_experiment(
    experiment: str,
    levels: list[str],
    n_runs: int,
    controller: str | None = None,
) -> dict:
    """Run full benchmark for an experiment across levels."""
    if controller is None:
        controller = find_controller(experiment)
        if controller is None:
            raise FileNotFoundError(
                f"No controller found for {experiment}. "
                f"Create {CONTROL_DIR}/attitude_rl_{experiment}.py "
                f"or use --controller attitude_rl_generic.py"
            )

    benchmarks = []
    for level in levels:
        print(f"[Benchmark] {level} x {n_runs} runs with {controller}...")
        try:
            runs = run_benchmark(level, controller, n_runs)
        except subprocess.TimeoutExpired:
            print(f"[Benchmark] TIMEOUT on {level} — skipping")
            continue
        except Exception as e:
            print(f"[Benchmark] ERROR on {level}: {e}")
            continue

        if not runs:
            print(f"[Benchmark] No results parsed for {level}")
            continue

        finished_runs = [r for r in runs if r["finished"]]
        entry = {
            "level": level,
            "controller": controller,
            "n_runs": len(runs),
            "results": runs,
            "avg_time": round(sum(r["time"] for r in runs) / len(runs), 2),
            "finish_rate": round(len(finished_runs) / len(runs), 2),
            "avg_gates": round(sum(r["gates"] for r in runs) / len(runs), 1),
        }
        if finished_runs:
            entry["avg_finish_time"] = round(
                sum(r["time"] for r in finished_runs) / len(finished_runs), 2
            )
        benchmarks.append(entry)

        # Print summary
        print(f"  -> {len(finished_runs)}/{len(runs)} finished, "
              f"avg gates: {entry['avg_gates']}, avg time: {entry['avg_time']}s")

    return {"experiment": experiment, "benchmarks": benchmarks}


def main():
    parser = argparse.ArgumentParser(description="Structured benchmark runner for drone-rl-lab")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment name")
    parser.add_argument("--controller", "-c", help="Controller filename (overrides auto-detect)")
    parser.add_argument("--level", "-l", action="append", help="Level(s) to benchmark (repeatable)")
    parser.add_argument("--n_runs", "-n", type=int, default=5, help="Runs per level")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    # Check prerequisites
    if not os.path.isdir(LSY_DIR):
        print(f"ERROR: lsy_drone_racing not found at {LSY_DIR}", file=sys.stderr)
        sys.exit(1)

    # Default levels: detect from metrics.json
    if args.level is None:
        metrics_path = os.path.join(LAB_DIR, "results", args.experiment, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            trained_level = metrics.get("level", "level0")
            args.level = [trained_level]
            if trained_level != "level2":
                args.level.append("level2")
        else:
            args.level = ["level0", "level2"]

    results = benchmark_experiment(args.experiment, args.level, args.n_runs, args.controller)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(LAB_DIR, "results", args.experiment, "benchmark.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {output_path}")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
