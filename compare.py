"""
Drone RL Lab — Experiment Leaderboard
======================================
Scans results/*/metrics.json and prints a sorted comparison table.

Usage:
    python compare.py                      # all experiments
    python compare.py --backend hover      # hover experiments only
    python compare.py --backend racing     # racing experiments only
    python compare.py --json               # output as JSON
    python compare.py --csv                # output as CSV
    python compare.py --filter level=level2  # filter by key=value
    python compare.py --generate-log       # auto-generate memory/EXPERIMENT_LOG.md
"""

import argparse
import csv
import io
import os
import json
import sys


def load_all_metrics(results_dir: str) -> list:
    """Load metrics.json from every experiment folder."""
    experiments = []
    if not os.path.isdir(results_dir):
        return experiments

    for folder in sorted(os.listdir(results_dir)):
        metrics_path = os.path.join(results_dir, folder, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                data["_folder"] = folder

            # Load benchmark.json if it exists
            bench_path = os.path.join(results_dir, folder, "benchmark.json")
            if os.path.isfile(bench_path):
                with open(bench_path, "r") as f:
                    data["_benchmark"] = json.load(f)

            experiments.append(data)

    return experiments


def apply_filters(experiments: list, backend: str = None, filters: list = None) -> list:
    """Apply backend and key=value filters."""
    if backend:
        experiments = [e for e in experiments if e.get("backend", "hover") == backend]

    if filters:
        for filt in filters:
            if "=" not in filt:
                print(f"Warning: ignoring malformed filter '{filt}' (expected key=value)",
                      file=sys.stderr)
                continue
            key, val = filt.split("=", 1)
            experiments = [
                e for e in experiments
                if str(e.get(key, "")).lower() == val.lower()
                or str(e.get("racing_kwargs", {}).get(key, "")).lower() == val.lower()
                or str(e.get("ppo_kwargs", {}).get(key, "")).lower() == val.lower()
            ]

    return experiments


def get_benchmark_summary(exp: dict) -> dict:
    """Extract benchmark summary from experiment data."""
    bench = exp.get("_benchmark") or exp.get("benchmark")
    if not bench:
        return {}

    results = {}
    benchmarks = bench.get("benchmarks", [])
    for b in benchmarks:
        level = b.get("level", "?")
        results[level] = {
            "finish_rate": b.get("finish_rate", 0),
            "avg_time": b.get("avg_time", 0),
            "avg_gates": b.get("avg_gates", 0),
            "avg_finish_time": b.get("avg_finish_time"),
        }
    return results


def print_table(experiments: list):
    """Print a formatted leaderboard sorted by mean_reward."""
    if not experiments:
        print("No experiments found matching filters.")
        return

    # Sort by mean_reward descending
    experiments.sort(key=lambda x: x.get("mean_reward", 0), reverse=True)

    # Check if any have benchmark data
    has_bench = any(get_benchmark_summary(e) for e in experiments)

    # Header
    print(f"\n{'='*110}")
    print(f"  DRONE RL LAB — EXPERIMENT LEADERBOARD")
    print(f"{'='*110}\n")

    # Table header
    hdr = (f"{'Rank':<5} {'Experiment':<30} {'Backend':<8} {'Level':<7} "
           f"{'Reward':>10} {'Std':>8} {'Steps':>10} {'Time':>8}")
    sep = (f"{'-'*5} {'-'*30} {'-'*8} {'-'*7} "
           f"{'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    if has_bench:
        hdr += f"  {'Finish%':>7} {'AvgLap':>7} {'Gates':>5}"
        sep += f"  {'-'*7} {'-'*7} {'-'*5}"
    print(hdr)
    print(sep)

    best_reward = experiments[0].get("mean_reward", 0)

    for i, exp in enumerate(experiments):
        name = exp.get("experiment", exp.get("_folder", "???"))
        backend = exp.get("backend", "hover")
        level = exp.get("level", "--")
        reward = exp.get("mean_reward", 0)
        std = exp.get("std_reward", 0)
        steps = exp.get("timesteps_trained", 0)
        wall = exp.get("elapsed_seconds", 0)

        # Mark the best
        delta = f"({reward - best_reward:+.1f})" if reward != best_reward else "(best)"

        line = (f"{i+1:<5} {name:<30} {backend:<8} {level:<7} "
                f"{reward:>10.3f} {std:>8.3f} {steps:>10,} {wall:>7.0f}s  {delta}")

        if has_bench:
            bench = get_benchmark_summary(exp)
            if bench:
                # Show the primary (trained) level, or first available
                lvl = exp.get("level", "")
                b = bench.get(lvl) or next(iter(bench.values()), {})
                fr = b.get("finish_rate", 0)
                at = b.get("avg_time", 0)
                ag = b.get("avg_gates", 0)
                line += f"  {fr:>6.0%} {at:>6.1f}s {ag:>5.1f}"
            else:
                line += f"  {'--':>7} {'--':>7} {'--':>5}"

        print(line)

    print()

    # Summary stats
    rewards = [e.get("mean_reward", 0) for e in experiments]
    backends = set(e.get("backend", "hover") for e in experiments)
    print(f"  Experiments: {len(experiments)}")
    print(f"  Backends:    {', '.join(sorted(backends))}")
    print(f"  Best reward: {max(rewards):.3f}")
    print(f"  Worst:       {min(rewards):.3f}")
    print()


def print_csv(experiments: list):
    """Print CSV output with all key fields including benchmark data."""
    if not experiments:
        return

    experiments.sort(key=lambda x: x.get("mean_reward", 0), reverse=True)

    fields = [
        "experiment", "backend", "level", "mean_reward", "std_reward",
        "timesteps_trained", "elapsed_seconds", "hypothesis",
        "bench_finish_rate", "bench_avg_time", "bench_avg_gates",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    for exp in experiments:
        row = {
            "experiment": exp.get("experiment", exp.get("_folder", "")),
            "backend": exp.get("backend", "hover"),
            "level": exp.get("level", ""),
            "mean_reward": exp.get("mean_reward", ""),
            "std_reward": exp.get("std_reward", ""),
            "timesteps_trained": exp.get("timesteps_trained", ""),
            "elapsed_seconds": exp.get("elapsed_seconds", ""),
            "hypothesis": exp.get("hypothesis", ""),
        }
        bench = get_benchmark_summary(exp)
        if bench:
            lvl = exp.get("level", "")
            b = bench.get(lvl) or next(iter(bench.values()), {})
            row["bench_finish_rate"] = b.get("finish_rate", "")
            row["bench_avg_time"] = b.get("avg_time", "")
            row["bench_avg_gates"] = b.get("avg_gates", "")

        writer.writerow(row)

    print(buf.getvalue(), end="")


def generate_log(experiments: list, lab_dir: str):
    """Auto-generate memory/EXPERIMENT_LOG.md from metrics + benchmark data."""
    if not experiments:
        print("No experiments found — skipping log generation.")
        return

    experiments.sort(key=lambda x: x.get("experiment", x.get("_folder", "")))

    lines = [
        "# Experiment Log",
        "",
        "> Auto-generated by `python compare.py --generate-log`. Do not edit manually.",
        "> To update: run `python compare.py --generate-log` after adding new results.",
        "",
        "| Exp | Backend | Level | Reward | Steps | Finish% | Avg Lap | Gates | Hypothesis |",
        "|-----|---------|-------|--------|-------|---------|---------|-------|------------|",
    ]

    for exp in experiments:
        name = exp.get("experiment", exp.get("_folder", ""))
        # Extract short exp number
        short = name.replace("exp_", "").split("_")[0] if name.startswith("exp_") else name
        backend = exp.get("backend", "hover")
        level = exp.get("level", "--")
        reward = exp.get("mean_reward", 0)
        steps = exp.get("timesteps_trained", 0)

        # Benchmark data
        bench = get_benchmark_summary(exp)
        if bench:
            lvl = exp.get("level", "")
            b = bench.get(lvl) or next(iter(bench.values()), {})
            fr = f"{b.get('finish_rate', 0):.0%}"
            at = f"{b.get('avg_time', 0):.1f}s" if b.get("avg_time") else "--"
            ag = f"{b.get('avg_gates', 0):.1f}"
        else:
            fr = "--"
            at = "--"
            ag = "--"

        # Hypothesis (truncated)
        hyp = exp.get("hypothesis", "")
        if len(hyp) > 60:
            hyp = hyp[:57] + "..."

        # Summary override from metrics.json if present
        summary = exp.get("summary", {})
        if summary.get("key_change"):
            hyp = summary["key_change"]

        steps_str = f"{steps:,}" if steps else "--"

        lines.append(
            f"| {short} | {backend} | {level} | {reward:.2f} | {steps_str} "
            f"| {fr} | {at} | {ag} | {hyp} |"
        )

    lines.append("")

    log_path = os.path.join(lab_dir, "memory", "EXPERIMENT_LOG.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Generated] {log_path} ({len(experiments)} experiments)")


def main():
    parser = argparse.ArgumentParser(description="Drone RL Lab experiment leaderboard")
    parser.add_argument("--backend", "-b", help="Filter by backend (hover/racing)")
    parser.add_argument("--filter", "-f", action="append", default=[],
                        help="Filter by key=value (repeatable)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--generate-log", action="store_true",
                        help="Auto-generate memory/EXPERIMENT_LOG.md")
    args = parser.parse_args()

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    experiments = load_all_metrics(results_dir)
    experiments = apply_filters(experiments, backend=args.backend, filters=args.filter)

    if args.generate_log:
        generate_log(experiments, lab_dir)
    elif args.json:
        print(json.dumps(experiments, indent=2))
    elif args.csv:
        print_csv(experiments)
    else:
        print_table(experiments)


if __name__ == "__main__":
    main()
