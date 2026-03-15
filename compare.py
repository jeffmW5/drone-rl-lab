"""
Drone RL Lab — Experiment Leaderboard
======================================
Scans results/*/metrics.json and prints a sorted comparison table.

Usage:
    python compare.py                      # all experiments
    python compare.py --backend hover      # hover experiments only
    python compare.py --backend racing     # racing experiments only
    python compare.py --json               # output as JSON
"""

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
                experiments.append(data)

    return experiments


def print_table(experiments: list, backend_filter: str = None):
    """Print a formatted leaderboard sorted by mean_reward."""
    if backend_filter:
        experiments = [e for e in experiments if e.get("backend", "hover") == backend_filter]

    if not experiments:
        print(f"No experiments found" +
              (f" for backend '{backend_filter}'" if backend_filter else " in results/"))
        return

    # Sort by mean_reward descending
    experiments.sort(key=lambda x: x.get("mean_reward", 0), reverse=True)

    # Header
    print(f"\n{'='*90}")
    title = "DRONE RL LAB — EXPERIMENT LEADERBOARD"
    if backend_filter:
        title += f" ({backend_filter})"
    print(f"  {title}")
    print(f"{'='*90}\n")

    # Table header
    print(f"{'Rank':<5} {'Experiment':<30} {'Backend':<8} {'Reward':>10} {'Std':>8} "
          f"{'Steps':>10} {'Time':>8}")
    print(f"{'-'*5} {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    best_reward = experiments[0].get("mean_reward", 0)

    for i, exp in enumerate(experiments):
        name = exp.get("experiment", exp.get("_folder", "???"))
        backend = exp.get("backend", "hover")
        reward = exp.get("mean_reward", 0)
        std = exp.get("std_reward", 0)
        steps = exp.get("timesteps_trained", 0)
        wall = exp.get("elapsed_seconds", 0)

        # Mark the best
        marker = " *" if reward == best_reward else ""
        delta = f"({reward - best_reward:+.1f})" if reward != best_reward else "(best)"

        print(f"{i+1:<5} {name:<30} {backend:<8} {reward:>10.3f} {std:>8.3f} "
              f"{steps:>10,} {wall:>7.0f}s  {delta}{marker}")

    print(f"\n* = current best\n")

    # Summary stats
    rewards = [e.get("mean_reward", 0) for e in experiments]
    backends = set(e.get("backend", "hover") for e in experiments)
    print(f"  Experiments run: {len(experiments)}")
    print(f"  Backends:        {', '.join(sorted(backends))}")
    print(f"  Best reward:     {max(rewards):.3f}")
    print(f"  Worst reward:    {min(rewards):.3f}")
    print(f"  Range:           {max(rewards) - min(rewards):.3f}")
    print()


def main():
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    experiments = load_all_metrics(results_dir)

    # Parse --backend filter
    backend_filter = None
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            backend_filter = sys.argv[i + 1]

    if "--json" in sys.argv:
        if backend_filter:
            experiments = [e for e in experiments if e.get("backend", "hover") == backend_filter]
        print(json.dumps(experiments, indent=2))
    else:
        print_table(experiments, backend_filter=backend_filter)


if __name__ == "__main__":
    main()
