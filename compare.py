"""
Drone RL Lab — Experiment Leaderboard
======================================
Scans results/*/metrics.json and prints a sorted comparison table.

Usage:
    python compare.py              # all experiments
    python compare.py --json       # output as JSON
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


def print_table(experiments: list):
    """Print a formatted leaderboard sorted by mean_reward."""
    if not experiments:
        print("No experiments found in results/")
        return

    # Sort by mean_reward descending
    experiments.sort(key=lambda x: x.get("mean_reward", 0), reverse=True)

    # Header
    print(f"\n{'='*80}")
    print(f"  DRONE RL LAB — EXPERIMENT LEADERBOARD")
    print(f"{'='*80}\n")

    # Table header
    print(f"{'Rank':<5} {'Experiment':<30} {'Reward':>10} {'Std':>8} "
          f"{'Steps':>10} {'Time':>8}")
    print(f"{'-'*5} {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    best_reward = experiments[0].get("mean_reward", 0)

    for i, exp in enumerate(experiments):
        name = exp.get("experiment", exp.get("_folder", "???"))
        reward = exp.get("mean_reward", 0)
        std = exp.get("std_reward", 0)
        steps = exp.get("timesteps_trained", 0)
        wall = exp.get("elapsed_seconds", 0)

        # Mark the best
        marker = " *" if reward == best_reward else ""
        delta = f"({reward - best_reward:+.1f})" if reward != best_reward else "(best)"

        print(f"{i+1:<5} {name:<30} {reward:>10.3f} {std:>8.3f} "
              f"{steps:>10,} {wall:>7.0f}s  {delta}{marker}")

    print(f"\n* = current best\n")

    # Summary stats
    rewards = [e.get("mean_reward", 0) for e in experiments]
    print(f"  Experiments run: {len(experiments)}")
    print(f"  Best reward:     {max(rewards):.3f}")
    print(f"  Worst reward:    {min(rewards):.3f}")
    print(f"  Range:           {max(rewards) - min(rewards):.3f}")
    print()


def main():
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    experiments = load_all_metrics(results_dir)

    if "--json" in sys.argv:
        print(json.dumps(experiments, indent=2))
    else:
        print_table(experiments)


if __name__ == "__main__":
    main()
