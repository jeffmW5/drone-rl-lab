"""
Drone RL Lab — Training Curve Plotter
=======================================
Generates plots from experiment data:
  1. Reward vs timesteps (all experiments overlaid)
  2. Per-step distance/velocity profiles (if steps.csv exists)

Usage:
    python plot.py                          # all experiments
    python plot.py exp_001 exp_005          # specific experiments
    python plot.py --steps exp_001          # per-step detail for one experiment

Output: results/comparison_plot.png (and results/steps_plot.png if --steps)
"""

import os
import sys
import json
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (works without display)
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_evaluations(results_dir: str, exp_name: str):
    """Load evaluations.npz for an experiment."""
    npz_path = os.path.join(results_dir, exp_name, "evaluations.npz")
    if not os.path.isfile(npz_path):
        return None, None
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    # evaluations shape: (n_evals, n_eval_episodes)
    mean_rewards = data["results"].mean(axis=1)
    return timesteps, mean_rewards


def load_steps_csv(results_dir: str, exp_name: str):
    """Load steps.csv for an experiment."""
    csv_path = os.path.join(results_dir, exp_name, "steps.csv")
    if not os.path.isfile(csv_path):
        return None
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestep": int(row["timestep"]),
                "distance": float(row["distance"]),
                "velocity": float(row["velocity"]),
                "z_pos": float(row["z_pos"]),
            })
    return rows


def plot_reward_curves(results_dir: str, experiment_names: list, output_path: str):
    """Plot reward vs timesteps for multiple experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(experiment_names))))

    for i, name in enumerate(experiment_names):
        timesteps, rewards = load_evaluations(results_dir, name)
        if timesteps is None:
            print(f"  [skip] {name} — no evaluations.npz")
            continue

        # Load metrics for label
        metrics_path = os.path.join(results_dir, name, "metrics.json")
        label = name
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
                label = f"{name} (best={m.get('mean_reward', '?')})"

        ax.plot(timesteps, rewards, label=label, color=colors[i], linewidth=1.5)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Eval Reward", fontsize=12)
    ax.set_title("Drone RL Lab — Training Curves", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Saved] {output_path}")
    plt.close()


def plot_steps_detail(results_dir: str, exp_name: str, output_path: str):
    """Plot per-step distance and velocity for a single experiment."""
    rows = load_steps_csv(results_dir, exp_name)
    if rows is None:
        print(f"  [skip] {exp_name} — no steps.csv (run with v2 train_rl.py to generate)")
        return

    timesteps = [r["timestep"] for r in rows]
    distances = [r["distance"] for r in rows]
    velocities = [r["velocity"] for r in rows]
    z_positions = [r["z_pos"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Distance from target
    axes[0].plot(timesteps, distances, linewidth=0.5, alpha=0.7, color="blue")
    axes[0].set_ylabel("Distance from target (m)")
    axes[0].set_title(f"{exp_name} — Per-Step Metrics")
    axes[0].grid(True, alpha=0.3)

    # Velocity
    axes[1].plot(timesteps, velocities, linewidth=0.5, alpha=0.7, color="red")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)

    # Z position (should converge to 1.0)
    axes[2].plot(timesteps, z_positions, linewidth=0.5, alpha=0.7, color="green")
    axes[2].axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="target z=1.0")
    axes[2].set_ylabel("Z position (m)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Saved] {output_path}")
    plt.close()


def main():
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")

    # Parse args
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_steps = "--steps" in sys.argv

    # Find experiment names
    if args:
        experiment_names = args
    else:
        experiment_names = sorted([
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and d != ".gitkeep"
        ])

    if not experiment_names:
        print("No experiments found in results/")
        return

    print(f"\nPlotting {len(experiment_names)} experiments: {', '.join(experiment_names)}\n")

    # Reward curves
    reward_plot_path = os.path.join(results_dir, "comparison_plot.png")
    plot_reward_curves(results_dir, experiment_names, reward_plot_path)

    # Per-step detail
    if do_steps:
        for name in experiment_names:
            steps_plot_path = os.path.join(results_dir, f"{name}_steps.png")
            plot_steps_detail(results_dir, name, steps_plot_path)
    else:
        # Check if any have steps.csv and mention it
        has_steps = [
            n for n in experiment_names
            if os.path.isfile(os.path.join(results_dir, n, "steps.csv"))
        ]
        if has_steps:
            print(f"Tip: {len(has_steps)} experiments have per-step data. "
                  f"Run: python plot.py --steps {has_steps[0]}")


if __name__ == "__main__":
    main()
