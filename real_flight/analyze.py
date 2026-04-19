#!/usr/bin/env python3
"""Analyze real flight logs and compare against simulation.

Usage:
    python analyze.py real_flight/logs/flight_20260419_143000.npz
    python analyze.py real_flight/logs/flight_*.npz --compare  # overlay multiple flights
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_flight(path: str) -> dict:
    data = dict(np.load(path, allow_pickle=True))
    print(f"Loaded {path}: {len(data['t'])} samples, {data['t'][-1]:.1f}s duration")
    return data


def print_summary(data: dict, label: str = ""):
    t = data["t"]
    pos = data["pos"]
    vel = data.get("vel", None)
    action = data.get("action", None)

    print(f"\n{'=' * 50}")
    if label:
        print(f"  {label}")
    print(f"  Duration: {t[-1]:.2f}s ({len(t)} samples)")
    print(f"  Position range:")
    print(f"    X: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]")
    print(f"    Y: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}]")
    print(f"    Z: [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")
    if vel is not None:
        speed = np.linalg.norm(vel, axis=-1)
        print(f"  Speed: mean={speed.mean():.2f} max={speed.max():.2f} m/s")
    if action is not None:
        print(f"  Actions (mean abs):")
        labels = ["roll", "pitch", "yaw", "thrust"]
        for i, lbl in enumerate(labels):
            print(f"    {lbl}: mean={action[:, i].mean():.3f} std={action[:, i].std():.3f}")
    print(f"{'=' * 50}")


def plot_flight(data: dict, title: str = ""):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plotting: pip install matplotlib")
        return

    t = data["t"]
    pos = data["pos"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title or "Flight Analysis", fontsize=14)

    # 3D trajectory (projected to 2D views)
    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], "b-", linewidth=0.8, alpha=0.7)
    ax.plot(pos[0, 0], pos[0, 1], "go", markersize=8, label="start")
    ax.plot(pos[-1, 0], pos[-1, 1], "rs", markersize=8, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position vs time
    ax = axes[0, 1]
    for i, (lbl, c) in enumerate(zip(["X", "Y", "Z"], ["r", "g", "b"])):
        ax.plot(t, pos[:, i], c, label=lbl, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[1, 0]
    if "vel" in data:
        vel = data["vel"]
        speed = np.linalg.norm(vel, axis=-1)
        ax.plot(t, speed, "k-", linewidth=0.8, label="speed")
        for i, (lbl, c) in enumerate(zip(["vx", "vy", "vz"], ["r", "g", "b"])):
            ax.plot(t, vel[:, i], c, linewidth=0.5, alpha=0.5, label=lbl)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Velocity")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Actions
    ax = axes[1, 1]
    if "action" in data:
        action = data["action"]
        labels = ["roll", "pitch", "yaw", "thrust"]
        for i, lbl in enumerate(labels):
            ax.plot(t[: len(action)], action[:, i], linewidth=0.8, label=lbl)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Action (normalized)")
        ax.set_title("Policy Actions")
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(title.replace(" ", "_") + ".png") if title else Path("flight_analysis.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Path to flight .npz log files")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    for path in args.logs:
        data = load_flight(path)
        print_summary(data, label=Path(path).stem)
        if args.plot:
            plot_flight(data, title=Path(path).stem)


if __name__ == "__main__":
    main()
