"""
Drone RL Lab — Unified Dispatcher
===================================
Routes experiment configs to the correct backend trainer.

Usage:
    python train.py configs/exp_NNN.yaml

Backends:
    hover   — gym-pybullet-drones HoverAviary + SB3 PPO (default)
    racing  — lsy_drone_racing + CleanRL PPO
"""

import sys
import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py configs/exp_NNN.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    backend = config.get("backend", "hover")

    if backend == "hover":
        from train_hover import run
    elif backend == "racing":
        from train_racing import run
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'hover' or 'racing'.")
        sys.exit(1)

    print(f"[Dispatcher] Backend: {backend}")
    run(config_path)


if __name__ == "__main__":
    main()
