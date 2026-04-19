#!/usr/bin/env python3
"""
Capture reproducibility metadata for a completed experiment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"


def _detect_lsy_dir() -> Path:
    override = os.environ.get("DRONE_RL_LSY_DIR")
    candidates = [
        override,
        "/root/lsy_drone_racing",
        "/media/lsy_drone_racing",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path("/media/lsy_drone_racing")


def _pixi_version() -> str:
    override = os.environ.get("DRONE_RL_PIXI_BIN")
    candidates = [
        override,
        shutil.which("pixi"),
        "/root/.pixi/bin/pixi",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return _command_output([candidate, "--version"])
    return ""


def _command_output(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_info(repo_dir: Path) -> dict:
    if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
        return {"path": str(repo_dir), "available": False}
    return {
        "path": str(repo_dir),
        "available": True,
        "branch": _command_output(["git", "branch", "--show-current"], cwd=repo_dir),
        "commit": _command_output(["git", "rev-parse", "HEAD"], cwd=repo_dir),
        "dirty": bool(_command_output(["git", "status", "--short"], cwd=repo_dir)),
    }


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _gpu_info() -> list[dict]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        devices.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_mb": int(parts[2]),
            }
        )
    return devices


def capture_provenance(experiment: str) -> dict:
    exp_dir = RESULTS_DIR / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "repo": _git_info(LAB_DIR),
        "external_repos": {
            "lsy_drone_racing": _git_info(_detect_lsy_dir()),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pixi_version": _pixi_version(),
            "module_versions": {
                "numpy": _module_version("numpy"),
                "torch": _module_version("torch"),
                "jax": _module_version("jax"),
                "stable_baselines3": _module_version("stable_baselines3"),
                "yaml": _module_version("yaml"),
            },
            "gpus": _gpu_info(),
        },
        "environment": {
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "drone_rl_lsy_dir": os.environ.get("DRONE_RL_LSY_DIR"),
            "drone_rl_pixi_bin": os.environ.get("DRONE_RL_PIXI_BIN"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Write results/<exp>/provenance.json")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment folder name")
    args = parser.parse_args()

    provenance = capture_provenance(args.experiment)
    output_path = RESULTS_DIR / args.experiment / "provenance.json"
    output_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[provenance] Wrote {output_path}")


if __name__ == "__main__":
    main()
