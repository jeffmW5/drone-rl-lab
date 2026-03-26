#!/usr/bin/env python3
"""
Build a machine-readable snapshot of the lab's current state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from task_queue import get_next_task, parse_tasks

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
STATE_DIR = LAB_DIR / "state"
STATE_PATH = STATE_DIR / "current.json"
INBOX_PATH = LAB_DIR / "inbox" / "INBOX.md"
AGENTS_DIR = LAB_DIR / "agents"


def _git_output(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_info(repo_dir: Path) -> dict:
    if not (repo_dir / ".git").exists():
        return {"path": str(repo_dir), "available": False}
    return {
        "path": str(repo_dir),
        "available": True,
        "branch": _git_output(repo_dir, "branch", "--show-current"),
        "commit": _git_output(repo_dir, "rev-parse", "HEAD"),
        "dirty": bool(_git_output(repo_dir, "status", "--short")),
    }


def _load_experiments(excluded_ids: set[str] | None = None) -> list[dict]:
    experiments = []
    excluded_ids = {item.lower() for item in (excluded_ids or set())}
    if not RESULTS_DIR.is_dir():
        return experiments

    for folder in sorted(RESULTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        metrics_path = folder / "metrics.json"
        if not metrics_path.is_file():
            continue

        data = json.loads(metrics_path.read_text())
        experiment_id = str(data.get("experiment") or folder.name).lower()
        if any(
            experiment_id == excluded_id or experiment_id.startswith(f"{excluded_id}_")
            for excluded_id in excluded_ids
        ):
            continue
        benchmark_path = folder / "benchmark.json"
        if benchmark_path.is_file():
            data["benchmark"] = json.loads(benchmark_path.read_text())
        experiments.append(data)

    return experiments


def _benchmark_entry(exp: dict) -> dict | None:
    benchmark = exp.get("benchmark") or {}
    entries = benchmark.get("benchmarks", [])
    if not entries:
        return None

    level = exp.get("level")
    for entry in entries:
        if entry.get("level") == level:
            return entry
    return entries[0]


def _highest_training_reward(experiments: list[dict]) -> dict | None:
    candidates = [
        exp for exp in experiments
        if exp.get("backend") == "racing" and exp.get("level") == "level2"
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda exp: exp.get("mean_reward", float("-inf")))
    return {
        "experiment": best.get("experiment"),
        "mean_reward": best.get("mean_reward"),
        "timesteps_trained": best.get("timesteps_trained"),
        "timestamp": best.get("timestamp"),
    }


def _latest_result(experiments: list[dict]) -> dict | None:
    timestamped = [exp for exp in experiments if exp.get("timestamp")]
    if not timestamped:
        return None

    latest = max(timestamped, key=lambda exp: exp.get("timestamp", ""))
    return {
        "experiment": latest.get("experiment"),
        "timestamp": latest.get("timestamp"),
        "mean_reward": latest.get("mean_reward"),
        "timesteps_trained": latest.get("timesteps_trained"),
    }


def _best_benchmark(experiments: list[dict]) -> dict | None:
    ranked = []
    for exp in experiments:
        entry = _benchmark_entry(exp)
        if not entry:
            continue
        avg_time = entry.get("avg_time")
        ranked.append(
            (
                entry.get("finish_rate", 0.0),
                entry.get("avg_gates", 0.0),
                -(avg_time if avg_time is not None else 1e9),
                exp,
                entry,
            )
        )

    if not ranked:
        return None

    _, _, _, exp, entry = max(ranked)
    return {
        "experiment": exp.get("experiment"),
        "level": entry.get("level"),
        "finish_rate": entry.get("finish_rate"),
        "avg_gates": entry.get("avg_gates"),
        "avg_time": entry.get("avg_time"),
        "avg_finish_time": entry.get("avg_finish_time"),
    }


def _active_agents() -> list[dict]:
    agents = []
    if not AGENTS_DIR.is_dir():
        return agents

    for agent_file in sorted(AGENTS_DIR.glob("*.json")):
        try:
            data = json.loads(agent_file.read_text())
        except json.JSONDecodeError:
            continue
        agents.append(
            {
                "id": data.get("id"),
                "status": data.get("status"),
                "task": data.get("task"),
                "heartbeat": data.get("heartbeat"),
            }
        )
    return agents


def _non_done_queue_experiment_ids(tasks: list[dict]) -> set[str]:
    ids = set()
    for task in tasks:
        if task["status_kind"] == "done":
            continue
        task_id = task.get("task_id")
        if task_id:
            ids.add(task_id.lower())
    return ids


def build_state() -> dict:
    tasks = parse_tasks(INBOX_PATH)
    next_task = get_next_task(tasks)
    excluded_ids = _non_done_queue_experiment_ids(tasks)
    experiments = _load_experiments(excluded_ids)
    running_ids = {
        task["task_id"]
        for task in tasks
        if task["status_kind"] in ("claimed", "in_progress") and task.get("task_id")
    }

    queue_summary = {
        "counts": {},
        "running": [],
        "actionable": [],
        "next_task": {k: v for k, v in next_task.items() if not k.startswith("_")} if next_task else None,
    }
    for task in tasks:
        kind = task["status_kind"]
        queue_summary["counts"][kind] = queue_summary["counts"].get(kind, 0) + 1
        public_task = {k: v for k, v in task.items() if not k.startswith("_")}
        if kind in ("claimed", "in_progress"):
            queue_summary["running"].append(public_task)
        elif kind in ("next", "ready", "implemented", "queued"):
            if task.get("task_id") and task["task_id"] in running_ids:
                continue
            queue_summary["actionable"].append(public_task)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": _git_info(LAB_DIR),
        "external_repos": {
            "lsy_drone_racing": _git_info(Path("/media/lsy_drone_racing")),
        },
        "agents": _active_agents(),
        "queue": queue_summary,
        "evaluation": {
            "racing_primary_metric": "benchmark",
            "benchmark_ranking": ["finish_rate", "avg_gates", "avg_time"],
            "training_reward_note": (
                "Training reward is secondary and only comparable within similar reward definitions."
            ),
        },
        "experiments": {
            "total_results": len(experiments),
            "highest_recorded_mean_reward": _highest_training_reward(experiments),
            "best_benchmark": _best_benchmark(experiments),
            "latest_result": _latest_result(experiments),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Write state/current.json")
    parser.add_argument("--output", default=str(STATE_PATH), help="Output JSON path")
    parser.add_argument("--print", action="store_true", dest="print_json", help="Print JSON to stdout")
    args = parser.parse_args()

    state = build_state()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state, indent=2) + "\n")

    if args.print_json:
        print(json.dumps(state, indent=2))
    else:
        print(f"[state] Wrote {output_path}")


if __name__ == "__main__":
    main()
