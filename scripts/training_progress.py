#!/usr/bin/env python3
"""
Lightweight training progress monitor for drone-rl-lab.

Supports:
- local log files
- RunPod logs over SSH, using the same local key/pod-id conventions as manage_pod.sh

Typical usage:
    python3 scripts/training_progress.py --remote --latest
    python3 scripts/training_progress.py --remote --experiment exp_071_obs_normalization
    python3 scripts/training_progress.py --log /root/drone-rl-lab/logs/exp_071_....log --remote
    python3 scripts/training_progress.py --log results/exp_001_baseline/train.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_REMOTE_LOG_DIR = "/root/drone-rl-lab/logs"
DEFAULT_POD_ID_FILE = Path.home() / ".config" / "drone-rl-lab" / "runpod_pod_id"
SPINNER_FRAMES = "|/-\\"
SPARKLINE = " .:-=+*#%@"
NON_TRAINING_LOG_MARKERS = ("throughput", "benchmark_runner", "runner.log", "after_exp")

ITER_RE = re.compile(
    r"iter\s+(?P<iteration>\d+)/(?:\s*)?(?P<total_iterations>\d+)\s+\|\s+"
    r"step\s+(?P<step>[\d,]+)\s+\|\s+"
    r"reward\s+(?P<reward>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"pg_loss\s+(?P<pg_loss>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"v_loss\s+(?P<v_loss>-?\d+(?:\.\d+)?)\s+\|\s+"
    r"time\s+(?P<elapsed>\d+)s"
)


@dataclass
class ProgressState:
    experiment: Optional[str] = None
    budget_seconds: Optional[int] = None
    iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    step: Optional[int] = None
    reward: Optional[float] = None
    pg_loss: Optional[float] = None
    v_loss: Optional[float] = None
    elapsed_seconds: Optional[int] = None
    training_started: bool = False
    completed: bool = False
    status: str = "waiting_for_logs"
    recent_rewards: list[float] | None = None
    log_age_seconds: Optional[float] = None
    log_size_bytes: Optional[int] = None
    process_running: Optional[bool] = None
    process_summary: Optional[str] = None
    pod_id: Optional[str] = None
    ssh_target: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor training progress from lab logs.")
    parser.add_argument("--log", help="Path to a local or remote log file.")
    parser.add_argument("--remote", action="store_true", help="Read log from the current RunPod pod over SSH.")
    parser.add_argument("--latest", action="store_true", help="Pick the newest matching log file.")
    parser.add_argument("--experiment", help="Experiment name used to find a matching log.")
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--tail-lines", type=int, default=200, help="How many lines to fetch on each refresh.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    return parser.parse_args()


def detect_deploy_key() -> Optional[str]:
    override = os.environ.get("DRONE_RL_DEPLOY_KEY")
    candidates = [
        override,
        str(Path.home() / ".ssh" / "id_ed25519_runpod"),
        str(Path.home() / ".ssh" / "id_ed25519"),
        "/media/id_ed25519_runpod",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.R_OK):
            return candidate
    return None


def get_pod_id() -> str:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    pod_file = Path(os.environ.get("DRONE_RL_RUNPOD_POD_ID_FILE", DEFAULT_POD_ID_FILE))
    if not pod_id and pod_file.is_file():
        pod_id = pod_file.read_text().strip()
    if pod_id:
        return pod_id
    return get_latest_running_pod_id()


def runpod_query(query: str) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is not set.")
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-d",
            json.dumps({"query": query}),
            "https://api.runpod.io/graphql",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "RunPod API request failed.")
    return json.loads(proc.stdout)


def resolve_active_pod() -> tuple[str, dict]:
    pod_id = get_pod_id()
    pod = lookup_pod(pod_id)
    if not pod:
        fallback_id = get_latest_running_pod_id()
        pod = lookup_pod(fallback_id)
        if not pod:
            raise RuntimeError(f"RunPod pod '{pod_id}' was not found, and no running pod fallback was available.")
        pod_id = fallback_id
    return pod_id, pod


def get_pod_ssh_target() -> tuple[str, str, str]:
    pod_id, pod = resolve_active_pod()
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("type") == "tcp":
            return pod_id, str(port["ip"]), str(port["publicPort"])
    raise RuntimeError(f"Pod '{pod_id}' has no public SSH endpoint yet.")


def lookup_pod(pod_id: str) -> Optional[dict]:
    data = runpod_query(
        f'{{ pod(input: {{podId: "{pod_id}"}}) {{ id desiredStatus runtime {{ ports {{ ip privatePort publicPort type }} }} }} }}'
    )
    return (data.get("data") or {}).get("pod")


def get_latest_running_pod_id() -> str:
    data = runpod_query("{ myself { pods { id desiredStatus } } }")
    pods = ((data.get("data") or {}).get("myself") or {}).get("pods") or []
    for pod in reversed(pods):
        if pod.get("desiredStatus") == "RUNNING":
            return str(pod["id"])
    raise RuntimeError(
        "No RunPod pod id found and no running pod was discoverable. "
        "Set RUNPOD_POD_ID or populate ~/.config/drone-rl-lab/runpod_pod_id."
    )


def run_ssh(command: str) -> str:
    key = detect_deploy_key()
    if not key:
        raise RuntimeError("No readable SSH key found for pod access.")
    _, host, port = get_pod_ssh_target()
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            key,
            "-p",
            port,
            f"root@{host}",
            command,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "SSH command failed.")
    return proc.stdout


def find_remote_log(experiment: Optional[str], latest: bool, explicit_log: Optional[str]) -> str:
    if explicit_log:
        return explicit_log
    patterns = []
    if experiment:
        task_prefix = experiment.split("_", 2)
        short_prefix = "_".join(task_prefix[:2]) if len(task_prefix) >= 2 else experiment
        patterns.extend(
            [
                f"{DEFAULT_REMOTE_LOG_DIR}/{experiment}*.log",
                f"{DEFAULT_REMOTE_LOG_DIR}/*{experiment}*.log",
                f"{DEFAULT_REMOTE_LOG_DIR}/{short_prefix}*.log",
                f"{DEFAULT_REMOTE_LOG_DIR}/*{short_prefix}*.log",
            ]
        )
    patterns.append(f"{DEFAULT_REMOTE_LOG_DIR}/*.log")

    if latest:
        cmd = (
            "python3 - <<'PY'\n"
            "import glob, os\n"
            f"patterns = {repr(patterns)}\n"
            f"bad = {repr(NON_TRAINING_LOG_MARKERS)}\n"
            "paths = []\n"
            "for pat in patterns:\n"
            "    paths.extend(glob.glob(pat))\n"
            "paths = [p for p in set(paths) if not any(marker in os.path.basename(p) for marker in bad)]\n"
            "paths = sorted(paths, key=os.path.getmtime, reverse=True)\n"
            "print(paths[0] if paths else '')\n"
            "PY"
        )
        log_path = run_ssh(cmd).strip()
        if log_path:
            return log_path
    for pattern in patterns:
        cmd = (
            "python3 - <<'PY'\n"
            "import glob, os\n"
            f"pattern = {pattern!r}\n"
            f"bad = {repr(NON_TRAINING_LOG_MARKERS)}\n"
            "paths = [p for p in glob.glob(pattern) if not any(marker in os.path.basename(p) for marker in bad)]\n"
            "paths.sort()\n"
            "print(paths[0] if paths else '')\n"
            "PY"
        )
        log_path = run_ssh(cmd).strip()
        if log_path:
            return log_path
    raise RuntimeError(f"No remote log found for experiment: {experiment or '(latest)'}")


def read_remote_log(log_path: str, tail_lines: int) -> str:
    return run_ssh(f"tail -n {tail_lines} {log_path}")


def read_local_log(log_path: str, tail_lines: int) -> str:
    path = Path(log_path)
    if not path.is_file():
        raise RuntimeError(f"Local log not found: {log_path}")
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-tail_lines:])


def parse_progress(text: str) -> ProgressState:
    state = ProgressState()
    recent_rewards: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "EXPERIMENT:" in stripped:
            state.experiment = stripped.split("EXPERIMENT:", 1)[1].strip()
        elif "BUDGET:" in stripped:
            match = re.search(r"(\d+)s", stripped)
            if match:
                state.budget_seconds = int(match.group(1))
        elif "[Training] Starting CleanRL PPO loop..." in stripped:
            state.training_started = True
            state.status = "training"
        elif "[TimeBudget]" in stripped:
            state.completed = True
            state.status = "time_budget_reached"
        elif stripped.startswith("RESULTS") or stripped.startswith("[Saved]"):
            state.completed = True
            state.status = "completed"

        match = ITER_RE.search(stripped)
        if match:
            state.iteration = int(match.group("iteration"))
            state.total_iterations = int(match.group("total_iterations"))
            state.step = int(match.group("step").replace(",", ""))
            state.reward = float(match.group("reward"))
            state.pg_loss = float(match.group("pg_loss"))
            state.v_loss = float(match.group("v_loss"))
            state.elapsed_seconds = int(match.group("elapsed"))
            state.training_started = True
            state.status = "training"
            recent_rewards.append(state.reward)

    if state.status == "waiting_for_logs" and state.experiment:
        state.status = "starting_up"
    state.recent_rewards = recent_rewards[-12:]
    return state


def format_seconds(seconds: Optional[int]) -> str:
    if seconds is None:
        return "--"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}h {mins:02d}m {secs:02d}s"
    if mins:
        return f"{mins:d}m {secs:02d}s"
    return f"{secs:d}s"


def render_bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(width * frac)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def sparkline(values: list[float] | None) -> str:
    if not values:
        return "--"
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return SPARKLINE[len(SPARKLINE) // 2] * len(values)
    chars = []
    for value in values:
        idx = int((value - lo) / (hi - lo) * (len(SPARKLINE) - 1))
        chars.append(SPARKLINE[idx])
    return "".join(chars)


def render(state: ProgressState, log_path: str, frame_index: int) -> str:
    cols = shutil.get_terminal_size((100, 20)).columns
    bar_width = max(20, min(48, cols - 40))
    spinner = SPINNER_FRAMES[frame_index % len(SPINNER_FRAMES)]

    frac_iter = (
        state.iteration / state.total_iterations
        if state.iteration is not None and state.total_iterations
        else None
    )
    frac_time = (
        state.elapsed_seconds / state.budget_seconds
        if state.elapsed_seconds is not None and state.budget_seconds
        else None
    )
    frac = frac_iter if frac_iter is not None else (frac_time if frac_time is not None else 0.0)
    pct = f"{100 * frac:5.1f}%"
    eta = (
        int((state.elapsed_seconds / frac_iter) - state.elapsed_seconds)
        if state.elapsed_seconds is not None and frac_iter and frac_iter > 0
        else None
    )
    freshness = (
        f"{state.log_age_seconds:.0f}s old" if state.log_age_seconds is not None else "--"
    )
    stale = (
        state.log_age_seconds is not None
        and state.process_running
        and state.log_age_seconds > 180
    )
    status = f"{spinner} {state.status}"
    if stale:
        status += " (stale log warning)"

    lines = [
        f"Training Monitor  status={status}",
        f"experiment: {state.experiment or '--'}",
        f"pod:        {state.pod_id or '--'}  ssh={state.ssh_target or '--'}",
        f"log:        {log_path}",
        f"log_age:    {freshness}  size={state.log_size_bytes or '--'} bytes",
        f"process:    {state.process_summary or '--'}",
        f"progress:   {render_bar(frac, bar_width)} {pct}",
        f"iter:       {state.iteration or '--'} / {state.total_iterations or '--'}",
        f"steps:      {state.step or '--'}",
        f"elapsed:    {format_seconds(state.elapsed_seconds)} / {format_seconds(state.budget_seconds)}  eta={format_seconds(eta)}",
        f"reward:     {state.reward if state.reward is not None else '--'}",
        f"pg_loss:    {state.pg_loss if state.pg_loss is not None else '--'}",
        f"v_loss:     {state.v_loss if state.v_loss is not None else '--'}",
        f"trend:      {sparkline(state.recent_rewards)}",
    ]
    return "\n".join(lines)


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def enrich_remote_state(state: ProgressState, experiment: Optional[str], log_path: str) -> None:
    pod_id, host, port = get_pod_ssh_target()
    state.pod_id = pod_id
    state.ssh_target = f"{host}:{port}"

    task_prefix = None
    if experiment:
        parts = experiment.split("_", 2)
        task_prefix = "_".join(parts[:2]) if len(parts) >= 2 else experiment

    remote_cmd = f"""
python3 - <<'PY'
import os, time, subprocess, json
log_path = {json.dumps(log_path)}
prefix = {json.dumps(task_prefix)}
out = {{}}
try:
    st = os.stat(log_path)
    out["log_age_seconds"] = max(0.0, time.time() - st.st_mtime)
    out["log_size_bytes"] = st.st_size
except FileNotFoundError:
    pass
pattern = prefix if prefix else "train.py /root/drone-rl-lab/configs/"
proc = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
out["process_running"] = bool(lines)
if lines:
    out["process_summary"] = lines[0][:160]
print(json.dumps(out))
PY
"""
    raw = run_ssh(remote_cmd)
    data = json.loads(raw.strip() or "{}")
    state.log_age_seconds = data.get("log_age_seconds")
    state.log_size_bytes = data.get("log_size_bytes")
    state.process_running = data.get("process_running")
    state.process_summary = data.get("process_summary")


def main() -> int:
    args = parse_args()

    if args.remote:
        resolver = lambda: find_remote_log(args.experiment, True if not args.log else args.latest, args.log)
        reader = lambda p: read_remote_log(p, args.tail_lines)
    else:
        if not args.log:
            raise SystemExit("--log is required unless --remote is used.")
        log_path_local = args.log
        resolver = lambda: log_path_local
        reader = lambda p: read_local_log(p, args.tail_lines)

    last_render = None
    log_path = None
    frame_index = 0

    while True:
        try:
            log_path = resolver()
            text = reader(log_path)
            state = parse_progress(text)
            if args.remote:
                enrich_remote_state(state, args.experiment, log_path)
            current = render(state, log_path, frame_index)
        except Exception as exc:
            current = f"Training Monitor  status=error\nerror: {exc}"
            state = None

        if args.once:
            print(current)
            return 0

        clear_screen()
        print(current)
        last_render = current

        if state and state.completed:
            return 0

        frame_index += 1
        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
