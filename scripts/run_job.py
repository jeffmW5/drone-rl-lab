#!/usr/bin/env python3
"""
Resumable job runner for drone-rl-lab experiments.

Executes a task through a sequence of steps, persisting state after each step
so an interrupted run can resume from the last completed step.

Steps:
    claimed -> preparing -> training -> capturing_provenance
            -> benchmarking -> documenting -> publishing -> done

Usage:
    python scripts/run_job.py exp_066                    # run from JSON task
    python scripts/run_job.py exp_066 --resume           # resume interrupted job
    python scripts/run_job.py exp_066 --dry-run          # show steps without executing
    python scripts/run_job.py exp_066 --agent-id myagent # set agent identity
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from task_store import TaskStore

REPO_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_DIR / "state" / "jobs"
RUNS_DIR = REPO_DIR / "state" / "runs"

# Ordered pipeline steps
STEPS = [
    "claimed",
    "preparing",
    "training",
    "capturing_provenance",
    "benchmarking",
    "documenting",
    "publishing",
    "done",
]

# Terminal states
TERMINAL = {"done", "failed", "blocked"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


def _make_job_id(task_id: str, attempt: int = 1) -> str:
    return f"job_{task_id}_{attempt:03d}"


class JobRunner:
    """Manages lifecycle of a single job execution."""

    def __init__(self, task_id: str, agent_id: str | None = None, dry_run: bool = False):
        self.task_id = task_id
        self.agent_id = agent_id
        self.dry_run = dry_run
        self.store = TaskStore()
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return JOBS_DIR / f"{job_id}.json"

    def _find_active_job(self) -> dict | None:
        """Find an existing non-terminal job for this task."""
        for f in sorted(JOBS_DIR.glob(f"job_{self.task_id}_*.json"), reverse=True):
            try:
                job = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if job.get("status") not in TERMINAL:
                return job
        return None

    def _save_job(self, job: dict):
        job["updated_at"] = _now_iso()
        job["heartbeat"] = _now_iso()
        path = self._job_path(job["job_id"])
        path.write_text(json.dumps(job, indent=2) + "\n")

    def _log_step(self, job: dict, step: str, status: str, detail: str = ""):
        entry = {"step": step, "status": status, "timestamp": _now_iso()}
        if detail:
            entry["detail"] = detail
        job.setdefault("step_log", []).append(entry)

    def _next_step(self, current_step: str) -> str | None:
        """Return the step after current, or None if at the end."""
        try:
            idx = STEPS.index(current_step)
        except ValueError:
            return None
        if idx + 1 >= len(STEPS):
            return None
        return STEPS[idx + 1]

    def _advance(self, job: dict, to_step: str, detail: str = ""):
        """Advance job to a new step."""
        self._log_step(job, to_step, "started", detail)
        job["step"] = to_step
        if to_step in TERMINAL:
            job["status"] = "completed" if to_step == "done" else to_step
        else:
            job["status"] = "running"
        self._save_job(job)

    def _complete_step(self, job: dict, step: str, detail: str = ""):
        """Mark the current step as completed in the log."""
        self._log_step(job, step, "completed", detail)
        self._save_job(job)

    def _fail(self, job: dict, error: str):
        """Mark job as failed."""
        job["step"] = "failed"
        job["status"] = "failed"
        job["last_error"] = error
        self._log_step(job, "failed", "failed", error)
        self._save_job(job)

    def create_job(self, task: dict) -> dict:
        """Create a new job for a task."""
        # Find the next attempt number
        existing = list(JOBS_DIR.glob(f"job_{self.task_id}_*.json"))
        attempt = len(existing) + 1

        job = {
            "job_id": _make_job_id(self.task_id, attempt),
            "task_id": self.task_id,
            "run_id": _make_run_id(),
            "attempt": attempt,
            "step": "claimed",
            "status": "pending",
            "agent_id": self.agent_id,
            "heartbeat": _now_iso(),
            "lease_expires_at": None,
            "resume_from_step": None,
            "last_error": None,
            "step_log": [],
            "artifacts": {},
            "config_snapshot": {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }

        # Snapshot config path
        config_path = task.get("config")
        if config_path:
            job["artifacts"]["config_path"] = config_path
            full_config = REPO_DIR / config_path
            if full_config.is_file():
                try:
                    import yaml
                    job["config_snapshot"] = yaml.safe_load(full_config.read_text()) or {}
                except Exception:
                    pass  # yaml not available or parse error, fine

        # Create run directory
        run_dir = RUNS_DIR / job["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        self._log_step(job, "claimed", "started", f"agent={self.agent_id}")
        self._save_job(job)
        return job

    # ─── Step executors ────────────────────────────────────────────────────

    def _exec_preparing(self, job: dict, task: dict):
        """Validate config exists and environment is ready."""
        config_path = task.get("config")
        if config_path:
            full_path = REPO_DIR / config_path
            if not full_path.is_file():
                self._fail(job, f"Config not found: {config_path}")
                return False
            job["artifacts"]["config_path"] = config_path
        return True

    def _exec_training(self, job: dict, task: dict):
        """Execute training. This is the long step."""
        config = task.get("config")
        if not config:
            self._fail(job, "No config specified in task")
            return False

        if self.dry_run:
            print(f"  [dry-run] Would run: python train.py {config}")
            return True

        result = subprocess.run(
            ["python3", "train.py", config],
            cwd=str(REPO_DIR),
            capture_output=False,
        )
        if result.returncode != 0:
            self._fail(job, f"Training failed with exit code {result.returncode}")
            return False

        # Record results dir
        task_id = task["task_id"]
        results_candidates = list((REPO_DIR / "results").glob(f"{task_id}*"))
        if results_candidates:
            job["artifacts"]["results_dir"] = str(results_candidates[-1].relative_to(REPO_DIR))
        return True

    def _exec_capturing_provenance(self, job: dict, task: dict):
        """Capture provenance metadata."""
        if self.dry_run:
            print(f"  [dry-run] Would run: python3 scripts/capture_provenance.py --experiment {self.task_id}")
            return True

        result = subprocess.run(
            ["python3", "scripts/capture_provenance.py", "--experiment", self.task_id],
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Non-fatal
            print(f"  [warn] Provenance capture failed: {result.stderr.strip()}")
        return True

    def _exec_benchmarking(self, job: dict, task: dict):
        """Run benchmark."""
        benchmark_script = REPO_DIR / "scripts" / "benchmark.py"
        if not benchmark_script.is_file():
            print("  [skip] No benchmark.py found")
            return True

        if self.dry_run:
            print(f"  [dry-run] Would run: python3 scripts/benchmark.py -e {self.task_id}")
            return True

        result = subprocess.run(
            ["python3", "scripts/benchmark.py", "-e", self.task_id, "-n", "5"],
            cwd=str(REPO_DIR),
            capture_output=False,
        )
        if result.returncode != 0:
            # Non-fatal for now
            print(f"  [warn] Benchmark failed with exit code {result.returncode}")
        else:
            results_dir = job.get("artifacts", {}).get("results_dir")
            if results_dir:
                job["artifacts"]["benchmark_path"] = f"{results_dir}/benchmark.json"
        return True

    def _exec_documenting(self, job: dict, task: dict):
        """Placeholder for documentation step. Future: auto-generate EXPERIMENT.md."""
        if self.dry_run:
            print(f"  [dry-run] Would generate documentation for {self.task_id}")
            return True
        # For now, this is a manual step or delegated to Claude
        print(f"  [info] Documentation step: write results/{self.task_id}*/EXPERIMENT.md")
        return True

    def _exec_publishing(self, job: dict, task: dict):
        """Refresh lab state and generate log."""
        if self.dry_run:
            print("  [dry-run] Would refresh lab state and generate log")
            return True

        subprocess.run(
            ["python3", "compare.py", "--generate-log"],
            cwd=str(REPO_DIR),
            capture_output=True,
        )
        subprocess.run(
            ["python3", "scripts/lab_state.py"],
            cwd=str(REPO_DIR),
            capture_output=True,
        )
        return True

    STEP_EXECUTORS = {
        "preparing": "_exec_preparing",
        "training": "_exec_training",
        "capturing_provenance": "_exec_capturing_provenance",
        "benchmarking": "_exec_benchmarking",
        "documenting": "_exec_documenting",
        "publishing": "_exec_publishing",
    }

    def run(self, resume: bool = False) -> dict:
        """Execute the full job pipeline, creating or resuming as needed."""
        task = self.store.load(self.task_id)
        if task is None:
            print(f"Task {self.task_id} not found in task store", file=sys.stderr)
            sys.exit(1)

        # Find or create job
        job = None
        if resume:
            job = self._find_active_job()
            if job:
                print(f"Resuming job {job['job_id']} from step '{job['step']}'")
            else:
                print(f"No active job found for {self.task_id}, creating new one")

        if job is None:
            job = self.create_job(task)
            print(f"Created job {job['job_id']}")

        # Determine starting step
        current = job["step"]
        if current in TERMINAL:
            print(f"Job {job['job_id']} is already {current}")
            return job

        # If resuming, start from the step after the last completed one
        if resume and current != "claimed":
            # current step was interrupted, re-run it
            start_step = current
        else:
            start_step = self._next_step(current) if current == "claimed" else current

        if start_step is None:
            self._advance(job, "done")
            return job

        # Walk through steps
        step = start_step
        while step and step not in TERMINAL:
            print(f"\n--- Step: {step} ---")
            self._advance(job, step)

            executor_name = self.STEP_EXECUTORS.get(step)
            if executor_name:
                executor = getattr(self, executor_name)
                ok = executor(job, task)
                if not ok:
                    print(f"Step {step} failed. Job state saved.")
                    return job
                self._complete_step(job, step)
            else:
                self._complete_step(job, step, "no executor, auto-complete")

            step = self._next_step(step)

        # All steps done
        if step == "done":
            self._advance(job, "done", "all steps completed")
            self.store.complete(self.task_id)
            print(f"\nJob {job['job_id']} completed successfully.")

        return job


def main():
    parser = argparse.ArgumentParser(description="Resumable job runner")
    parser.add_argument("task_id", help="Task ID to execute (e.g. exp_066)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted job")
    parser.add_argument("--dry-run", action="store_true", help="Show steps without executing")
    parser.add_argument("--agent-id", default=None, help="Agent identity")
    args = parser.parse_args()

    runner = JobRunner(
        task_id=args.task_id,
        agent_id=args.agent_id,
        dry_run=args.dry_run,
    )
    job = runner.run(resume=args.resume)

    # Print final state
    print(f"\nFinal state: step={job['step']} status={job['status']}")
    if job.get("last_error"):
        print(f"Last error: {job['last_error']}")


if __name__ == "__main__":
    main()
