#!/usr/bin/env python3
"""
Job state store for drone-rl-lab.

Provides read/query access to persisted job states in state/jobs/.
Used by task_store for failure-aware task selection and by run_job for
persistence.

Usage:
    from job_store import JobStore
    store = JobStore()
    jobs = store.list_for_task("exp_066")
    failures = store.recent_failures("exp_066")
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_DIR / "state" / "jobs"

# A task with this many consecutive failures should not be auto-claimed
MAX_CONSECUTIVE_FAILURES = 2


class JobStore:
    """Read-only queries over persisted job state files."""

    def __init__(self, jobs_dir: Path | None = None):
        self.jobs_dir = jobs_dir or JOBS_DIR

    def _load(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def list_all(self) -> list[dict]:
        """Load all job states, sorted by created_at."""
        if not self.jobs_dir.is_dir():
            return []
        jobs = []
        for f in sorted(self.jobs_dir.glob("*.json")):
            job = self._load(f)
            if job:
                jobs.append(job)
        jobs.sort(key=lambda j: j.get("created_at", ""))
        return jobs

    def list_for_task(self, task_id: str) -> list[dict]:
        """List all jobs for a given task, sorted by attempt."""
        return [j for j in self.list_all() if j.get("task_id") == task_id]

    def latest_for_task(self, task_id: str) -> dict | None:
        """Return the most recent job for a task."""
        jobs = self.list_for_task(task_id)
        return jobs[-1] if jobs else None

    def recent_failures(self, task_id: str) -> list[dict]:
        """Return recent consecutive failures for a task (newest first)."""
        jobs = self.list_for_task(task_id)
        failures = []
        for job in reversed(jobs):
            if job.get("status") == "failed":
                failures.append(job)
            else:
                break  # stop at first non-failure
        return failures

    def consecutive_failure_count(self, task_id: str) -> int:
        """Count consecutive recent failures for a task."""
        return len(self.recent_failures(task_id))

    def has_repeated_failures(self, task_id: str, threshold: int = MAX_CONSECUTIVE_FAILURES) -> bool:
        """Check if a task has too many consecutive failures."""
        return self.consecutive_failure_count(task_id) >= threshold

    def task_ids_with_repeated_failures(self, threshold: int = MAX_CONSECUTIVE_FAILURES) -> set[str]:
        """Return task IDs that have too many consecutive failures."""
        # Group by task_id
        by_task: dict[str, list[dict]] = {}
        for job in self.list_all():
            tid = job.get("task_id")
            if tid:
                by_task.setdefault(tid, []).append(job)

        blocked = set()
        for tid, jobs in by_task.items():
            jobs.sort(key=lambda j: j.get("created_at", ""))
            consecutive = 0
            for job in reversed(jobs):
                if job.get("status") == "failed":
                    consecutive += 1
                else:
                    break
            if consecutive >= threshold:
                blocked.add(tid)
        return blocked
