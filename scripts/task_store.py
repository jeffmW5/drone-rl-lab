#!/usr/bin/env python3
"""
Typed task store backed by JSON files in inbox/tasks/.

This module is the programmatic interface for reading and writing task
artifacts. It coexists with the legacy Markdown queue in inbox/INBOX.md:

- JSON tasks in inbox/tasks/ are the source of truth for new tasks.
- The Markdown INBOX continues to work for legacy tasks.
- render_inbox.py can regenerate INBOX.md from JSON + legacy entries.

Usage from other scripts:
    from task_store import TaskStore
    store = TaskStore()
    task = store.load("exp_066")
    store.claim("exp_066", agent_id="my-agent")
    store.complete("exp_066", result_summary="MIXED -- 2/15 det gates")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_DIR / "inbox" / "tasks"
SCHEMA_PATH = REPO_DIR / "schemas" / "task.schema.json"

VALID_STATUSES = {"ready", "claimed", "in_progress", "done", "blocked", "deferred", "failed"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStore:
    """Read/write typed task artifacts."""

    def __init__(self, tasks_dir: Path | None = None):
        self.tasks_dir = tasks_dir or TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def exists(self, task_id: str) -> bool:
        return self._path(task_id).is_file()

    def load(self, task_id: str) -> dict | None:
        path = self._path(task_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def save(self, task: dict) -> Path:
        """Write a task dict to disk. Updates updated_at automatically."""
        task["updated_at"] = _now_iso()
        path = self._path(task["task_id"])
        path.write_text(json.dumps(task, indent=2) + "\n")
        return path

    def list_all(self) -> list[dict]:
        """Load all task artifacts, sorted by priority then task_id."""
        tasks = []
        for f in sorted(self.tasks_dir.glob("*.json")):
            try:
                tasks.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        tasks.sort(key=lambda t: (t.get("priority", 999), t.get("task_id", "")))
        return tasks

    def list_by_status(self, *statuses: str) -> list[dict]:
        return [t for t in self.list_all() if t.get("status") in statuses]

    def get_next(self) -> dict | None:
        """Return the highest-priority ready task with no unmet dependencies."""
        done_ids = {t["task_id"] for t in self.list_by_status("done")}
        for task in self.list_by_status("ready"):
            dep = task.get("depends_on")
            if dep and dep not in done_ids:
                continue
            return task
        return None

    def claim(self, task_id: str, agent_id: str) -> dict | None:
        """Claim a task. Returns the updated task or None if not claimable."""
        task = self.load(task_id)
        if task is None or task["status"] != "ready":
            return None
        task["status"] = "claimed"
        task["claimed_by"] = agent_id
        self.save(task)
        return task

    def start(self, task_id: str) -> dict | None:
        """Move a claimed task to in_progress."""
        task = self.load(task_id)
        if task is None or task["status"] != "claimed":
            return None
        task["status"] = "in_progress"
        self.save(task)
        return task

    def complete(
        self,
        task_id: str,
        result_summary: str | None = None,
        result_diagnosis: str | None = None,
        result_path: str | None = None,
    ) -> dict | None:
        """Mark a task as done."""
        task = self.load(task_id)
        if task is None:
            return None
        task["status"] = "done"
        task["completed_at"] = _now_iso()
        if result_summary is not None:
            task["result_summary"] = result_summary
        if result_diagnosis is not None:
            task["result_diagnosis"] = result_diagnosis
        if result_path is not None:
            task["result_path"] = result_path
        self.save(task)
        return task

    def fail(self, task_id: str, reason: str | None = None) -> dict | None:
        """Mark a task as failed."""
        task = self.load(task_id)
        if task is None:
            return None
        task["status"] = "failed"
        task["completed_at"] = _now_iso()
        if reason:
            task["result_diagnosis"] = reason
        self.save(task)
        return task

    def release(self, task_id: str) -> dict | None:
        """Release a claimed task back to ready."""
        task = self.load(task_id)
        if task is None or task["status"] not in ("claimed", "in_progress"):
            return None
        task["status"] = "ready"
        task["claimed_by"] = None
        self.save(task)
        return task
