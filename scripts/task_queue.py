#!/usr/bin/env python3
"""
Shared queue parsing and mutation helpers for inbox/INBOX.md.

Supports both the original queue format:
  - [NEXT] / [QUEUED] / [DONE]

And the current workflow format:
  - [IMPLEMENTED ...] / [READY] / [IN PROGRESS] / [DONE ...] / [NOTE]
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

TASK_BLOCK_PATTERN = re.compile(
    r"^(###\s+)\[([^\]]+)\]\s+(.+?)$(.*?)(?=^###|\Z)",
    re.MULTILINE | re.DOTALL,
)
META_PATTERN = re.compile(r"^- \*\*(.+?)\*\*:\s*(.+?)\s*$")
EXPERIMENT_PATTERN = re.compile(r"\b(exp_[0-9]+[a-z]?(?:_[a-z0-9]+)*)\b", re.IGNORECASE)

ACTIONABLE_KINDS = ("next", "ready", "implemented", "queued")
RUNNING_KINDS = ("claimed", "in_progress")


def normalize_status(status_raw: str) -> str:
    """Map a raw heading status to a normalized kind."""
    upper = status_raw.upper()
    if status_raw.startswith("CLAIMED:"):
        return "claimed"
    if upper.startswith("DONE"):
        return "done"
    if upper.startswith("IN PROGRESS"):
        return "in_progress"
    if upper.startswith("IMPLEMENTED"):
        return "implemented"
    if upper.startswith("NOTE"):
        return "note"
    if upper.startswith("BLOCKED"):
        return "blocked"
    if upper == "NEXT":
        return "next"
    if upper == "QUEUED":
        return "queued"
    if upper == "READY":
        return "ready"
    return upper.lower().replace(" ", "_")


def claimed_agent_id(status_raw: str) -> str | None:
    if not status_raw.startswith("CLAIMED:"):
        return None
    return status_raw.split(":", 1)[1]


def _extract_task_id(title: str, metadata: dict[str, str]) -> str | None:
    for key in ("experiment", "config"):
        value = metadata.get(key)
        if not value:
            continue
        match = EXPERIMENT_PATTERN.search(value)
        if match:
            return match.group(1).lower()

    match = EXPERIMENT_PATTERN.search(title)
    if match:
        return match.group(1).lower()
    return None


def _task_public_view(task: dict) -> dict:
    return {k: v for k, v in task.items() if not k.startswith("_")}


def parse_tasks(inbox_path: Path) -> list[dict]:
    """Parse task blocks from the markdown inbox."""
    if not inbox_path.is_file():
        return []

    text = inbox_path.read_text()
    tasks = []

    for match in TASK_BLOCK_PATTERN.finditer(text):
        status_raw = match.group(2).strip()
        title = match.group(3).strip()
        body = match.group(4).strip()

        metadata = {}
        for line in body.splitlines():
            meta_match = META_PATTERN.match(line.strip())
            if not meta_match:
                continue
            key = meta_match.group(1).strip().lower().replace(" ", "_")
            metadata[key] = meta_match.group(2).strip()

        task = {
            "status": status_raw,
            "status_kind": normalize_status(status_raw),
            "title": title,
            "body": body,
            "task_id": _extract_task_id(title, metadata),
            "_block_start": match.start(),
            "_block_end": match.end(),
            "_status_start": match.start(2),
            "_status_end": match.end(2),
        }
        task.update(metadata)
        tasks.append(task)

    return tasks


def get_next_task(tasks: list[dict]) -> dict | None:
    """Return the next actionable task in queue order."""
    done_titles = {t["title"] for t in tasks if t["status_kind"] == "done"}
    done_ids = {t["task_id"] for t in tasks if t["status_kind"] == "done" and t.get("task_id")}
    running_ids = {
        t["task_id"]
        for t in tasks
        if t["status_kind"] in RUNNING_KINDS and t.get("task_id")
    }

    for preferred_kind in ACTIONABLE_KINDS:
        for task in tasks:
            if task["status_kind"] != preferred_kind:
                continue
            task_id = task.get("task_id")
            if task_id and task_id in running_ids:
                continue

            dependency = task.get("depends_on", "").strip().strip("`")
            if dependency and dependency not in done_titles and dependency.lower() not in done_ids:
                continue
            return task

    return None


def _replace_task_status(
    text: str,
    task: dict,
    new_status: str,
    completed_date: str | None = None,
) -> str:
    """Replace a task status and optionally insert a completion line."""
    block = text[task["_block_start"] : task["_block_end"]]
    status_start = task["_status_start"] - task["_block_start"]
    status_end = task["_status_end"] - task["_block_start"]

    new_block = block[:status_start] + new_status + block[status_end:]

    if completed_date and "Completed:" not in task.get("body", ""):
        newline = new_block.find("\n")
        completed_line = f"- **Completed:** {completed_date}\n"
        if newline == -1:
            new_block = new_block + "\n" + completed_line
        else:
            new_block = new_block[: newline + 1] + completed_line + new_block[newline + 1 :]

    return text[: task["_block_start"]] + new_block + text[task["_block_end"] :]


def claim_next_task(inbox_path: Path, agent_id: str) -> dict | None:
    """Claim the next actionable task."""
    tasks = parse_tasks(inbox_path)
    task = get_next_task(tasks)
    if task is None:
        return None

    text = inbox_path.read_text()
    updated = _replace_task_status(text, task, f"CLAIMED:{agent_id}")
    inbox_path.write_text(updated)
    return _task_public_view(task)


def mark_claimed_task_done(inbox_path: Path, agent_id: str) -> dict | None:
    """Mark the given agent's claimed task as done."""
    tasks = parse_tasks(inbox_path)
    target = None
    for task in tasks:
        if task["status"] == f"CLAIMED:{agent_id}":
            target = task
            break

    if target is None:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    text = inbox_path.read_text()
    updated = _replace_task_status(text, target, f"DONE {today}", completed_date=today)
    inbox_path.write_text(updated)
    return _task_public_view(target)


def advance_queue(inbox_path: Path) -> dict | None:
    """Mark the next actionable task as done."""
    tasks = parse_tasks(inbox_path)
    task = get_next_task(tasks)
    if task is None:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    text = inbox_path.read_text()
    updated = _replace_task_status(text, task, f"DONE {today}", completed_date=today)
    inbox_path.write_text(updated)
    return _task_public_view(task)


def reclaim_claims(inbox_path: Path, reclaim_agent_ids: set[str]) -> list[str]:
    """Reset claimed tasks owned by missing or stale agents back to READY."""
    if not reclaim_agent_ids:
        return []

    tasks = parse_tasks(inbox_path)
    text = inbox_path.read_text()
    reclaimed = []

    for task in sorted(tasks, key=lambda item: item["_block_start"], reverse=True):
        owner = claimed_agent_id(task["status"])
        if owner not in reclaim_agent_ids:
            continue
        text = _replace_task_status(text, task, "READY")
        reclaimed.append(task["title"])

    if reclaimed:
        inbox_path.write_text(text)

    reclaimed.reverse()
    return reclaimed


def list_claimed_agent_ids(tasks: list[dict]) -> set[str]:
    """Return agent IDs that currently own claimed tasks."""
    owners = set()
    for task in tasks:
        owner = claimed_agent_id(task["status"])
        if owner:
            owners.add(owner)
    return owners
