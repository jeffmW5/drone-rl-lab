#!/usr/bin/env python3
"""
Render inbox/INBOX.md from JSON task artifacts + legacy Markdown entries.

This is the compatibility bridge between the typed task store and the
existing Markdown-first workflow. It:

1. Reads all JSON tasks from inbox/tasks/.
2. Reads legacy (non-JSON-backed) entries from inbox/INBOX.md.
3. Writes a merged INBOX.md that preserves both views.

Usage:
    python scripts/render_inbox.py              # preview merged output
    python scripts/render_inbox.py --write      # overwrite inbox/INBOX.md
    python scripts/render_inbox.py --check      # exit 1 if INBOX.md is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from task_store import TaskStore
from task_queue import parse_tasks

REPO_DIR = Path(__file__).resolve().parent.parent
INBOX_PATH = REPO_DIR / "inbox" / "INBOX.md"

# Tasks already in JSON that should not be duplicated from Markdown
def _json_task_ids(store: TaskStore) -> set[str]:
    return {t["task_id"] for t in store.list_all() if t.get("task_id")}


STATUS_DISPLAY = {
    "ready": "READY",
    "claimed": "CLAIMED",
    "in_progress": "IN PROGRESS",
    "done": "DONE",
    "blocked": "BLOCKED",
    "deferred": "DEFERRED",
    "failed": "DONE",
}


def _render_json_task(task: dict) -> str:
    """Render a single JSON task as a Markdown queue block."""
    status_raw = STATUS_DISPLAY.get(task["status"], task["status"].upper())

    if task["status"] == "claimed" and task.get("claimed_by"):
        status_raw = f"CLAIMED:{task['claimed_by']}"
    elif task["status"] == "done" and task.get("completed_at"):
        date = task["completed_at"][:10]
        status_raw = f"DONE {date}"
        if task.get("result_summary"):
            status_raw = f"DONE {date}"

    lines = [f"### [{status_raw}] {task['title']}"]

    if task.get("result_summary"):
        lines.append(f"- **Result:** {task['result_summary']}")
    if task.get("result_diagnosis"):
        lines.append(f"- **Diagnosis:** {task['result_diagnosis']}")
    if task.get("hypothesis"):
        lines.append(f"- **Hypothesis:** {task['hypothesis']}")
    if task.get("what_to_change"):
        lines.append(f"- **What to change:** {task['what_to_change']}")
    if task.get("expected_outcome"):
        lines.append(f"- **Expected outcome:** {task['expected_outcome']}")
    if task.get("scope_note"):
        lines.append(f"- **Scope note:** {task['scope_note']}")
    if task.get("config"):
        lines.append(f"- **Config:** `{task['config']}`")
    if task.get("depends_on"):
        lines.append(f"- **Depends on:** `{task['depends_on']}`")
    if task.get("result_path"):
        lines.append(f"- See `{task['result_path']}`")

    return "\n".join(lines)


def _render_section(title: str, blocks: list[str]) -> str:
    """Wrap blocks in a section."""
    if not blocks:
        return ""
    return f"## {title}\n\n" + "\n\n".join(blocks) + "\n"


def render_merged(store: TaskStore) -> str:
    """Generate the full INBOX.md content."""
    json_tasks = store.list_all()
    json_ids = {t["task_id"] for t in json_tasks if t.get("task_id")}

    # Parse legacy tasks that are NOT backed by JSON
    legacy_tasks = []
    if INBOX_PATH.is_file():
        for task in parse_tasks(INBOX_PATH):
            task_id = task.get("task_id")
            if task_id and task_id in json_ids:
                continue  # skip -- JSON version is source of truth
            legacy_tasks.append(task)

    # Group JSON tasks by status
    active_blocks = []
    done_blocks = []
    deferred_blocks = []

    for task in json_tasks:
        block = _render_json_task(task)
        if task["status"] in ("done", "failed"):
            done_blocks.append(block)
        elif task["status"] == "deferred":
            deferred_blocks.append(block)
        else:
            active_blocks.append(block)

    # Group legacy tasks by status
    legacy_done_blocks = []
    legacy_active_blocks = []
    for task in legacy_tasks:
        # Reconstruct the markdown block
        block = f"### [{task['status']}] {task['title']}"
        if task.get("body"):
            block += "\n" + task["body"]
        if task["status_kind"] == "done":
            legacy_done_blocks.append(block)
        else:
            legacy_active_blocks.append(block)

    output = "# INBOX -- Experiment Queue\n\n"
    output += "> Mark each task [DONE] when complete.\n"
    output += "> Tasks backed by JSON artifacts in `inbox/tasks/` are the source of truth.\n"
    output += "> Legacy Markdown-only entries appear below and will be migrated over time.\n\n"
    output += "---\n\n"

    output += _render_section("Queue", active_blocks + legacy_active_blocks)
    if deferred_blocks:
        output += "\n" + _render_section("Deferred", deferred_blocks)
    output += "\n---\n\n"
    output += _render_section("Completed", done_blocks + legacy_done_blocks)

    return output


def main():
    parser = argparse.ArgumentParser(description="Render INBOX.md from JSON tasks")
    parser.add_argument("--write", action="store_true", help="Overwrite inbox/INBOX.md")
    parser.add_argument("--check", action="store_true", help="Exit 1 if INBOX.md is stale")
    args = parser.parse_args()

    store = TaskStore()
    rendered = render_merged(store)

    if args.check:
        if INBOX_PATH.is_file():
            current = INBOX_PATH.read_text()
            if current.strip() == rendered.strip():
                print("INBOX.md is up to date.")
                sys.exit(0)
            else:
                print("INBOX.md is stale. Run: python scripts/render_inbox.py --write")
                sys.exit(1)
        else:
            print("INBOX.md does not exist.")
            sys.exit(1)

    if args.write:
        INBOX_PATH.write_text(rendered)
        print(f"Wrote {INBOX_PATH}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
