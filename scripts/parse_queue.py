#!/usr/bin/env python3
"""
Parse the INBOX queue and print task status.

Reads from both the Markdown INBOX and the JSON task store, merging results
so that JSON-backed tasks appear alongside legacy Markdown entries.

Usage:
    python scripts/parse_queue.py              # print all tasks
    python scripts/parse_queue.py --next       # print first actionable task
    python scripts/parse_queue.py --json       # JSON output
    python scripts/parse_queue.py --advance    # mark [NEXT] as [DONE], promote next [QUEUED]
    python scripts/parse_queue.py --source md  # only show Markdown tasks
    python scripts/parse_queue.py --source json # only show JSON tasks
"""

import argparse
import json
import sys
from pathlib import Path

from task_queue import advance_queue, get_next_task, parse_tasks
from task_store import TaskStore

INBOX_PATH = Path(__file__).resolve().parent.parent / "inbox" / "INBOX.md"

# Map JSON task status to the status_kind format used by task_queue
_JSON_KIND_MAP = {
    "ready": "ready",
    "claimed": "claimed",
    "in_progress": "in_progress",
    "done": "done",
    "blocked": "blocked",
    "deferred": "deferred",
    "failed": "done",
}


def _json_tasks_as_queue_format(store: TaskStore) -> list[dict]:
    """Convert JSON tasks to the dict format parse_tasks() returns."""
    result = []
    for task in store.list_all():
        status_display = task["status"].upper()
        if task["status"] == "claimed" and task.get("claimed_by"):
            status_display = f"CLAIMED:{task['claimed_by']}"
        elif task["status"] == "done" and task.get("completed_at"):
            status_display = f"DONE {task['completed_at'][:10]}"

        entry = {
            "status": status_display,
            "status_kind": _JSON_KIND_MAP.get(task["status"], task["status"]),
            "title": task["title"],
            "body": "",
            "task_id": task["task_id"],
            "config": task.get("config"),
            "experiment": task.get("task_id"),
            "_source": "json",
        }
        result.append(entry)
    return result


def _merged_tasks(inbox_path: Path, store: TaskStore) -> list[dict]:
    """Merge Markdown and JSON tasks, deduplicating by task_id."""
    md_tasks = parse_tasks(inbox_path) if inbox_path.is_file() else []
    json_tasks = _json_tasks_as_queue_format(store)
    json_ids = {t["task_id"] for t in json_tasks if t.get("task_id")}

    # Keep MD tasks that don't have a JSON counterpart
    merged = []
    for t in md_tasks:
        if t.get("task_id") and t["task_id"] in json_ids:
            continue  # JSON version takes precedence
        t["_source"] = "md"
        merged.append(t)

    # Add all JSON tasks
    merged.extend(json_tasks)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Parse INBOX task queue")
    parser.add_argument("--next", action="store_true", help="Print first actionable task")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--advance", action="store_true", help="Mark NEXT as DONE, promote next QUEUED")
    parser.add_argument("--inbox", default=str(INBOX_PATH), help="Path to INBOX.md")
    parser.add_argument(
        "--source",
        choices=["all", "md", "json"],
        default="all",
        help="Filter by task source (default: all)",
    )
    args = parser.parse_args()

    inbox_path = Path(args.inbox)
    store = TaskStore()

    if args.advance:
        task = advance_queue(inbox_path)
        if task:
            print(f"Advanced queue (marked DONE): {task['title']}")
        else:
            print("No actionable task found to advance")
        return

    # Choose task source
    if args.source == "md":
        tasks = parse_tasks(inbox_path)
    elif args.source == "json":
        tasks = _json_tasks_as_queue_format(store)
    else:
        tasks = _merged_tasks(inbox_path, store)

    if args.next:
        # For --next, prefer JSON store's get_next over Markdown
        json_next = store.get_next()
        md_next = get_next_task(parse_tasks(inbox_path)) if inbox_path.is_file() else None

        # Use JSON next if available, fall back to Markdown
        task = None
        if json_next:
            task = {
                "status": json_next["status"].upper(),
                "status_kind": json_next["status"],
                "title": json_next["title"],
                "task_id": json_next["task_id"],
                "config": json_next.get("config"),
                "experiment": json_next.get("task_id"),
            }
        elif md_next:
            task = md_next

        if task:
            if args.json:
                print(json.dumps({k: v for k, v in task.items() if not k.startswith("_")}, indent=2))
            else:
                print(f"[{task['status']}] {task['title']}")
                if task.get("experiment"):
                    print(f"  Experiment: {task['experiment']}")
                if task.get("config"):
                    print(f"  Config: {task['config']}")
        else:
            print("No actionable tasks in queue.")
            sys.exit(1)
        return

    if args.json:
        public_tasks = [{k: v for k, v in task.items() if not k.startswith("_")} for task in tasks]
        print(json.dumps(public_tasks, indent=2))
    else:
        if not tasks:
            print("Queue is empty.")
            return
        for t in tasks:
            kind = t["status_kind"]
            marker = {
                "done": "x",
                "next": ">",
                "ready": ">",
                "implemented": "+",
                "queued": " ",
                "claimed": "~",
                "in_progress": "~",
                "note": "i",
                "blocked": "!",
                "deferred": "-",
                "failed": "X",
            }.get(kind, "?")
            source = t.get("_source", "")
            source_tag = f" ({source})" if source else ""
            print(f"  [{marker}] [{t['status']}] {t['title']}{source_tag}")


if __name__ == "__main__":
    main()
