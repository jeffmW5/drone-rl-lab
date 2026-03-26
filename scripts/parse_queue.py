#!/usr/bin/env python3
"""
Parse the INBOX queue and print task status.

Usage:
    python scripts/parse_queue.py              # print all tasks
    python scripts/parse_queue.py --next       # print first actionable task
    python scripts/parse_queue.py --json       # JSON output
    python scripts/parse_queue.py --advance    # mark [NEXT] as [DONE], promote next [QUEUED]
"""

import argparse
import json
import sys
from pathlib import Path

from task_queue import advance_queue, get_next_task, parse_tasks

INBOX_PATH = Path(__file__).resolve().parent.parent / "inbox" / "INBOX.md"


def main():
    parser = argparse.ArgumentParser(description="Parse INBOX task queue")
    parser.add_argument("--next", action="store_true", help="Print first actionable task")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--advance", action="store_true", help="Mark NEXT as DONE, promote next QUEUED")
    parser.add_argument("--inbox", default=str(INBOX_PATH), help="Path to INBOX.md")
    args = parser.parse_args()

    inbox_path = Path(args.inbox)

    if args.advance:
        task = advance_queue(inbox_path)
        if task:
            print(f"Advanced queue (marked DONE): {task['title']}")
        else:
            print("No actionable task found to advance")
        return

    tasks = parse_tasks(inbox_path)

    if args.next:
        task = get_next_task(tasks)
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
            }.get(kind, "?")
            print(f"  [{marker}] [{t['status']}] {t['title']}")


if __name__ == "__main__":
    main()
