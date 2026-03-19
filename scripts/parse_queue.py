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
import re
import sys
from datetime import datetime
from pathlib import Path

INBOX_PATH = Path(__file__).resolve().parent.parent / "inbox" / "INBOX.md"


def parse_tasks(inbox_path: Path = INBOX_PATH) -> list[dict]:
    """Parse [NEXT]/[QUEUED]/[DONE] tasks from INBOX.md."""
    if not inbox_path.is_file():
        return []

    text = inbox_path.read_text()
    tasks = []

    # Match task blocks: ### [STATUS] Title
    # Followed by optional metadata lines and body until next ### or end
    pattern = re.compile(
        r"^###\s+\[(\w+)\]\s+(.+?)$"
        r"(.*?)(?=^###|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    for m in pattern.finditer(text):
        status = m.group(1).strip()
        title = m.group(2).strip()
        body = m.group(3).strip()

        task = {"status": status, "title": title, "body": body}

        # Extract metadata from body
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("- **Experiment:**"):
                task["experiment"] = line.split(":**", 1)[1].strip()
            elif line.startswith("- **Config:**"):
                task["config"] = line.split(":**", 1)[1].strip()
            elif line.startswith("- **Depends on:**"):
                task["depends_on"] = line.split(":**", 1)[1].strip()
            elif line.startswith("- **Completed:**"):
                task["completed"] = line.split(":**", 1)[1].strip()

        tasks.append(task)

    return tasks


def get_next_task(tasks: list[dict]) -> dict | None:
    """Get the first actionable task ([NEXT], or first [QUEUED] with no unmet deps)."""
    for t in tasks:
        if t["status"] == "NEXT":
            return t

    done_titles = {t["title"] for t in tasks if t["status"] == "DONE"}
    for t in tasks:
        if t["status"] == "QUEUED":
            dep = t.get("depends_on", "")
            if not dep or dep in done_titles:
                return t

    return None


def advance_queue(inbox_path: Path = INBOX_PATH) -> str:
    """Mark [NEXT] as [DONE] with today's date, promote next [QUEUED] to [NEXT]."""
    if not inbox_path.is_file():
        return "No INBOX.md found"

    text = inbox_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    changed = False

    # Mark [NEXT] → [DONE]
    new_text, n = re.subn(
        r"^(###\s+)\[NEXT\]",
        rf"\1[DONE]",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n > 0:
        # Add completion date if not present
        new_text = re.sub(
            r"(\[DONE\]\s+.+?\n)((?:- \*\*.*\n)*)",
            lambda m: m.group(0) if "Completed:" in m.group(0)
            else m.group(1) + m.group(2) + f"- **Completed:** {today}\n",
            new_text,
            count=1,
        )
        changed = True

    # Promote first [QUEUED] → [NEXT]
    if changed:
        new_text, n2 = re.subn(
            r"^(###\s+)\[QUEUED\]",
            r"\1[NEXT]",
            new_text,
            count=1,
            flags=re.MULTILINE,
        )
        if n2 > 0:
            changed = True

    if changed:
        inbox_path.write_text(new_text)
        return f"Advanced queue (marked DONE, promoted next QUEUED)"
    else:
        return "No [NEXT] task found to advance"


def main():
    parser = argparse.ArgumentParser(description="Parse INBOX task queue")
    parser.add_argument("--next", action="store_true", help="Print first actionable task")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--advance", action="store_true", help="Mark NEXT as DONE, promote next QUEUED")
    parser.add_argument("--inbox", default=str(INBOX_PATH), help="Path to INBOX.md")
    args = parser.parse_args()

    inbox_path = Path(args.inbox)

    if args.advance:
        result = advance_queue(inbox_path)
        print(result)
        return

    tasks = parse_tasks(inbox_path)

    if args.next:
        task = get_next_task(tasks)
        if task:
            if args.json:
                print(json.dumps(task, indent=2))
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
        print(json.dumps(tasks, indent=2))
    else:
        if not tasks:
            print("Queue is empty.")
            return
        for t in tasks:
            marker = {"DONE": "x", "NEXT": ">", "QUEUED": " "}.get(t["status"], "?")
            print(f"  [{marker}] [{t['status']}] {t['title']}")


if __name__ == "__main__":
    main()
