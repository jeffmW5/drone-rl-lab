#!/usr/bin/env python3
"""
Agent coordination for parallel drone-rl-lab instances.

Each agent registers with a unique ID, claims tasks atomically via git,
and maintains a heartbeat so stale claims can be detected and reclaimed.

Usage:
    python scripts/agent_lock.py register              # register, print agent ID
    python scripts/agent_lock.py heartbeat <ID>        # update heartbeat timestamp
    python scripts/agent_lock.py heartbeat <ID> --task <task> --status <status>
    python scripts/agent_lock.py status                # list all active agents
    python scripts/agent_lock.py stale                 # list stale agents (30+ min)
    python scripts/agent_lock.py deregister <ID>       # remove agent file
    python scripts/agent_lock.py claim <ID>            # claim next available task
    python scripts/agent_lock.py release <ID>          # mark claimed task [DONE]
    python scripts/agent_lock.py reclaim-stale <ID>    # reclaim tasks from dead agents
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_DIR / "agents"
INBOX_PATH = REPO_DIR / "inbox" / "INBOX.md"
STALE_MINUTES = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _agent_path(agent_id: str) -> Path:
    return AGENTS_DIR / f"{agent_id}.json"


def _read_agent(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_agent(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2) + "\n")


def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
        check=check,
    )


def _git_sync_and_push(message: str, max_retries: int = 3) -> bool:
    """Pull, add, commit, push. Returns True if push succeeds."""
    for attempt in range(max_retries):
        _git("pull", "--rebase", check=False)
        _git("add", "-A")
        result = _git("commit", "-m", message, check=False)
        if result.returncode != 0 and "nothing to commit" in result.stdout:
            return True  # nothing changed, that's fine
        result = _git("push", check=False)
        if result.returncode == 0:
            return True
        # Push failed (likely conflict), pull and retry
        time.sleep(1)
        _git("pull", "--rebase", check=False)
    return False


def _all_agents() -> list[dict]:
    """Read all agent status files."""
    agents = []
    for f in AGENTS_DIR.glob("*.json"):
        data = _read_agent(f)
        if data:
            agents.append(data)
    return agents


def _is_stale(agent: dict) -> bool:
    """Check if agent heartbeat is older than STALE_MINUTES."""
    try:
        hb = datetime.fromisoformat(agent["heartbeat"])
        now = datetime.now(timezone.utc)
        return (now - hb).total_seconds() > STALE_MINUTES * 60
    except (KeyError, ValueError):
        return True  # no heartbeat = stale


# ─── Commands ────────────────────────────────────────────────────────────────


def cmd_register(args):
    """Register a new agent, print its ID."""
    hostname = socket.gethostname()
    pid = os.getpid()
    ts = int(time.time())
    agent_id = f"{hostname}-{pid}-{ts}"

    data = {
        "id": agent_id,
        "hostname": hostname,
        "pid": pid,
        "started": _now_iso(),
        "heartbeat": _now_iso(),
        "task": None,
        "status": "idle",
    }

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_agent(_agent_path(agent_id), data)
    _git_sync_and_push(f"agent {agent_id}: registered")
    print(agent_id)


def cmd_heartbeat(args):
    """Update heartbeat timestamp and optionally task/status."""
    path = _agent_path(args.agent_id)
    data = _read_agent(path)
    if not data:
        print(f"Agent {args.agent_id} not found", file=sys.stderr)
        sys.exit(1)

    data["heartbeat"] = _now_iso()
    if args.task is not None:
        data["task"] = args.task
    if args.status is not None:
        data["status"] = args.status

    _write_agent(path, data)
    # Heartbeat commits are lightweight — push but don't fail hard
    _git_sync_and_push(f"agent {args.agent_id}: heartbeat", max_retries=1)
    print(f"Heartbeat updated: {data['heartbeat']}")


def cmd_status(args):
    """List all active agents."""
    _git("pull", check=False)
    agents = _all_agents()
    if not agents:
        print("No active agents.")
        return

    for a in agents:
        stale = " [STALE]" if _is_stale(a) else ""
        task = a.get("task") or "idle"
        status = a.get("status", "unknown")
        print(f"  {a['id']}  |  {status}: {task}{stale}  |  heartbeat: {a.get('heartbeat', '?')}")


def cmd_stale(args):
    """List stale agents (no heartbeat for 30+ min)."""
    _git("pull", check=False)
    agents = _all_agents()
    stale = [a for a in agents if _is_stale(a)]
    if not stale:
        print("No stale agents.")
        return
    for a in stale:
        task = a.get("task") or "idle"
        print(f"  STALE: {a['id']}  |  task: {task}  |  last heartbeat: {a.get('heartbeat', '?')}")


def cmd_deregister(args):
    """Remove agent status file."""
    path = _agent_path(args.agent_id)
    if path.exists():
        path.unlink()
        _git_sync_and_push(f"agent {args.agent_id}: deregistered")
        print(f"Deregistered {args.agent_id}")
    else:
        print(f"Agent {args.agent_id} not found (already deregistered?)")


def cmd_claim(args):
    """Claim the next available task in INBOX. Uses git push as atomic lock."""
    agent_id = args.agent_id

    for attempt in range(3):
        _git("pull", "--rebase", check=False)

        if not INBOX_PATH.is_file():
            print("No INBOX.md found", file=sys.stderr)
            sys.exit(1)

        text = INBOX_PATH.read_text()

        # Find all claimed tasks (by any agent)
        claimed_pattern = re.compile(r"\[CLAIMED:([^\]]+)\]")
        claimed_tasks = set(claimed_pattern.findall(text))

        # Find first [NEXT] or [QUEUED] that isn't claimed
        task_pattern = re.compile(
            r"^(###\s+)\[(NEXT|QUEUED)\]\s+(.+?)$",
            re.MULTILINE,
        )

        match = None
        for m in task_pattern.finditer(text):
            match = m
            break

        if not match:
            print("No available tasks to claim.")
            sys.exit(1)

        task_title = match.group(3).strip()
        old_status = match.group(2)

        # Replace [NEXT] or [QUEUED] with [CLAIMED:agent-id]
        new_text = (
            text[: match.start()]
            + f"{match.group(1)}[CLAIMED:{agent_id}] {task_title}"
            + text[match.end() :]
        )
        INBOX_PATH.write_text(new_text)

        # Update agent status
        agent_path = _agent_path(agent_id)
        agent_data = _read_agent(agent_path)
        if agent_data:
            agent_data["task"] = task_title
            agent_data["status"] = "claimed"
            agent_data["heartbeat"] = _now_iso()
            _write_agent(agent_path, agent_data)

        # Try to push — if fails, another agent beat us
        if _git_sync_and_push(f"agent {agent_id}: claimed '{task_title}'", max_retries=1):
            print(f"Claimed: {task_title}")
            return
        else:
            print(f"Claim attempt {attempt + 1} failed (conflict), retrying...", file=sys.stderr)
            # Re-pull will get the other agent's claim, loop will skip it
            time.sleep(1)

    print("Failed to claim a task after 3 attempts.", file=sys.stderr)
    sys.exit(1)


def cmd_release(args):
    """Mark the agent's claimed task as [DONE]."""
    agent_id = args.agent_id
    today = datetime.now().strftime("%Y-%m-%d")

    _git("pull", "--rebase", check=False)

    if not INBOX_PATH.is_file():
        print("No INBOX.md found", file=sys.stderr)
        sys.exit(1)

    text = INBOX_PATH.read_text()

    # Find [CLAIMED:agent-id] and replace with [DONE]
    pattern = re.compile(
        rf"^(###\s+)\[CLAIMED:{re.escape(agent_id)}\]\s+(.+?)$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        print(f"No claimed task found for agent {agent_id}", file=sys.stderr)
        sys.exit(1)

    task_title = match.group(2).strip()
    new_text = pattern.sub(rf"\1[DONE] \2\n- **Completed:** {today}", text, count=1)

    # Promote next [QUEUED] → [NEXT] if no other [NEXT] exists
    if "[NEXT]" not in new_text:
        new_text = re.sub(
            r"^(###\s+)\[QUEUED\]",
            r"\1[NEXT]",
            new_text,
            count=1,
            flags=re.MULTILINE,
        )

    INBOX_PATH.write_text(new_text)

    # Update agent status
    agent_path = _agent_path(agent_id)
    agent_data = _read_agent(agent_path)
    if agent_data:
        agent_data["task"] = None
        agent_data["status"] = "idle"
        agent_data["heartbeat"] = _now_iso()
        _write_agent(agent_path, agent_data)

    _git_sync_and_push(f"agent {agent_id}: released '{task_title}' [DONE]")
    print(f"Released: {task_title} → [DONE]")


def cmd_reclaim_stale(args):
    """Find tasks claimed by stale agents and reset them to [NEXT]."""
    agent_id = args.agent_id  # the agent doing the reclaiming

    _git("pull", "--rebase", check=False)

    stale_agents = [a for a in _all_agents() if _is_stale(a) and a["id"] != agent_id]
    if not stale_agents:
        print("No stale agents to reclaim from.")
        return

    text = INBOX_PATH.read_text()
    reclaimed = []

    for sa in stale_agents:
        pattern = re.compile(
            rf"^(###\s+)\[CLAIMED:{re.escape(sa['id'])}\]\s+(.+?)$",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if match:
            task_title = match.group(2).strip()
            text = pattern.sub(rf"\1[NEXT] \2", text, count=1)
            reclaimed.append(f"{task_title} (from {sa['id']})")

        # Remove stale agent file
        stale_path = _agent_path(sa["id"])
        if stale_path.exists():
            stale_path.unlink()

    if reclaimed:
        INBOX_PATH.write_text(text)
        _git_sync_and_push(f"agent {agent_id}: reclaimed {len(reclaimed)} stale task(s)")
        for r in reclaimed:
            print(f"Reclaimed: {r}")
    else:
        print("No stale claims found in INBOX.")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Agent coordination for drone-rl-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("register", help="Register a new agent")

    hb = sub.add_parser("heartbeat", help="Update heartbeat")
    hb.add_argument("agent_id")
    hb.add_argument("--task", default=None)
    hb.add_argument("--status", default=None)

    sub.add_parser("status", help="List all active agents")
    sub.add_parser("stale", help="List stale agents")

    dereg = sub.add_parser("deregister", help="Remove agent")
    dereg.add_argument("agent_id")

    cl = sub.add_parser("claim", help="Claim next available task")
    cl.add_argument("agent_id")

    rel = sub.add_parser("release", help="Mark claimed task [DONE]")
    rel.add_argument("agent_id")

    recl = sub.add_parser("reclaim-stale", help="Reclaim tasks from stale agents")
    recl.add_argument("agent_id")

    args = parser.parse_args()

    cmds = {
        "register": cmd_register,
        "heartbeat": cmd_heartbeat,
        "status": cmd_status,
        "stale": cmd_stale,
        "deregister": cmd_deregister,
        "claim": cmd_claim,
        "release": cmd_release,
        "reclaim-stale": cmd_reclaim_stale,
    }

    cmds[args.command](args)


if __name__ == "__main__":
    main()
