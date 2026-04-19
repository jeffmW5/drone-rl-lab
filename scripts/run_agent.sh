#!/bin/bash
# =============================================================================
# Drone RL Lab - Autonomous Agent Launcher (parallel-safe)
# =============================================================================
# Registers an agent, starts a background heartbeat, launches Claude Code in
# interactive mode, and cleans up on exit. Multiple instances can run safely.
#
# Usage:
#   bash scripts/run_agent.sh                # launch agent
#   bash scripts/run_agent.sh --dry-run      # show queue + active agents
#   bash scripts/run_agent.sh --status       # show active agents only
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== Dry Run ==="
    echo "Repo: $REPO_DIR"
    echo ""
    echo "Active agents:"
    python3 scripts/agent_lock.py status 2>/dev/null || echo "  (none)"
    echo ""
    echo "Queue:"
    python3 scripts/parse_queue.py 2>/dev/null || echo "  (empty)"
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    python3 scripts/agent_lock.py status
    exit 0
fi

git pull 2>/dev/null || true
source /home/jeff/drones-venv/bin/activate 2>/dev/null || source /media/drones-venv/bin/activate 2>/dev/null || true

AGENT_ID=$(python3 scripts/agent_lock.py register)
echo "Registered agent: $AGENT_ID"

python3 scripts/agent_lock.py reclaim-stale "$AGENT_ID" 2>/dev/null || true

(
    while true; do
        sleep 300
        python3 "$REPO_DIR/scripts/agent_lock.py" heartbeat "$AGENT_ID" 2>/dev/null || true
    done
) &
HEARTBEAT_PID=$!

cleanup() {
    echo ""
    echo "Cleaning up agent $AGENT_ID..."
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    python3 "$REPO_DIR/scripts/agent_lock.py" deregister "$AGENT_ID" 2>/dev/null || true
    echo "Agent deregistered."
}
trap cleanup EXIT INT TERM

echo "Launching Claude Code (interactive, autonomous)..."
echo "Agent ID: $AGENT_ID"
echo ""

claude --dangerously-skip-permissions --initial-prompt "$(cat <<PROMPT
You are an autonomous drone-rl-lab agent.
Your assigned agent ID is: $AGENT_ID

Read and follow CLAUDE.md exactly before doing anything else.

Use the assigned agent ID above for all \`scripts/agent_lock.py\` coordination
commands. Do not register a second agent. A background heartbeat is already
running, but you should still push explicit heartbeat updates when your task or
status changes.

Then execute the repo workflow from CLAUDE.md end-to-end.
PROMPT
)"
