#!/bin/bash
# =============================================================================
# Drone RL Lab — Watch for New Results (Windows-side)
# =============================================================================
# Polls git for new pushes from the executor. When outbox/STATUS.md changes,
# prints a notification. Run this on the orchestrator (Windows) machine.
#
# Usage:
#   bash scripts/watch_results.sh              # poll every 5 min
#   bash scripts/watch_results.sh 120          # poll every 2 min
# =============================================================================

INTERVAL=${1:-300}  # seconds between polls
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LAST_HASH=""

echo "Watching for new results (polling every ${INTERVAL}s)..."
echo "Press Ctrl+C to stop."
echo ""

while true; do
    # Fetch without merging
    git fetch origin 2>/dev/null

    # Check if STATUS.md changed on remote
    CURRENT_HASH=$(git log -1 --format="%H" origin/main -- outbox/STATUS.md 2>/dev/null || echo "")

    if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
        if [ -n "$LAST_HASH" ]; then
            echo ""
            echo "══════════════════════════════════════════"
            echo "  NEW RESULTS DETECTED — $(date '+%Y-%m-%d %H:%M')"
            echo "══════════════════════════════════════════"
            git pull 2>/dev/null
            echo ""
            echo "--- outbox/STATUS.md ---"
            cat outbox/STATUS.md 2>/dev/null
            echo ""
            echo "── Review inbox/INBOX.md for completed tasks ──"
            echo ""
        fi
        LAST_HASH="$CURRENT_HASH"
    fi

    sleep "$INTERVAL"
done
