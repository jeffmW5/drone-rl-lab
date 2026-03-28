#!/bin/bash
# =============================================================================
# Drone RL Lab — Full Pipeline Runner
# =============================================================================
# Processes the INBOX queue: for each task, runs train → benchmark → document.
# Commits and pushes after each completed experiment.
#
# Usage:
#   bash scripts/pipeline.sh              # process entire queue
#   bash scripts/pipeline.sh --dry-run    # show queue without executing
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "╔══════════════════════════════════════╗"
echo "║  Drone RL Lab — Pipeline Runner      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Step 1: Pull latest
echo "[1] Pulling latest from GitHub..."
git pull 2>&1 || true
echo ""

# Step 2: Show queue
echo "[2] Current queue:"
python3 scripts/parse_queue.py || true
echo ""

if $DRY_RUN; then
    echo "[dry-run] Would process tasks above. Exiting."
    exit 0
fi

# Step 3: Process tasks
TASKS_DONE=0
MAX_TASKS=10  # safety limit

while [ $TASKS_DONE -lt $MAX_TASKS ]; do
    # Get next task
    NEXT=$(python3 scripts/parse_queue.py --next 2>/dev/null) || break
    echo ""
    echo "════════════════════════════════════════"
    echo "[3] Processing: $NEXT"
    echo "════════════════════════════════════════"

    # Get task details as JSON
    TASK_JSON=$(python3 scripts/parse_queue.py --next --json 2>/dev/null) || break
    CONFIG=$(echo "$TASK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('config',''))" 2>/dev/null)
    EXPERIMENT=$(echo "$TASK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('experiment',''))" 2>/dev/null)

    if [ -z "$CONFIG" ]; then
        echo "[WARN] No config found for task. Skipping to Claude Code."
        # Fall back to Claude Code for non-standard tasks
        claude --dangerously-skip-permissions --print \
            "You are the executor in the drone-rl-lab loop. Read and follow CLAUDE.md exactly, then execute the next actionable task from inbox/INBOX.md. Commit and push before exiting."
    else
        echo "[3a] Training: python3 train.py configs/$CONFIG"
        source /media/drones-venv/bin/activate 2>/dev/null || true
        python3 train.py "configs/$CONFIG"

        echo ""
        echo "[3a.1] Capturing provenance..."
        python3 scripts/capture_provenance.py --experiment "$EXPERIMENT" || echo "[WARN] Provenance capture failed (non-fatal)"

        echo ""
        echo "[3b] Post-training: generating log..."
        python3 compare.py --generate-log

        # Auto-benchmark if benchmark.py exists and experiment name is known
        if [ -n "$EXPERIMENT" ] && [ -f scripts/benchmark.py ]; then
            echo ""
            echo "[3c] Benchmarking: $EXPERIMENT"
            python3 scripts/benchmark.py -e "$EXPERIMENT" -n 5 || echo "[WARN] Benchmark failed (non-fatal)"
        fi

        echo ""
        echo "[3d] Documenting results..."
        # Use Claude Code for documentation (EXPERIMENT.md, outbox)
        claude --dangerously-skip-permissions --print \
            "You are the executor. Read and follow CLAUDE.md exactly. The experiment $EXPERIMENT just finished training. Write results/$EXPERIMENT/EXPERIMENT.md and outbox/$EXPERIMENT.md per program.md, update any required memory/outbox/state files, then commit and push."
    fi

    # Advance queue
    echo ""
    echo "[3e] Advancing queue..."
    python3 scripts/parse_queue.py --advance

    echo "[3e.1] Refreshing lab state..."
    python3 scripts/lab_state.py || true

    # Commit and push
    echo "[3f] Committing and pushing..."
    git add -A
    git commit -m "$EXPERIMENT: pipeline run complete" 2>/dev/null || true
    git push 2>/dev/null || true

    TASKS_DONE=$((TASKS_DONE + 1))
    echo "[✓] Task $TASKS_DONE complete: $EXPERIMENT"
done

echo ""
echo "════════════════════════════════════════"
echo "Pipeline finished. Tasks completed: $TASKS_DONE"
python3 scripts/parse_queue.py || true
