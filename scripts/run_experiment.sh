#!/bin/bash
# =============================================================================
# Drone RL Lab — One-Command Experiment Runner
# =============================================================================
# Launches Claude Code to autonomously execute the current INBOX queue.
# Just run this and walk away.
#
# Usage:
#   bash scripts/run_experiment.sh             # process queue via Claude Code
#   bash scripts/run_experiment.sh --pipeline  # use pipeline.sh (no Claude Code for training)
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Pull latest (orchestrator may have pushed new INBOX)
echo "[1/3] Pulling latest from GitHub..."
git pull 2>&1 || true

# Show queue status
echo ""
echo "[2/3] Current queue:"
echo "──────────────────────────────────────"
python3 scripts/parse_queue.py 2>/dev/null || head -20 inbox/INBOX.md 2>/dev/null || echo "  (no INBOX.md found)"
echo "──────────────────────────────────────"
echo ""

if [[ "${1:-}" == "--pipeline" ]]; then
    echo "[3/3] Running pipeline..."
    bash scripts/pipeline.sh
else
    echo "[3/3] Launching Claude Code..."
    echo ""
    claude --dangerously-skip-permissions --print \
        "You are the executor in the drone-rl-lab agentic loop. Read CLAUDE.md first, then read memory/HARD_RULES.md, memory/EXPERIMENT_LOG.md, memory/INSIGHTS.md. Process the task queue in inbox/INBOX.md autonomously. For each task: train, benchmark, document per program.md, update memory/NEXT.md and outbox/STATUS.md. When done: commit all results, push to GitHub, and exit."
fi
