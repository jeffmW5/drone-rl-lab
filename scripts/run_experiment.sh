#!/bin/bash
# =============================================================================
# Drone RL Lab — One-Command Experiment Runner
# =============================================================================
# Launches Claude Code to autonomously execute the current INBOX.md task.
# Just run this and walk away.
#
# Usage: bash scripts/run_experiment.sh
# =============================================================================

set -e

REPO_DIR="/media/drone-rl-lab"
cd "$REPO_DIR"

# Pull latest (orchestrator may have pushed new INBOX)
echo "[1/3] Pulling latest from GitHub..."
git pull 2>&1

# Show what's in the INBOX
echo ""
echo "[2/3] Current INBOX task:"
echo "──────────────────────────────────────"
head -5 inbox/INBOX.md 2>/dev/null || echo "  (no INBOX.md found)"
echo "──────────────────────────────────────"
echo ""

# Launch Claude Code with the task
echo "[3/3] Launching Claude Code..."
echo ""
claude --print "You are the executor in the drone-rl-lab agentic loop. Read CLAUDE.md, then execute the task in inbox/INBOX.md autonomously. When completely done: commit all results, push to GitHub, and exit."
