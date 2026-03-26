#!/bin/bash
# =============================================================================
# Drone RL Lab — Autonomous Agent Launcher (parallel-safe)
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

# ─── Dry-run / status ───────────────────────────────────────────────────────

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

# ─── Pull latest ─────────────────────────────────────────────────────────────

git pull 2>/dev/null || true

# ─── Activate venv ───────────────────────────────────────────────────────────

source /media/drones-venv/bin/activate 2>/dev/null || true

# ─── Register agent ─────────────────────────────────────────────────────────

AGENT_ID=$(python3 scripts/agent_lock.py register)
echo "Registered agent: $AGENT_ID"

# ─── Reclaim any stale tasks from dead agents ───────────────────────────────

python3 scripts/agent_lock.py reclaim-stale "$AGENT_ID" 2>/dev/null || true

# ─── Background heartbeat (every 5 min) ─────────────────────────────────────

(
    while true; do
        sleep 300
        python3 "$REPO_DIR/scripts/agent_lock.py" heartbeat "$AGENT_ID" 2>/dev/null || true
    done
) &
HEARTBEAT_PID=$!

# ─── Cleanup on exit ────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "Cleaning up agent $AGENT_ID..."
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    python3 "$REPO_DIR/scripts/agent_lock.py" deregister "$AGENT_ID" 2>/dev/null || true
    echo "Agent deregistered."
}
trap cleanup EXIT INT TERM

# ─── Launch Claude Code ─────────────────────────────────────────────────────

echo "Launching Claude Code (interactive, autonomous)..."
echo "Agent ID: $AGENT_ID"
echo ""

claude --dangerously-skip-permissions --initial-prompt "$(cat <<PROMPT
You are an autonomous drone-rl-lab agent. Your agent ID is: $AGENT_ID

You handle the ENTIRE research loop alone. Multiple agents may be running in
parallel — use the coordination tools below to avoid conflicts.

## CRITICAL: Parallel Agent Coordination

Before doing ANY work, you MUST coordinate with other agents:

1. **Check active agents:**
   \`\`\`bash
   python3 scripts/agent_lock.py status
   \`\`\`

2. **Claim a task before working on it:**
   \`\`\`bash
   python3 scripts/agent_lock.py claim $AGENT_ID
   \`\`\`
   This atomically claims the next available task via git push. If another
   agent already claimed it, this will retry automatically.

3. **Update your heartbeat periodically** (auto-runs in background, but also
   update when changing status):
   \`\`\`bash
   python3 scripts/agent_lock.py heartbeat $AGENT_ID --task "exp_NNN" --status "training"
   \`\`\`

4. **Release task when done:**
   \`\`\`bash
   python3 scripts/agent_lock.py release $AGENT_ID
   \`\`\`

5. **Reclaim stale tasks** from dead agents (30+ min no heartbeat):
   \`\`\`bash
   python3 scripts/agent_lock.py reclaim-stale $AGENT_ID
   \`\`\`

## On start (do this FIRST)

\`\`\`bash
cd /media/drone-rl-lab
git pull 2>/dev/null || true
\`\`\`

Read these files in order:
1. \`memory/HARD_RULES.md\` — absolute constraints, NEVER violate
2. \`memory/EXPERIMENT_LOG.md\` — full history
3. \`memory/INSIGHTS.md\` — targets, benchmarks, paper references
4. \`memory/NEXT.md\` — priorities and open questions
5. \`inbox/INBOX.md\` — task queue (check for [CLAIMED:*] tasks = other agents)
6. \`outbox/STATUS.md\` — latest results
7. \`state/current.json\` if present — canonical machine-readable lab state
8. Check other active agents: \`python3 scripts/agent_lock.py status\`

## Decision loop

### If INBOX has unclaimed actionable tasks → claim and execute
1. \`python3 scripts/agent_lock.py claim $AGENT_ID\`
2. If task type is \`research\`: run /research command
3. If task type is training: run the experiment, benchmark, document
4. \`python3 scripts/agent_lock.py release $AGENT_ID\`
5. Continue to next task

### If INBOX is empty or all tasks claimed → analyze and self-direct
1. Check for plateau (3+ consecutive experiments, no improvement)
2. If plateau: trigger paper research via /research, then design experiments
3. Add new tasks to INBOX, claim one, execute it
4. Always git pull before writing to shared files

## After EVERY experiment

1. Write \`results/exp_NNN/EXPERIMENT.md\` per program.md standard
2. Write \`outbox/exp_NNN.md\` summary
3. Run \`python3 scripts/capture_provenance.py --experiment exp_NNN\`
4. Run \`python compare.py --generate-log\`
5. Update \`outbox/STATUS.md\`
6. Run \`python3 scripts/lab_state.py\`
7. Update \`memory/NEXT.md\` — strikethrough completed items
8. If new hard rule → add to \`memory/HARD_RULES.md\`
9. If paper insight → add to \`memory/INSIGHTS.md\` Paper References table
10. \`python3 scripts/agent_lock.py release $AGENT_ID\`
11. \`git add -A && git commit -m "exp_NNN: <description>" && git push\`

## Git safety for parallel agents

- **ALWAYS \`git pull --rebase\`** before committing
- Each agent writes to its OWN experiment dir (\`results/exp_NNN/\`)
- For shared files (HARD_RULES, NEXT, INSIGHTS): pull first, append only
- If git push fails: pull --rebase, retry (don't force push)

## GPU Training

For \`cuda: true\` experiments: \`bash scripts/manage_pod.sh\`
Required env vars (in ~/.bashrc): RUNPOD_API_KEY, RUNPOD_POD_ID

## Paper Research

1. Search HF Papers via MCP tools (Papers Semantic Search)
2. Fetch papers: \`https://huggingface.co/papers/ARXIV_ID.md\`
3. Write summary to \`research/<topic>.md\`
4. Propose experiment configs with paper citations

## Rules

- Read \`memory/HARD_RULES.md\` FIRST — never repeat known failures
- Do NOT modify train.py, train_hover.py, train_racing.py, compare.py, plot.py
- All experiment params go in YAML configs, not code changes
- Do NOT manually edit \`memory/EXPERIMENT_LOG.md\` — use compare.py --generate-log
- ALWAYS commit and push before exiting
- ALWAYS claim tasks before working on them
- ALWAYS release tasks when done
- If stuck after research, document what you tried and exit cleanly
PROMPT
)"
