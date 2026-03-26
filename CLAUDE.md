# Claude Code Instructions -- drone-rl-lab

## On session start (do this FIRST, every time)

```bash
cd /media/drone-rl-lab
git pull 2>/dev/null || true
```

Then read these files in order:
1. **`memory/HARD_RULES.md`** -- Absolute constraints. NEVER violate these.
2. **`memory/EXPERIMENT_LOG.md`** -- Full experiment history.
3. **`memory/INSIGHTS.md`** -- Kaggle targets, benchmarks, paper references.
4. **`memory/NEXT.md`** -- Current priorities and open questions.
5. **`inbox/INBOX.md`** -- Your current task queue.
6. **`outbox/STATUS.md`** -- Latest results summary.
7. **`state/current.json`** -- Canonical machine-readable lab state, if present.

## Workflow

You are a **single autonomous agent** that handles the research loop:
analyze -> detect plateau -> research papers -> design experiment -> train ->
benchmark -> document -> commit -> loop.

Work fully autonomously. When done making meaningful progress (or blocked),
commit, push, and exit cleanly.

## Decision loop

After reading memory, decide what to do:

### If INBOX has actionable tasks -> claim and execute them
Actionable statuses are `[READY]`, `[IMPLEMENTED]`, or the legacy `[NEXT]` /
`[QUEUED]`. Statuses like `[IN PROGRESS]` or `[CLAIMED:agent-id]` are already
being worked.

1. Process in order: first `[NEXT]` / `[READY]`, then `[IMPLEMENTED]` /
   `[QUEUED]` with no unmet dependencies.
2. If task type is `research`: use `/research` or manually search papers.
3. If task type is training: run the experiment, benchmark, document.
4. Mark `[DONE]`, refresh state, continue.

### If INBOX is empty -> analyze and self-direct
1. Check for **plateau**: 3+ consecutive experiments with no improvement on
   the primary metric (benchmark finish rate, gate count, or lap time).
2. **If plateau detected** -> trigger paper research:
   - Identify the specific bottleneck from recent experiments.
   - Search Hugging Face Papers for solutions (MCP semantic search or fetch
     `https://huggingface.co/papers/ARXIV_ID.md`).
   - Read top 3-5 papers and extract actionable techniques.
   - Write a research summary to `research/<topic>.md`.
   - Design experiment configs based on findings.
   - Add them to INBOX and execute immediately.
3. **If no plateau** -> design the next experiment based on latest results:
   - Analyze what recent results suggest as the next step.
   - Design a config with a clear hypothesis (single-variable change preferred).
   - Add it to INBOX and execute.

## Parallel Agent Coordination

Multiple agents can run simultaneously. Use these tools to avoid conflicts:

```bash
# Check what other agents are doing
python3 scripts/agent_lock.py status

# Claim the next available task (atomic via git push)
python3 scripts/agent_lock.py claim <YOUR_AGENT_ID>

# Update your heartbeat + status
python3 scripts/agent_lock.py heartbeat <YOUR_AGENT_ID> --task "exp_NNN" --status "training"

# Release task when done (marks [DONE YYYY-MM-DD] in INBOX)
python3 scripts/agent_lock.py release <YOUR_AGENT_ID>

# Reclaim tasks from dead or missing agents (30+ min stale)
python3 scripts/agent_lock.py reclaim-stale <YOUR_AGENT_ID>
```

**Rules:**
- ALWAYS claim a task before working on it.
- ALWAYS release when done.
- ALWAYS `git pull --rebase` before committing.
- Each agent writes to its OWN `results/exp_NNN/` and `outbox/exp_NNN.md`.
- For shared files (`HARD_RULES`, `NEXT`, `INSIGHTS`): pull first, append only.
- Tasks with `[CLAIMED:agent-id]` or `[IN PROGRESS]` in INBOX are already running.
- `state/current.json` is the automation source of truth when it exists.

## Queue processing

1. Claim the first actionable task (`[NEXT]`, `[READY]`, `[IMPLEMENTED]`, or
   legacy `[QUEUED]` with no unmet dependencies).
2. Execute the task.
3. Release it through `scripts/agent_lock.py` so the queue is marked
   `[DONE YYYY-MM-DD]`.
4. Refresh `state/current.json`.
5. If more actionable tasks remain, continue processing.
6. When the queue is empty, enter self-directed mode.

## After every experiment

1. Write `results/exp_NNN/EXPERIMENT.md` per the standard in `program.md`.
2. Write `outbox/exp_NNN.md` summary.
3. Run `python3 scripts/capture_provenance.py --experiment exp_NNN`.
4. Run `python compare.py --generate-log` to update `memory/EXPERIMENT_LOG.md`.
5. Update `outbox/STATUS.md` with latest results.
6. Run `python3 scripts/lab_state.py`.
7. If you discover a new hard rule, add it to `memory/HARD_RULES.md`.
8. If paper insight was used, add it to `memory/INSIGHTS.md`.
9. Update `memory/NEXT.md` by striking through completed items.
10. `git add -A && git commit -m "exp_NNN: <description>" && git push`

## RunPod GPU Training

When a task requires GPU training (`cuda: true`), use the pod manager:

```bash
bash scripts/manage_pod.sh
```

Fresh pods are bootstrapped by `scripts/setup_runpod.sh` via the fork's Pixi GPU
environment. On-pod training commands should use `drone-rl-gpu-python ...`
rather than raw `python ...` so the local editable `lsy_drone_racing` checkout
and Pixi-managed dependencies are used consistently.

**Required env vars** (set in `~/.bashrc` on VM -- never in the repo):
- `RUNPOD_API_KEY` -- RunPod API key
- `RUNPOD_POD_ID` -- pod ID for "desirable_brown_mongoose" (`l4lu7w9i2rvfxm`)

The pod auto-stops after 4 hours via a safety timer in `setup_runpod.sh`.

## Paper Research

When hitting a plateau or when a research task is queued, search for papers:

1. **HF MCP tools**: use Papers Semantic Search (requires HF MCP server in
   `.claude/settings.json`).
2. **Direct fetch**: `https://huggingface.co/papers/ARXIV_ID.md` for any paper.
3. **Extract**: architectures, reward designs, training recipes, hyperparameters.
4. **Write**: summary to `research/<topic>.md` with proposed experiment configs.
5. **Update**: `memory/INSIGHTS.md` Paper References table.

Research tasks in INBOX use this format:

```markdown
### [NEXT] Research -- <topic>
- **Type:** research
- **Query:** <search terms>
- **Output:** research/<topic_slug>.md
```

**Requirements:**
- HF MCP server configured in `.claude/settings.json`
- `HF_TOKEN` env var set (in `.claude/settings.local.json` or `~/.bashrc`)

## Critical rules

- **Read `memory/HARD_RULES.md`** before starting -- never repeat known failures.
- **Do NOT modify** `train.py`, `train_hover.py`, `train_racing.py`, `compare.py`,
  or `plot.py` unless INBOX or the project owner explicitly allows it.
- All experiment parameters go in YAML configs, not code changes.
- **Do NOT manually edit** `memory/EXPERIMENT_LOG.md` -- it is auto-generated by
  `compare.py --generate-log`.
- **Benchmark metrics are primary for racing** -- finish rate, gates, and lap
  time on the standardized benchmark matter more than raw training reward.
- **Training reward is only a local signal** -- do not compare mean reward across
  very different reward definitions as if it were one global leaderboard.
- **Always commit and push** before ending your session -- results that are not
  pushed are lost.
- When designing experiments, always state a clear hypothesis.
- Prefer single-variable changes from the best previous experiment.
