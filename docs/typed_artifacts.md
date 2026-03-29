# Typed Artifacts -- Reference Guide

**Added:** 2026-03-28
**Status:** Phase 1 complete, Phase 2 skeleton in place

## What changed

The harness now has a typed JSON artifact layer alongside the existing
Markdown queue. JSON tasks are the source of truth for new experiments.
The Markdown `inbox/INBOX.md` remains readable and usable but can be
regenerated from JSON.

## New files

### Schemas

| File | Purpose |
|------|---------|
| `schemas/task.schema.json` | Defines the shape of a task artifact |
| `schemas/job_state.schema.json` | Defines the shape of a job execution state |

### Task artifacts

| Directory | Contents |
|-----------|----------|
| `inbox/tasks/*.json` | One file per task, named `<task_id>.json` |
| `state/jobs/*.json` | One file per job execution |
| `state/runs/<run_id>/` | Per-run artifacts (future) |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_artifact.py` | Validate JSON artifacts against schemas |
| `scripts/task_store.py` | Programmatic read/write for task artifacts |
| `scripts/render_inbox.py` | Regenerate `INBOX.md` from JSON + legacy entries |
| `scripts/run_job.py` | Resumable job runner (Phase 2 skeleton) |

## How to use

### Validate artifacts

```bash
# Validate a single file (auto-detects schema from path)
python scripts/validate_artifact.py inbox/tasks/exp_066.json

# Validate all known artifacts
python scripts/validate_artifact.py --all

# Force a specific schema
python scripts/validate_artifact.py --schema task inbox/tasks/exp_066.json
```

### Read and write tasks programmatically

```python
from task_store import TaskStore

store = TaskStore()

# List all tasks sorted by priority
tasks = store.list_all()

# Get the next actionable task
task = store.get_next()

# Claim a task
store.claim("exp_066", agent_id="my-agent")

# Complete a task with results
store.complete("exp_066",
    result_summary="MIXED -- 2/15 det gates",
    result_diagnosis="Network capacity helps but 87% det failure persists",
    result_path="results/exp_066_asym_entropy_annealing/")
```

### Regenerate INBOX.md

```bash
# Preview what the rendered INBOX.md would look like
python scripts/render_inbox.py

# Overwrite INBOX.md from JSON + legacy
python scripts/render_inbox.py --write

# Check if INBOX.md is stale
python scripts/render_inbox.py --check
```

### Run a job (Phase 2)

```bash
# Dry run -- show steps without executing
python scripts/run_job.py exp_066 --dry-run

# Real run
python scripts/run_job.py exp_066 --agent-id my-agent

# Resume an interrupted job
python scripts/run_job.py exp_066 --resume
```

## Source of truth

| Data | Source of truth | Compatibility view |
|------|-----------------|--------------------|
| Active task definitions | `inbox/tasks/*.json` | `inbox/INBOX.md` (rendered) |
| Legacy done tasks | `inbox/INBOX.md` | stays in Markdown |
| Job execution state | `state/jobs/*.json` | none yet |
| Lab summary | `state/current.json` | unchanged |

## What still uses Markdown as source of truth

- **Legacy completed experiments** (exp_057 and earlier) remain Markdown-only
  in `inbox/INBOX.md`. They do not need migration -- they are historical.
- **NOTE entries** in the INBOX are not tasks and stay in Markdown.
- `task_queue.py` and `agent_lock.py` still read/write Markdown directly.
  They continue to work unchanged.

## Migration path

New tasks should be created as JSON in `inbox/tasks/`. The compatibility
layer ensures both systems coexist. Over time:

1. New tasks: always create as JSON first.
2. `render_inbox.py --write` keeps the Markdown view fresh.
3. `task_queue.py` remains operational for agents that use it.
4. Future work: wire `agent_lock.py` claim/release to use `task_store.py`.

## Job runner steps

The `run_job.py` runner walks through these steps in order:

```
claimed -> preparing -> training -> capturing_provenance
        -> benchmarking -> documenting -> publishing -> done
```

Each step writes state to `state/jobs/<job_id>.json`. If the process is
killed, re-running with `--resume` picks up from the last incomplete step.

Terminal states: `done`, `failed`, `blocked`.

## Remaining work

- **Phase 2 completion:** Wire `run_job.py` into the actual training loop
  (currently only `--dry-run` exercises all steps without GPU).
- **Phase 3:** Skills and handoff artifacts for fresh session recovery.
- **Phase 4:** Semantic tool layer (MCP or CLI) wrapping these primitives.
- **Phase 5:** Harness eval suite.
