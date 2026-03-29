# Agentic Harness Upgrade Plan

**Date:** 2026-03-28
**Scope:** Upgrade the outer loop of `drone-rl-lab` into a typed, resumable, eval-driven agent harness without rewriting the core training code.

## Why this exists

The repo already has the beginnings of a harness:
- queueing via `inbox/INBOX.md`
- state snapshots via `state/current.json`
- orchestration via `scripts/pipeline.sh`
- agent coordination via `scripts/agent_lock.py`

The weak spots are that queue mutation is Markdown-first, the pipeline is not fully resumable, agent context is broad rather than selective, and there is no dedicated eval suite for the harness itself.

This plan keeps `train.py`, `train_hover.py`, and `train_racing.py` mostly intact and hardens the orchestration layer around them.

## Guiding principles

1. Prefer simple workflows before adding more autonomy.
2. Keep Markdown as a human-facing view, not the source of truth.
3. Add typed artifacts before adding more agents.
4. Make interrupted runs resumable.
5. Evaluate the harness itself, not just policy metrics.
6. Add multi-agent parallelism only for independent sidecar work.

## Current repo seams

- `scripts/task_queue.py` parses and mutates Markdown directly.
- `scripts/lab_state.py` produces a useful but shallow snapshot.
- `scripts/pipeline.sh` is a linear shell flow with limited recovery.
- `CLAUDE.md` and memory files provide rich context, but startup loading is broad.
- Benchmarking, documentation, and memory promotion are only loosely coupled.

## Target architecture

### Source of truth

Add durable JSON artifacts:
- `schemas/task.schema.json`
- `schemas/job_state.schema.json`
- `schemas/run_trace.schema.json`
- `schemas/result_summary.schema.json`
- `schemas/memory_promotion.schema.json`
- `inbox/tasks/*.json`
- `state/jobs/*.json`
- `state/runs/<run_id>/*`

Markdown files remain in the repo, but become rendered views:
- `inbox/INBOX.md`
- `outbox/STATUS.md`

### Execution model

Replace the current mostly linear pipeline with a resumable job runner:
- claim task
- prepare environment
- train
- capture provenance
- benchmark
- document
- propose memory updates
- publish status
- complete or block

Every run should record:
- `job_id`
- `run_id`
- `attempt`
- `step`
- `status`
- `heartbeat`
- `lease_expires_at`
- `resume_from_step`
- `last_error`
- `artifacts`

### Context model

Keep `CLAUDE.md` as the charter, but move specialized procedures into skills:
- `skills/runpod_ops/`
- `skills/benchmark_triage/`
- `skills/experiment_design/`
- `skills/report_writer/`
- `skills/memory_promotion/`

Each run should emit a small handoff artifact so a fresh session can resume without rereading the entire repo.

### Tool model

Add a semantic tool layer via a local CLI or MCP server:
- `get_lab_context`
- `claim_next_task`
- `start_or_resume_run`
- `run_benchmark_and_summarize`
- `propose_memory_updates`
- `publish_status_snapshot`

These tools should return concise, schema-validated outputs.

### Eval model

Add a harness eval suite:
- `harness_evals/fixtures/`
- `harness_evals/cases/`
- `harness_evals/graders/`
- `harness_evals/run_eval.py`

The first evals should cover:
- choosing the next experiment
- honoring hard rules
- avoiding repeated failed work
- reclaiming stale claims
- handling missing controllers or missing benchmarks
- producing correct documentation from artifacts
- knowing when to stop and escalate

## Phased implementation

### Phase 1 -- Typed artifacts first

Goal: make JSON the source of truth for tasks, job state, traces, and summaries.

Deliverables:
- create `schemas/`
- create `inbox/tasks/`
- add validation helpers
- dual-write or render `inbox/INBOX.md` from JSON

Definition of done:
- no core queue mutation depends on raw Markdown editing
- artifacts validate before they are written

### Phase 2 -- Resumable job runner

Goal: replace the brittle linear flow with a stateful runner.

Deliverables:
- add `scripts/run_job.py`
- add `scripts/job_store.py`
- add step transitions and resume logic
- write step-by-step run artifacts into `state/runs/<run_id>/`

Definition of done:
- killing the process mid-run and rerunning resumes from the last completed step

### Phase 3 -- Skills and handoffs

Goal: reduce prompt bloat and make fresh sessions recover faster.

Deliverables:
- thin `CLAUDE.md`
- add workflow skills
- add `handoff.json` or `session_brief.md` per run

Definition of done:
- a fresh agent session can recover a run from the handoff artifact plus one lab summary

### Phase 4 -- Semantic tools

Goal: stop making agents parse raw files and shell output for common actions.

Deliverables:
- implement a local tool layer or MCP server
- expose the common harness operations
- enforce concise output contracts

Definition of done:
- the main orchestration path can run through semantic tools instead of ad hoc parsing

### Phase 5 -- Harness evals

Goal: regression-test agent behavior before scaling up autonomy.

Deliverables:
- add eval fixtures, cases, graders, and runner
- support repeated trials
- capture both outcome and trace failures

Definition of done:
- changes to prompts, tools, or workflow can be checked against a stable harness eval suite

### Phase 6 -- Observability and budgets

Goal: make cost, failure, and retry behavior auditable.

Deliverables:
- write per-run traces and metrics
- add retry ceilings and task-class budgets
- surface manual intervention count and runtime cost estimates

Definition of done:
- the repo can answer what happened, why, and what it cost from artifacts alone

### Phase 7 -- Memory promotion pipeline

Goal: turn evidence into memory updates in a controlled way.

Deliverables:
- add `memory/promotions/`
- link every promoted claim back to concrete experiment artifacts
- separate observations from interpretation

Definition of done:
- memory updates are traceable and not just narrative copy-paste

### Phase 8 -- Selective multi-agent work

Goal: use parallelism only where it helps.

Deliverables:
- sidecar patterns for literature review, benchmark comparison, and report drafting
- ownership rules so only one agent owns a run's write path

Definition of done:
- throughput improves without coordination bugs or conflicting writes

## Recommended file layout

```text
docs/
  agentic_harness_upgrade_plan.md
schemas/
  task.schema.json
  job_state.schema.json
  run_trace.schema.json
  result_summary.schema.json
  memory_promotion.schema.json
inbox/tasks/
state/jobs/
state/runs/
skills/
  runpod_ops/
  benchmark_triage/
  experiment_design/
  report_writer/
  memory_promotion/
harness_evals/
  fixtures/
  cases/
  graders/
```

## Best order of implementation

1. Phase 1 -- typed artifacts
2. Phase 2 -- resumable job runner
3. Phase 3 -- skills and handoffs
4. Phase 4 -- semantic tools
5. Phase 5 -- harness evals
6. Phase 6 -- observability and budgets
7. Phase 7 -- memory promotion
8. Phase 8 -- selective multi-agent work

## What to start tonight

If the goal is to get agents productively working on this tonight, do not try to build the whole system at once. Start with the first three slices below.

### Tonight slice A -- typed task/job scaffolding

Create:
- `schemas/task.schema.json`
- `schemas/job_state.schema.json`
- `inbox/tasks/`
- `scripts/validate_artifact.py`

Goal:
- represent one queue item as JSON without breaking the existing Markdown inbox

### Tonight slice B -- resumable runner skeleton

Create:
- `scripts/run_job.py`
- `state/jobs/`
- `state/runs/`

Goal:
- support claim -> train -> benchmark -> document as explicit steps, even if some steps still call existing scripts

### Tonight slice C -- harness eval skeleton

Create:
- `harness_evals/cases/`
- `harness_evals/fixtures/`
- `harness_evals/run_eval.py`

Goal:
- add at least three eval cases:
  - choose the next experiment correctly
  - reject a repeated failed experiment
  - recover from a stale claim

## Good first agent tickets

1. "Define `task.schema.json` and `job_state.schema.json`, plus a validator helper."
2. "Add `inbox/tasks/exp_066.json` from the current queue entry and render a matching Markdown view."
3. "Create `scripts/run_job.py` that can read one task JSON file and write step state into `state/jobs/`."
4. "Create the first three harness eval fixtures and a simple eval runner."
5. "Draft `skills/benchmark_triage/SKILL.md` and `skills/report_writer/SKILL.md`."

## Success criteria

- Markdown is no longer the system of record for orchestration.
- Interrupted runs resume from artifacts.
- Fresh sessions recover from a small handoff, not from rereading the whole repo.
- Harness behavior can be regression-tested.
- Future autonomy is earned by evals, not assumed.

## Source inspirations

- Anthropic: "Building effective agents" (2024-12-19)
- Anthropic: "Writing effective tools for agents" (2025-09-11)
- Anthropic: "Effective context engineering for AI agents" (2025-09-29)
- Anthropic: "Effective harnesses for long-running agents" (2025-11-26)
- Anthropic: "Demystifying evals for AI agents" (2026-01-09)
- OpenAI docs on tools, structured outputs, and long-running/background agent workflows
