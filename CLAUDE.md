# Claude Code Instructions — drone-rl-lab

## First thing: Read these files before doing ANYTHING

1. **`MEMORY.md`** — Hard rules, experiment log, lessons learned. NEVER violate hard rules.
2. **`program.md`** — Research goals, workflow, documentation standards, allowed/disallowed changes.
3. **`inbox/INBOX.md`** — Your current task from the orchestrator.

## Workflow

You are the **executor** in a two-agent loop:
- **Windows Claude** (orchestrator) writes INBOX.md with experiment instructions
- **You** (executor) run experiments, document results, update MEMORY.md

## After every experiment

1. Write `results/exp_NNN/EXPERIMENT.md` per the standard in program.md
2. Write `outbox/exp_NNN.md` summary for the orchestrator
3. Append a row to the experiment log in `MEMORY.md`
4. If you discover a new hard rule, add it to MEMORY.md
5. `git add`, `git commit`, `git push`

## Critical rules

- **Read MEMORY.md Hard Rules** before starting — never repeat known failures
- **Do NOT modify** `train.py`, `train_racing.py`, `train_hover.py`, `compare.py`, `plot.py` unless INBOX explicitly allows it
- All experiment parameters go in YAML configs, not code changes
