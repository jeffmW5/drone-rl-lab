# Hard Rules

> These are integrity and process constraints, not empirical conclusions.
> Only put something here if violating it would corrupt evaluation, reproducibility,
> or repo safety. Empirical claims belong in `FACTS.md`, `HYPOTHESES.md`, or
> `TENTATIVE_LESSONS.md`.

1. **Keep experiment configs frozen.** Never retroactively edit a config that already has results attached to it.
2. **Benchmark outcomes outrank training reward for racing decisions.** Use training reward as a local diagnostic, not the primary selection metric.
3. **Do not compare reward magnitudes across different reward definitions as if they are on one leaderboard.**
4. **Separate observation from inference in every report.** Direct measurements are facts; explanations are hypotheses.
5. **Every durable empirical claim must include source, scope, and confidence.** If one of those is missing, it does not belong in long-term memory.
6. **A single experiment can create a hypothesis, not a hard rule.** Promotion requires replication or a true invariant.
7. **Preserve counterevidence.** Never hide or delete conflicting results because they complicate the story.
8. **Distinguish committed, uncommitted, and queued results.** Do not present local workspace artifacts as committed repo history.
9. **Never commit secrets or private infrastructure credentials.** API keys, pod IDs, and deploy key material stay outside the repo.
10. **`memory/EXPERIMENT_LOG.md` is auto-generated.** Update it via `python compare.py --generate-log`, not by manual edits.
11. **When new evidence contradicts a stored belief, demote or rewrite the belief promptly.**
12. **When uncertain, downgrade confidence instead of upgrading certainty.**
