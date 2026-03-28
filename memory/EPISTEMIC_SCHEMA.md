# Epistemic Schema

> Purpose: keep the repo from turning one-off interpretations into doctrine.
> Use this file to decide where a claim belongs and how strongly it should be stated.

## Claim Types

### 1. Fact
- A direct observation from code, config, benchmark output, metrics, or paper text.
- No causal explanation baked in.
- Example: "`exp_060` reached 28.02 mean training reward and 0 gates on the mid-air benchmark."

### 2. Hypothesis
- A proposed explanation for one or more facts.
- Must be testable and falsifiable.
- Example: "The deterministic deployment mean is unstable because the learned distribution is still too wide."

### 3. Tentative Lesson
- A reusable heuristic supported by more than one result, or by one clean ablation plus no strong counterexample.
- Still scoped, still revisable.
- Example: "Confounded ablations produce misleading repo memory."

### 4. Hard Rule
- A process invariant, safety constraint, or evaluation integrity rule.
- Should remain true even if the science changes.
- Example: "Do not compare rewards across different reward definitions."

## Required Fields For Any Durable Empirical Claim

Use this template in `FACTS.md`, `HYPOTHESES.md`, and `TENTATIVE_LESSONS.md`.

```markdown
## ID: <short id>
- **Statement:** <the claim>
- **Type:** fact | hypothesis | tentative_lesson
- **Scope:** <exact experiment family, controller, benchmark, reward regime, etc.>
- **Supported by:** <experiment ids, files, papers>
- **Counterevidence:** <known contradictions or "none yet">
- **Confidence:** low | medium | high
- **Last reviewed:** YYYY-MM-DD
- **Next falsification test:** <what result would weaken or overturn this claim?>
```

## Promotion Rules

- **Observation -> Fact:** allowed after one direct measurement with a source.
- **Fact -> Hypothesis:** allowed when you add an explanation. Keep them separate.
- **Hypothesis -> Tentative Lesson:** requires replication, a clean ablation, or convergent evidence from multiple sources.
- **Anything -> Hard Rule:** only if it is about repo integrity, evaluation protocol, or a true invariant.

## Demotion Rules

- If a "lesson" is contradicted by a later experiment, demote it back to a hypothesis.
- If a "hard rule" depends on a specific reward setting, controller, benchmark, or budget, it is not a hard rule.
- If a claim was formed from a confounded experiment, lower confidence or split it into multiple scoped claims.

## Language Rules

Prefer:
- "Current best explanation"
- "Within this experiment family"
- "Evidence is mixed"
- "Supports but does not prove"

Avoid unless strongly replicated:
- "The bottleneck is ..."
- "X is exhausted"
- "Y does not work"
- "We now know ..."

## Review Cadence

- Review the top claims after every 5 completed experiments, or immediately after a contradiction appears.
- Update `Last reviewed` when touching a claim.
- Add falsified claims to `BELIEF_AUDIT.md` rather than silently deleting them.
