# Belief Audit

> High-risk claims that were over-promoted in earlier repo memory.
> Keep this file short and current. Its job is to stop old narratives from
> silently becoming permanent truth.

## AUDIT-001
- **Statement:** "Reward / obs / curriculum are exhausted."
- **Type:** hypothesis_under_review
- **Why flagged:** This was promoted before `exp_059` (asymmetric critic) and `exp_064` (entropy annealing) were tested, and `exp_057` changed both representation and reward strength at once.
- **Current status:** Demoted. Strong reward sweeps look exhausted, but major training-side structure is not.
- **Confidence:** low
- **Last reviewed:** 2026-03-27
- **What would falsify the demotion:** Multiple clean structural experiments fail after proper training, checkpointing, and evaluation.

## AUDIT-002
- **Statement:** "The domain gap is the bottleneck."
- **Type:** hypothesis_under_review
- **Why flagged:** Recent direct-race failures also occur on the easier mid-air benchmark, so race-start mismatch cannot explain everything.
- **Current status:** Narrowed. Race-start mismatch is real, but it is not sufficient as a global explanation.
- **Confidence:** low
- **Last reviewed:** 2026-03-27
- **What would falsify the demotion:** A controller that succeeds mid-air but collapses only from the real race start.

## AUDIT-003
- **Statement:** "The deployment gap is the bottleneck."
- **Type:** hypothesis_under_review
- **Why flagged:** `exp_061` and `exp_062` support a real deployment gap, but stochastic deployment alone does not yield reliable gates.
- **Current status:** Kept as a partial explanation, not an exclusive one.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **What would falsify the partial framing:** If stochastic or temperature-scaled deployment suddenly becomes robust without any training changes.

## AUDIT-004
- **Statement:** "Body-frame observations failed."
- **Type:** hypothesis_under_review
- **Why flagged:** `exp_057` reduced `progress_coef` from 50 to 20 while adding body-frame observations, so the result is confounded.
- **Current status:** Demoted. Body-frame observations remain promising but not cleanly isolated.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **What would falsify the demotion:** A clean matched ablation where only `body_frame_obs` changes and performance still degrades.

## AUDIT-005
- **Statement:** "`survive_coef=0.05` is optimal."
- **Type:** hypothesis_under_review
- **Why flagged:** This is only supported within the tight-logstd, short-budget, current-reward family.
- **Current status:** Scope-limited. Useful local setting, not a global optimum claim.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **What would falsify the scope limit:** Reproducing the same optimum across different policy variance schedules, critics, and longer budgets.
