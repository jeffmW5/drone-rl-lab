# Tentative Lessons

> Reusable lessons with some support, but still scoped and revisable.

## LESSON-001
- **Statement:** Confounded ablations are a major source of bad repo beliefs.
- **Type:** tentative_lesson
- **Scope:** Experiment interpretation and memory promotion across the whole repo
- **Supported by:** `exp_057` changing both representation and reward strength; earlier reward narratives promoted from mixed interventions
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Show that a confounded change still yields an explanation robust to later clean ablations.

## LESSON-002
- **Statement:** Benchmark and training metrics must be stored together because large training-reward improvements can coexist with flat benchmark outcomes.
- **Type:** tentative_lesson
- **Scope:** Racing line, especially direct `RaceCoreEnv`
- **Supported by:** `exp_044`, `exp_052`, `exp_056`, `exp_058`, `exp_060`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Find a family where reward monotonically predicts benchmark gains across changed reward definitions.

## LESSON-003
- **Statement:** Mid-air benchmark results should be treated as a subproblem signal, not as a complete diagnosis of race-start generalization.
- **Type:** tentative_lesson
- **Scope:** Direct-racing evaluation logic
- **Supported by:** `exp_046` to `exp_060` all use mid-air evaluations for fast iteration; separate race-start experiments such as `exp_054` show additional failure modes
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Show that mid-air success always transfers to race-start success without additional training.

## LESSON-004
- **Statement:** Final checkpoints can be misleading when training peaks and then collapses, so best-checkpoint saving should be preferred for decision-making.
- **Type:** tentative_lesson
- **Scope:** Direct-racing training loop and result interpretation
- **Supported by:** `exp_049` collapse narrative, current `train_racing.py` saves final checkpoint only
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Add best-checkpoint saving and show it never changes benchmark conclusions.

## LESSON-005
- **Statement:** Evaluation tooling must match both the training start-state regime and the deployed checkpoint architecture, or benchmark conclusions become confounded.
- **Type:** tentative_lesson
- **Scope:** Direct-racing evaluation for random-gate-start experiments and asymmetric checkpoints
- **Supported by:** `exp_059` requiring `level2_midair` rather than race-start evaluation, plus the actor-only asymmetric checkpoint patch in `attitude_rl_generic.py`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Show a mismatched evaluator or loader producing the same benchmark outcome as the matched path across multiple asymmetric experiments.
