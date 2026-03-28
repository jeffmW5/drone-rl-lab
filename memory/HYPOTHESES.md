# Hypotheses

> Explanations we are actively testing. These are not facts.

## HYP-001
- **Statement:** A real stochastic-to-deterministic deployment gap exists in the direct-racing line.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` policies around `exp_056` to `exp_062`
- **Supported by:** `exp_056`, `exp_060`, local `exp_061`, local `exp_062`
- **Counterevidence:** Stochastic and temperature-scaled deployment still fail to produce reliable gates.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Show a deterministic evaluation during training that tracks benchmark while stochastic deployment adds no benefit.

## HYP-002
- **Statement:** The direct-racing line is still materially undertrained, and mean-policy convergence has not completed by the current 3600s budget.
- **Type:** hypothesis
- **Scope:** GPU direct-racing runs with current 3600s budgets
- **Supported by:** `exp_060` reward still climbing at budget end; literature comparison in `research/stochastic_deployment_gap.md`
- **Counterevidence:** Some longer direct-racing runs plateau without solving navigation.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Run a longer-budget experiment with best-checkpoint saving and see whether deterministic benchmark meaningfully improves.

## HYP-003
- **Statement:** Asymmetric critic training may improve value estimation enough to stabilize the deployed mean policy.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` line with privileged critic observations
- **Supported by:** Literature review; `exp_059` remains untested locally
- **Counterevidence:** none yet inside this repo
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Train `exp_059` and compare against matched non-asymmetric baseline.

## HYP-004
- **Statement:** Tight logstd helps late deterministic deployment but may hurt early exploration and attribution if applied from the start.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` runs using `max_logstd`
- **Supported by:** `exp_046` as best deterministic benchmark reference; stochastic and temperature-scaled follow-ups suggest wider distributions still contain useful trajectories
- **Counterevidence:** none cleanly isolated yet
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Compare clamped vs unclamped training with matched budgets, checkpointing, and deployment evaluation.

## HYP-005
- **Statement:** Body-frame gate observations are still promising, but their repo evidence is confounded by simultaneous reward-strength changes.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv`, body-frame observation experiments
- **Supported by:** `exp_057` had one gate passage but changed both observations and `progress_coef`
- **Counterevidence:** `exp_060` combining body-frame observations with strong progress still failed on the benchmark
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Run a clean ablation where body-frame observations are toggled and reward weights are held fixed.
