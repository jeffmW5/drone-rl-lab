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
- **Scope:** GPU direct-racing runs with current 3600-7200s budgets
- **Supported by:** `exp_060` reward still climbing at budget end; literature comparison in `research/stochastic_deployment_gap.md`
- **Counterevidence:** `exp_068` doubled budget (7200s) to 42.84 reward but deterministic benchmark was flat (1.67s, 0 gates vs exp_067's 1.70s, 0 gates). Training reward gains did NOT translate to benchmark improvement.
- **Confidence:** low (weakened by exp_068)
- **Last reviewed:** 2026-03-28
- **Next falsification test:** If exp_069 (larger network, same budget) shows benchmark gates, then capacity was the bottleneck, not training duration.

## HYP-003
- **Statement:** Asymmetric critic improves training efficiency in this family, but by itself is insufficient to produce matched mid-air benchmark gains.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` line with privileged critic observations
- **Supported by:** `exp_059` training vs `exp_056` baseline; literature on asymmetric actor-critic
- **Counterevidence:** `exp_059` still achieved 0 gates on the matched mid-air benchmark
- **Confidence:** medium
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Combine asymmetric critic with a separate mean-policy stabilization change and show that benchmark performance then moves.

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

## HYP-006
- **Statement:** The 2×64 MLP lacks capacity for the deterministic mean to represent precise gate navigation, causing the persistent deployment gap.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` policies with 55D obs and 4D action
- **Supported by:** Swift (Nature 2023) uses 2×128 and achieves real-world gate navigation; our 2×64 (16K params) may be insufficient for 55D→4D mapping with the precision needed for gates; stochastic policy navigates (reward 43) but mean does not
- **Counterevidence:** none yet; exp_069 will test this
- **Confidence:** medium
- **Last reviewed:** 2026-03-28
- **Next falsification test:** exp_069 with hidden_size=128 (48K params). If benchmark gates improve, capacity was limiting. If unchanged, the gap is structural (reward/exploration), not capacity.
