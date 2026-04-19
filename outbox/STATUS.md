# Status -- Last Updated 2026-04-19

## Latest Result

**exp_071 -- Observation Normalization** completed 2026-04-19.
- Training: **45.63 mean reward** in **4.63M** steps over the full **7200s** budget
- Benchmark: **not verified** — the current generic benchmark/controller path produced zero parsed runs for `exp_071`
- **Key finding:** observation normalization improved training reward relative to `exp_069`, but no deployment conclusion should be drawn yet

## Current Read On exp_071

- `exp_071` is a clean training-side positive on reward, not yet a racing benchmark result.
- The right status is **mixed / unresolved**, not success or failure.
- Until a parsed benchmark exists, `exp_069` remains the last benchmark-backed positive signal in this direct-racing line.

## Throughput Finding (same pod / same config family)

Measured after `exp_071` on the same RTX 3090 pod:

- `env_samples_per_s`: **803.54**
- `rollout_samples_per_s`: **750.71**
- `policy_samples_per_s`: **11549.66**
- `ppo_update_samples_per_s`: **14443.32**

Interpretation: env stepping still dominates. Low GPU utilization is expected under the current stack because rollout is env-bound, not optimizer-bound.

## Research: Deployment Gap Literature Review (2026-03-28)

Literature review of 10+ papers targeting the stochastic-to-deterministic deployment gap.
See `research/deployment_gap_mean_policy.md` for full details.

**Key finding:** Two critical issues identified in our setup:
1. **No observation normalization** — the "What Matters" study (250K agents, ICLR 2021) identifies this as critical. Our Agent has NONE. Swift normalizes. SimpleFlight normalizes.
2. **No action smoothness penalty** — CAPS (ICRA 2021) and SimpleFlight both show action-difference penalties force the mean to be physically coherent. We have d_act_th/d_act_xy in the reward code but disabled (0.0) since exp_069.

**Theoretical explanation:** Montenegro et al. (ICML 2024) formally prove that PPO optimizes stochastic expected return — the mean is a side effect, not the objective. Without mechanisms to force mean quality (entropy annealing, smoothness penalties, obs normalization), the gap is expected.

## Next Steps

1. Repair or replace the generic benchmark/controller path so `exp_071` can be evaluated with real parsed runs.
2. Keep `exp_069` as the last benchmark-backed comparison point until `exp_071` is actually benchmarked.
3. Continue with `exp_072` if queue momentum matters more than immediate evaluation repair, but keep `exp_071` explicitly unresolved.
4. Use `docs/throughput_findings_2026_04_19.md` as the current reference for rollout bottleneck discussion.
