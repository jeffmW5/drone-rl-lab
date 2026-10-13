# Review 0110: first fine reward pair and requested tenfold duration

Last reviewed completed trial index: 882. Date: 2026-10-13 UTC. Review0109 ended at879;880-882 are the next three completed trials. The healthy serial daemon remains unpaused and active. No duplicate GPU job, restart, build or trainer interruption was used.

## Observations

Trial880 is the previously launched neutral-warm reset5 recovery control at seed70502, not a fine-distance reward arm. It completed262144 transitions in431.67s, selected its initial checkpoint, and ended with24 zone escapes, zero ground hover and zero peak hold; first pre-reset failure occurred at3.76s. Its paired reset6 arm remains queued and no second-seed recovery comparison is claimed.

The first fine-distance pair881/882 used ground reset3, the same initial checkpoint759, seed70601,128 environments,65536 transitions, observation2/control2/physics5/mask2 and a600s budget. Both completed all requested transitions in107.82/108.48s; neither is budget-censored. Initial policy tensors, physical metrics and trajectories match exactly, and the core source hashes match. Only the frozen reward definition differs. Both selected best_step0.

| Final metric | Reward15 control881 | Fine reward16 treatment882 |
|---|---:|---:|
| True zone escapes | 0 | 24 |
| Takeoffs | 24 | 24 |
| Sustained ground hover | 0 | 0 |
| Peak ground hold | 0.16s | 0.18s |
| Endpoint altitude error | 57.834mm | 733.094mm |
| Maximum speed in saved samples | 1.474m/s | 2.135m/s |
| Maximum tilt in saved samples | 7.527deg | 8.560deg |

The treatment's altitude endpoint is after reset and is not evidence of convergence. Its small hold increase does not compensate for added failures. Twenty-four repeated lanes are still one physical condition, not independent successes/failures. The new reward is numerically graduated, but the first short trained pair regresses. No reward16 promotion is justified.

Saved states are finite and the counterfactual reward16 core is nonnegative across the available samples. This does not prove absence of an inverted/high-speed edge case throughout the full training distribution. The known valid flipped-state reward taper counterexample from0109 remains preserved. Current limited trajectory evidence does not identify that edge as the cause of this failed arm. C0 square boundaries, normalized-quaternion assumptions and lower clipping on ground remain explicit limits. No universal reward/action sign reversal or physics defect is established.

## Methods read

Fresh HF search and2311.13081 metadata/markdown succeeded. Both2006.05990 HF404s are retained with original fallback. The installed official huggingface-papers skill was used; retrieval dates/URLs, exact versions and author repository provenance are recorded in retrieval.json/original_sources.json. Existing original methods were reread for this new local clipping/distance hypothesis, not claimed as a new publication or causal proof.

- [Learning to Fly in Seconds v2](https://arxiv.org/html/2311.13081v2), IV/V: actuator and initial-state coverage, action-history/privileged-critic assumptions, episode duration and replicated behavioral evaluation are relevant checks. Its untethered TD3 results do not validate this full-state PPO tether task. No asymmetric-critic or actuator-history factor is added here.
- [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study v1](https://arxiv.org/html/2006.05990v1),3.2-3.5: architecture/action scale, observation/value normalization, advantage/value-loss implementation and temporal data collection support checking actual optimization contracts. These generic results do not prescribe our reward16, guarantee that longer exposure improves hovering or show that incoming reward clipping caused the failure.

Published results are not this simulator's results; no author code was executed. No additional coefficient, learner or physics modification is justified from one failed reward seed before a matched duration/replication test.

## User-directed longer test

The user explicitly requested "Train for10x longer." New frozen priority queue methodology_queue_v000_user_abca_0_tenfold_reward.json increases requested transitions from65536 to655360, with1800s per-trial budget, fixed128 environments, unchanged50Hz control and evaluation every16384 transitions. This corresponds to102.4 nominal simulated seconds per environment, with resets still included. The wall-clock cap is separate from the requested transition count and any censoring must be reported.

Three reward15/16 seed pairs retain the original initial source, fresh Adam and all other settings. Each long run retains optimizer/environment state internally; it recreates the short prefix rather than restarting optimization every65k. First-seed short reference paths and hashes are frozen. Verify exact65536-transition policy equality before attributing later differences to duration; expected equality is not assumed from config alone. Remaining short and recovery plans stay immutable and pending. Already-active883 drains naturally before the new queue takes priority.

This duration extension is explicitly requested despite the first short-arm counterevidence; it is not evidence-based promotion of reward16. Longer training also failed to solve hover in the earlier completed duration/mask factorial. Keep all periodic regressions and final failures. Require replicated genuine ground sustained hover or credible motion/hold improvement without extra failures/lost takeoffs before promotion, and never compare changed reward magnitudes as one leaderboard. If long behavior regresses, diagnose saved trajectories/credited updates before another sweep.

Evidence:0109/fine_reward contains the first-pair immutable addenda and corrected dispatch record;0110/evaluation_manifest.json and results/ preserve all three completed source artifacts. tenfold_duration/ contains the frozen plan and queue generator. Reserved audit/hardware seeds remain unused. The original sustained1mm obstacle-navigation benchmark and hardware precision are not demonstrated; tether/contact/material/attachment remain uncalibrated.
