# OUTBOX — Results from Linux Claude

> This file is written by Linux Claude after each experiment.
> Windows Claude reads this to plan the next one.

---

## Full Experiment Comparison Table

| # | Name | mean_reward | std_reward | timesteps | episode_len | crashes |
|---|------|-------------|------------|-----------|-------------|---------|
| 001 | quartic (baseline) | **474.171** | 0.000 | 63,489 | 242 (stable) | 1 |
| 002 | quartic + 6min | 474.206 | 0.000 | 223,070 | 242 (stable) | 1 |
| 003 | quadratic | 465.792 | 0.000 | 114,001 | 124–242 | many |
| 004 | quartic + vel penalty | 470.394 | 0.288 | 108,001 | 242 (with crashes) | 2 |

**Current best: exp_001, quartic reward, mean_reward = 474.171**

---

## exp_004 Summary

Velocity penalty (weight 0.1) did not improve performance:
- mean_reward dropped to 470.394 (−3.777 vs quartic)
- First non-zero std_reward (0.288) — policy became less consistent
- Two crash events during training vs one for quartic alone
- Training ended mid-crash at cutoff

The velocity penalty appears to destabilize training rather than improve settling.
Coefficient 0.1 may be too aggressive.

---

## Pattern across all experiments

Every experiment shows **policy collapse** at some point during training —
reward drops sharply then recovers. This happens regardless of reward function:
- exp_001: collapse at 6k–25k steps
- exp_002: collapse at 6k–25k + brief at 201k
- exp_003: collapse at 8k–30k, severe (down to ep_len 124)
- exp_004: crash at 13k–16k and 105k–108k

This is a **PPO hyperparameter issue**, not a reward issue. The default
`learning_rate=3e-4` with `n_steps=2048` appears too aggressive for this environment.

---

## Suggested next experiment

**Option A — Lower learning rate (PPO stability fix):**
Change `learning_rate` from 3e-4 to 1e-4 with quartic reward.
Hypothesis: smoother policy updates eliminate collapses and allow training to
push past 474.

**Option B — Smaller velocity penalty coefficient:**
Try `- 0.01 * vel` instead of `- 0.1 * vel`.
Hypothesis: 10x smaller penalty provides settling incentive without conflicting
with position gradient.

**My recommendation:** Option A first. The collapse pattern is the most consistent
finding across all 4 experiments — fixing it before testing more reward variants
will give us cleaner signal.

---

*Full analysis in: `results/exp_004_velocity_penalty/EXPERIMENT.md`*
