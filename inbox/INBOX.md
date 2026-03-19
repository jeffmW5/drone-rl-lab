# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_022 -- Benchmark on Level 2 sim
**Config:** Use attitude_rl_race.py controller with exp_022 checkpoint
**Depends on:** exp_022 training (DONE)

**Result:** 0/10 finishes, 0.5 gates avg, all crash at 2.02s. Model passes gate 0 but flies through
ceiling (z=3.06). Root cause: no altitude penalty in training + 100-step grace period masks OOB.
See STATUS.md for full analysis.

---

### [DONE] exp_023 -- Extended RaceCoreEnv Training (5M+ steps)
**Config:** TBD (based on exp_022 benchmark results)
**Depends on:** exp_022 benchmark (DONE)

**Result:** Ran exp_023 (soft OOB, oob_coef=2.0) and exp_023b (hard OOB, oob_coef=5.0, z_high=1.8).
Both 0/5 finishes, 0 gates at benchmark. Model applies max thrust throughout, never learns altitude control.
Root cause: cumulative proximity reward (~10-15) >> gate_bonus (5.0). Model optimizes "fly fast, die at ceiling."
Also fixed grace period bug in race_core.py (was 2.0s, now 0.2s).
exp_024 now training with rebalanced rewards (gate_bonus=20, proximity_coef=1, z_high=1.5, oob_coef=10).

---

### [NEXT] exp_024 -- Benchmark + exp_025 fallback (altitude reward)
**Config:** `configs/exp_024_racecore_gatereward.yaml` (training) / `configs/exp_025_racecore_altreward.yaml` (fallback)
**Depends on:** exp_024 training completion

**Goal:** Benchmark exp_024 on Level 2. This experiment rebalanced rewards so gate passage dominates (gate_bonus=20, proximity_coef=1) with tighter altitude enforcement (z_high=1.5, oob_coef=10). The hypothesis is that the exp_023b model optimized proximity grinding rather than gate passage because gate_bonus (5) << cumulative proximity reward (~10-15). If exp_024 still fails altitude control, fall through to exp_025 which adds an explicit altitude-matching reward (alt_coef=1.5, survive_coef=0.3).

**Benchmark procedure:**
1. Wait for exp_024 training to complete if still running (5400s budget from ~11:23 AM 2026-03-19)
2. Run Level 2 benchmark (10 runs): `python benchmark.py --controller attitude_rl_race --exp exp_024 --level level2 --runs 10`
3. Write results to `outbox/exp_024.md` and update `STATUS.md`

**Success criteria (exp_024):**
- 2+ gates passed consistently (agent has learned altitude control)
- Crash altitude < 1.6m on average (not flying to ceiling)
- Any finish on Level 2 counts as a breakthrough

**If exp_024 fails (0 gates / ceiling crash — altitude still not learned):**
- Train exp_025: `configs/exp_025_racecore_altreward.yaml` (config already exists)
- exp_025 adds explicit altitude-matching reward: `alt_reward = alt_coef * exp(-3 * |z - gate_z|)`
- Also adds survive_coef=0.3 to reward staying alive (longer episodes)
- Same budget: 8M steps, 512 envs, 5400s on RTX 3090
- After training, benchmark exp_025 and write `outbox/exp_025.md`

**If both exp_024 and exp_025 fail:**
- Try curriculum learning: train first on Level 0 (fixed gates), then fine-tune on Level 2
- Or try larger network (2 hidden layers → 3, or 256 units → 512)
