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

### [DONE] exp_024 -- Benchmark + exp_025 fallback (altitude reward)
**Config:** `configs/exp_024_racecore_gatereward.yaml` / `configs/exp_025b_racecore_altreward_grace25.yaml`
**Depends on:** exp_024 training completion

**Result:** Ran exp_024, exp_025, and exp_025b. All 0 gates at benchmark.
- **exp_024:** proximity_coef=1 too weak → reward plateaued at ~0.3 for 5M steps. Killed early.
- **exp_025:** altitude reward works but 10-step grace period kills drone on ground contact (0.22s crash).
- **exp_025b:** Fixed 25-step grace. Reward 10.79±2.64 (peak 15.84). Model learns thrust modulation
  (reduces from 0.98→0.32 near z=1.1) but momentum carries it past z=1.5 to z=2.3+. Benchmark: 0/5, avg 1.16s.
- **Key progress:** First model to modulate thrust based on altitude. Gap is now momentum/braking, not awareness.
- **Grace period fixed:** `self.sim.freq→self.freq` + divisor `5→2` = 25 steps (0.5s). Both training and benchmark.
- **Next:** Need vertical velocity penalty or lower z_high to force earlier braking. See STATUS.md.
