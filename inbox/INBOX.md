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

---

### [NEXT] exp_026 -- Vertical Velocity Penalty + Tighter Ceiling
**Config:** `configs/exp_026_racecore_vzpenalty.yaml`
**Depends on:** None

**Goal:** exp_025b proved the model can learn altitude awareness (thrust modulation at z~1.1), but momentum carries it past the training ceiling (z=1.5) to z=2.3+ in benchmark. Two targeted fixes: (1) lower z_high from 1.5 to 1.3 (just above the highest gate at z=1.2), forcing the model to start braking closer to gate altitude; (2) add a vertical velocity penalty that penalizes upward velocity (vz > 0) when z > 0.5m, directly teaching the model to ascend slowly and precisely rather than shooting upward. Also increases gamma 0.94→0.97 so the model values long-term survival more.

**What's new:**
- `z_high: 1.3` — tighter ceiling (was 1.5), forces braking before z=1.3 vs z=1.5
- `vz_coef: 0.5, vz_threshold: 0.5` — new vertical velocity penalty (requires code change in training env)
- `gamma: 0.97` — longer reward horizon (was 0.94), model values surviving future steps more
- `oob_coef: 8.0` — slightly stronger OOB penalty (was 5.0) to reinforce the tighter ceiling

**Code change required:** The `vz_coef` and `vz_threshold` params are new. Linux Claude must implement them in the training reward function (wherever the altitude reward is computed — likely `train_racing.py` or `race_core.py`). Penalty form: `reward -= vz_coef * max(obs_vz, 0)` when `obs_z > vz_threshold`. If vz is not directly in the obs, use the env's drone state.

**Training:** 512 envs, 8M steps on GPU (RTX 3090), 5400s budget

**Success criteria:**
- Model brakes before z=1.3 in training (terminates from timeout, not OOB)
- Benchmark crash time improves beyond 1.16s (exp_025b)
- At least 1 gate passed in benchmark

**If it doesn't work:**
- If model learns to hover very low and never ascends: reduce vz_threshold to 0.7 or reduce vz_coef to 0.2
- If model still overshoots: try vz_coef=1.0 and z_high=1.2 (exact gate height)
- If code change is too complex: skip vz_coef and just run with z_high=1.3 + gamma=0.97 only (label as exp_026_simple)
