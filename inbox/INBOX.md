# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_027 -- Random Gate Starts (Swift-style initialization)
**Result:** Reward 11.67, but benchmark FAILED: 0/5 finishes, 0 gates, avg 1.1s flight time. 100% random gate spawns meant model never learned ground takeoff. Regressed to exp_025b levels.

---

### [NEXT] exp_027b -- 50/50 Random Gate Mix
**Config:** `configs/exp_027b_racecore_randomgate_mix.yaml`
**Depends on:** exp_027 (uses same code, just changes random_gate_ratio)

**Goal:** Fix exp_027's failure by using a 50/50 mix of ground starts and random gate starts. exp_027 proved that random gate spawns produce high training reward (11.67) but the model never learned takeoff because it always spawned mid-air. By splitting 50% of envs to start on ground (preserving exp_026's takeoff+hover skills) and 50% at random gates (learning gate approach/passage), the model should learn both capabilities simultaneously.

**What's new:**
- `random_gate_ratio: 0.5` — half envs start at gate 0 on ground, half at random gates mid-air
- All other params identical to exp_027 (inherits exp_026 reward shaping)

**Training:** 512 envs, 8M steps on GPU (RTX 3090), 5400s budget

**Success criteria:**
- Flight time at least matches exp_026 baseline (28.8s) — model still knows takeoff
- At least 1 gate passed in benchmark (first gate passage for RaceCoreEnv pipeline)
- Training reward ≥ 10.0 (between exp_026's 9.77 and exp_027's 11.67)

**If it doesn't work:**
- If takeoff still fails: increase ground ratio to 75/25 via `random_gate_ratio: 0.25`
- If training reward drops significantly: the two objectives may conflict — try curriculum (ground-only first 2M steps, then introduce random gates)
- If gates passed but only near-gate spawns: verify proximity reward correctly chains to next gate after passage
