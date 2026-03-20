# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [QUEUED] exp_027 -- Random Gate Starts (Swift-style initialization)
**Config:** `configs/exp_027_racecore_randomgate.yaml`
**Depends on:** exp_026 (use best reward shaping from exp_026 results)

**Goal:** Implement the initialization strategy from Kaufmann et al. "Champion-Level Drone Racing" (Nature 2023). Currently the drone always resets at the start position, so the policy only learns the first few gates well and never practices mid-track segments. Swift solved this by resetting agents at a **random gate** on the track with bounded perturbation around a state previously observed when passing that gate. This forces the policy to learn all track segments equally, dramatically improving finish rate and late-gate navigation.

**Context from exp_026:** Altitude control is SOLVED — model hovers stably at z=0.72 (gate altitude) for 28.8s. The remaining gap is purely horizontal navigation: model never moves laterally toward gates. All reward shaping from exp_026 is inherited.

**What's new (code changes required):**
- **Random gate spawn:** On `reset()`, pick a random gate index (0 to N-1). Place the drone ~0.5-1.0m before that gate, facing toward it (use gate yaw for orientation). Set `self.next_gate` accordingly so reward/progress tracking starts from the correct gate.
- **Bounded perturbation:** Add small noise to spawn position (±0.15m xyz) and velocity (±0.3 m/s) so the agent sees varied approach conditions. Attitude perturbation: ±5° roll/pitch, ±15° yaw.
- **Config flags:** `random_gate_start: true`, `spawn_offset: 0.75` (meters before gate), `spawn_pos_noise: 0.15`, `spawn_vel_noise: 0.3`
- **Keep start-gate spawning as fallback:** If `random_gate_start: false`, use current behavior (always gate 0).
- **Inherit all reward shaping from exp_026** (vz penalty, alt_coef, OOB, etc.) — adjust based on exp_026 results.

**Implementation hints:**
- Gate positions and orientations should be available from the race config (level2_attitude.toml or the env's gate list).
- Spawn position = `gate_pos - spawn_offset * gate_forward_vector`, at appropriate altitude.
- If the env tracks `self.next_gate` or similar, update it to match the spawn gate so proximity reward points to the correct target.
- The episode should still terminate on OOB, collision, or max_episode_steps — just the starting point changes.

**Training:** 512 envs, 8M steps on GPU (RTX 3090), 5400s budget

**Success criteria:**
- Model passes gates in the middle/end of the track (not just gate 0-1)
- Finish rate improves over exp_026 baseline
- Mean gates passed per episode increases across ALL gates, not just early ones
- Benchmark: at least 3 gates passed, ideally first complete lap

**If it doesn't work:**
- If reward drops significantly: try 50/50 mix (half envs start at gate 0, half random) via `random_gate_ratio: 0.5`
- If spawning near certain gates causes instant OOB: add gate-specific z-clamping at spawn
- If the agent learns to pass individual gates but can't chain them: increase max_episode_steps and ensure gate-to-gate transitions are well-represented
