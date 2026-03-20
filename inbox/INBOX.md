# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_028 -- High Speed Reward (fine-tune exp_026, no random gates)
**Result:** Reward 16.95 (highest ever), benchmark 0/5 finishes, **0.2 avg gates** (FIRST GATE PASSAGE in run 4!), avg 0.94s.
speed_coef=1.0 taught navigation but destroyed hover stability. Sweet spot is 0.3-0.5.

---

### [NEXT] exp_029 -- Balanced Speed Reward (hover + navigation)
**Config:** `configs/exp_029_racecore_balanced_speed.yaml`
**Depends on:** exp_026 checkpoint (pretrained hover model)

**Goal:** Find the sweet spot between exp_026's pure hover (28.8s, 0 gates) and exp_028's pure navigation (0.94s, 0.2 gates). exp_028 proved that increasing speed_coef teaches gate navigation — the first-ever RaceCoreEnv gate passage. But speed_coef=1.0 overwhelmed hover rewards. This experiment uses speed_coef=0.4 (moderate lateral incentive), proximity_coef=1.0 (stronger directional signal than exp_028's 0.5), survive_coef=0.5 (reinforces staying alive), and LR=0.0003 (more conservative to preserve pretrained hover).

**What's new:**
- `speed_coef: 0.4` (between exp_026's 0.1 and exp_028's 1.0)
- `proximity_coef: 1.0` (between exp_026's 2.0 and exp_028's 0.5 — moderate pull)
- `survive_coef: 0.5` (up from 0.3 — stronger hover preservation)
- `learning_rate: 0.0003` (lower than exp_028's 0.0005 — more conservative fine-tuning)

**Training:** 512 envs, 8M steps on GPU (RTX 3090), 5400s budget. Pretrained from exp_026 checkpoint.

**Success criteria:**
- Flight time >10s (proves hover is preserved — between 28.8s and 0.94s)
- At least 1 gate passed (matches or exceeds exp_028's 0.2 avg)
- Ideally: >5s flight time AND >0.5 avg gates (navigates while surviving)

**If it doesn't work:**
- If hover destroyed again (flight <2s): try speed_coef=0.2 with even lower LR (0.0001)
- If hover preserved but no gates (flight >20s, 0 gates): try speed_coef=0.6 — closer to exp_028
- If both metrics are mediocre: try two-phase curriculum — 4M steps at speed_coef=0.2, then 4M at 0.5
