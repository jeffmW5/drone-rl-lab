# INBOX — Experiment 017: Reward Shaping Investigation

**From:** Windows Claude (Orchestrator)
**Date:** 2026-03-16
**Priority:** HIGH

Refer to `MEMORY.md` for context on past experiments.

---

## Context

exp_016 is our best model: 10M GPU steps on Level 2, reward 7.71, **first RL to finish L2** (2/10 runs, 13.49s avg). But:
- 20% finish rate is not competitive
- 13.49s is 3-4x slower than Kaggle top 3 (~3.4-5.0s)
- Reward plateaued at ~7.7 — more steps won't help (see Hard Rule #7)

The current reward function only measures **trajectory following**. There's no explicit incentive for actually passing through gates. The agent may be getting high reward for smooth flight while missing gates entirely.

---

## Task: Investigate reward function in lsy_drone_racing

### Step 1: Understand the current reward
- Read the reward function source code in lsy_drone_racing
- Specifically find: `reward_coefs`, `rpy_coef`, and how rewards are computed in the JAX env
- Document: what does the agent get rewarded for today? Is there a gate-passage bonus?

### Step 2: Check if gate-passage reward exists
- Look for any gate-related terms in the reward function
- Look for `gate`, `passed`, `progress` in the env source
- If there IS already a gate reward, figure out why it's not working

### Step 3: Investigate what observations the agent sees
- Print the full observation space shape and contents
- Does the agent see gate positions? Gate normals? Its own distance to next gate?
- If gate info is in obs, the agent CAN learn to aim for gates — the reward just needs to encourage it

### Step 4: Write findings to outbox
Write `outbox/reward_investigation.md` with:
- Current reward function (simplified pseudocode)
- What observations include
- Whether gate-passage bonus exists
- Recommended changes for exp_017

### Step 5: If reward modification is feasible on CPU
- Create `configs/exp_017_gate_reward.yaml` with modified reward coefficients
- This should be testable on CPU (64 envs, 500k steps) as a quick signal check
- The question is: does adding gate-proximity reward change behavior at all?

---

## Important notes

- **Do NOT modify train_racing.py** unless absolutely necessary for reward access
- The reward function may be inside lsy_drone_racing's JAX env, not our code
- If modifying reward requires changing lsy_drone_racing source, document exactly what changed
- Read MEMORY.md Hard Rules before starting — especially #6, #7, and #9
- This is a RESEARCH task — understanding the reward is more valuable than running a bad experiment

---

## Expected output

1. `outbox/reward_investigation.md` — detailed analysis of reward function + observations
2. Optionally: `configs/exp_017_gate_reward.yaml` if modification is feasible
3. Updated `MEMORY.md` with findings
