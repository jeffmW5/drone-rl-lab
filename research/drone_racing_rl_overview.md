# Research: Drone Racing RL — General Overview

**Date:** 2026-03-23
**Plateau context:** After 39 experiments, reward tuning is exhausted. Deterministic mean policy always hovers or crashes — no stable navigation regime found. Need structural changes (architecture, curriculum, or entirely different approach).
**Search queries used:** "drone racing reinforcement learning", "autonomous drone racing PPO curriculum learning", "quadrotor reward shaping sim-to-real attitude control"

## Papers Reviewed

### Paper 1: RaceVLA — VLA-based Racing Drone Navigation with Human-like Behaviour
- **ArXiv:** 2503.02572 | **HF:** https://huggingface.co/papers/2503.02572
- **Core technique:** Vision-Language-Action model (fine-tuned OpenVLA-7b) that maps FPV camera frames + language commands → 4D action (Vx, Vy, Vz, yaw rate). Imitation learning from human pilot demonstrations, not RL.
- **Architecture:** 7B parameter VLA, LoRA rank-32 fine-tuning, int8 quantization for 4Hz inference on RTX 4090.
- **Training:** 200 episodes, ~20k images, batch=16, lr=5e-4, 7000 gradient steps on A100.
- **Results:** Avg 1.04 m/s, max 2.02 m/s. Outperforms RT-2 on all generalization axes. But speeds are far below competition level (~3-5s laps require >5 m/s).
- **Key insight for us:** VLA approach is fundamentally different from our PPO pipeline — requires onboard camera and language input. NOT directly applicable to our state-based attitude control setup. However, demonstrates that imitation from human demonstrations can solve the navigation problem our RL can't.
- **Proposed experiment:** N/A — architecture mismatch (vision-based vs state-based).

### Paper 2: Dream to Fly — Model-Based RL for Vision-Based Drone Flight
- **ArXiv:** 2501.14377 | **HF:** (not on HF papers)
- **Core technique:** DreamerV3 (model-based RL) learns to fly through race gates from raw camera pixels. Key finding: **PPO fails to learn from pixels, DreamerV3 succeeds.**
- **Architecture:** DreamerV3 world model — learns a latent dynamics model, then trains policy in "dream" rollouts within the learned model. No intermediate state estimation needed.
- **Reward:** Perception-aware reward terms become unnecessary when learning from pixels — the world model implicitly captures relevant structure.
- **Key insight for us:** PPO's failure from pixels aligns with our experience — PPO struggles with the credit assignment in long-horizon drone racing. Model-based RL (DreamerV3) might handle the exploration problem better because it can plan through its learned model. However, our setup uses state observations (not pixels), so the pixel advantage doesn't directly apply.
- **Proposed experiment:** Try DreamerV3 or similar model-based RL on our RaceCoreEnv as an alternative to PPO. Would require significant code changes.

### Paper 3: Autonomous Drone Racing with Deep RL (Song et al., IROS 2021)
- **ArXiv:** 2103.08624
- **Core technique:** Deep RL with **relative gate observations** generates near-time-optimal trajectories. Uses gate-relative state representation so the policy generalizes across track layouts.
- **Results:** ~60 km/h in real world. Near time-optimal trajectories adaptable to environment changes.
- **Key insight for us:** **Relative gate observations** are critical. Our obs space has 73 dims (drone state + trajectory lookahead) but NO gate info. This paper shows the policy needs to see gates directly. This aligns with our hard rule #10 ("training env has ZERO gate awareness").
- **Proposed experiment:** Add gate-relative observations to RaceCoreEnv obs space (distance to gate, bearing to gate, gate normal vector). This is a code change to the env, not just config.

### Paper 4: Swift — Champion-level Drone Racing (Kaufmann et al., Nature 2023)
- **ArXiv:** N/A (Nature paper) | **Ref:** nature.com/articles/s41586-023-06419-4
- **Core technique:** PPO in simulation + sim-to-real transfer. Beat human world champions in physical races. Uses **progress-based reward** (not proximity), privileged learning, and domain randomization.
- **Architecture:** MLP policy, trained with PPO. Observation includes drone state + gate-relative information.
- **Reward:** Progress along optimal trajectory + gate passage bonus. NOT proximity reward — aligns with our PBRS finding (exp_032).
- **Training:** Massive parallel simulation, then zero-shot transfer to real drone.
- **Key insight for us:** Swift proves PPO CAN solve drone racing at champion level. The key ingredients we're missing: (1) gate-relative observations in obs space, (2) progress reward (we added PBRS but still hover), (3) privileged learning — train a teacher with full state, then distill to student. Our hover-or-crash problem may stem from the obs space gap, not PPO itself.
- **Proposed experiment:** Add gate info to obs space (gate position, gate normal, distance). This is the most actionable finding.

### Paper 5: Learning to Fly — Gym Environment with PyBullet (Panerati et al.)
- **ArXiv:** 2103.02142 | **HF:** https://huggingface.co/papers/2103.02142
- **Core technique:** Open-source gym environment for quadcopter RL (gym-pybullet-drones). Multi-agent support, vision-based interface.
- **Key insight for us:** This is the environment our hover backend (exp_001-005) is built on. We already moved past it to RaceCoreEnv for racing. No new actionable insight.

## Synthesis

The papers collectively point to **two critical gaps** in our current setup:

1. **Observation space lacks gate information.** Both Swift (Nature) and Song et al. (IROS 2021) use gate-relative observations. Our agent can't navigate to gates it can't see. This is the #1 bottleneck — even perfect reward shaping can't compensate for missing information.

2. **PPO works for drone racing** when given the right observations and reward. Swift proved this at champion level. Our PPO isn't the problem — it's what we feed it. DreamerV3 (Dream to Fly) is an alternative but requires vision pipeline we don't have.

3. **Imitation learning** (RaceVLA) is an entirely different paradigm that works but requires camera input and human demonstrations. Not directly applicable.

**Priority action:** Add gate-relative observations to RaceCoreEnv, THEN retry PPO training.

## Proposed Experiments

### exp_040_gate_obs
- **Hypothesis:** Adding gate-relative observations (distance, bearing, gate normal) to the obs space will give the policy the information it needs to navigate toward gates, breaking the hover-or-crash binary.
- **What to change:** Modify RaceCoreEnv observation preprocessing to include: (1) vector from drone to target gate (3D), (2) distance to target gate (scalar), (3) gate forward normal vector (3D), (4) dot product of velocity with gate direction (scalar). ~8 new dims added to obs. Code change to `train_race.py` `_preprocess_obs()`.
- **Expected outcome:** Policy should learn directional navigation because it can "see" where gates are. Combined with PBRS progress reward, gate approach becomes strongly incentivized with clear gradient signal.
- **Paper basis:** Swift (Kaufmann et al. Nature 2023) + Song et al. IROS 2021 — both use gate-relative obs as core design choice.

### exp_041_gate_obs_curriculum
- **Hypothesis:** Gate observations + short episodes (300 steps from exp_039) creates optimal learning conditions — agent sees gates AND has urgency to reach them.
- **What to change:** Same gate obs from exp_040 + max_episode_steps=300 from exp_039.
- **Expected outcome:** Faster convergence to gate passage, breaking bistable policy from exp_039 into consistent navigation.
- **Paper basis:** Curriculum approach from exp_039 + gate obs from Swift.
