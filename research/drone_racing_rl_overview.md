# Research: Drone Racing RL — Deep Literature Review

**Date:** 2026-03-23
**Plateau context:** After 39 experiments, reward tuning is exhausted. Deterministic mean policy always hovers or crashes — no stable navigation regime found. Need structural changes (architecture, curriculum, or entirely different approach).
**Search queries used:** "drone racing reinforcement learning", "autonomous drone racing PPO curriculum learning", "quadrotor reward shaping sim-to-real attitude control", "asymmetric actor critic privileged information quadrotor gate racing", "PPO drone racing deterministic policy mode collapse"

---

## Papers Reviewed

### Paper 1: Swift — Champion-level Drone Racing (Kaufmann et al., Nature 2023)
- **Ref:** nature.com/articles/s41586-023-06419-4 | **PMC:** PMC10468397
- **Core technique:** PPO in simulation → sim-to-real. Beat human world champions.

**Exact reward function:**
```
r_t = r_prog + r_perc + r_cmd - r_crash
r_prog = λ₁(d_{t-1}^Gate - d_t^Gate)         # progress toward gate center
r_perc = λ₂ exp(λ₃ · δ_cam / 4)              # keep next gate in camera view
r_cmd  = λ₄||a_t^ω|| + λ₅||a_t - a_t-1||²   # smooth commands
r_crash = 5.0 if collision/OOB else 0
```

**Observation space (31D):**
- Platform state R¹⁵ (position, velocity, rotation matrix)
- Next gate relative pose R¹² (4 gate corners in body frame, 3D each)
- Previous action R⁴

**Action space (4D):** Mass-normalized collective thrust + body rates (roll, pitch, yaw)

**Network:** 2-layer MLP, 128 nodes/layer, LeakyReLU(α=0.2). 8ms inference on Jetson TX2.

**PPO hyperparams:** Adam lr=3e-4, episode_length=1500, 100 parallel envs, 1×10⁸ interactions (50 min on RTX 3090).

**Key insight for us:** Swift's obs has gate corners in body frame (12D). Our obs has ZERO gate info. Swift uses progress reward (distance delta) — matches our PBRS. Swift does NOT use domain randomization during training — uses residual models post-hoc instead. Network is tiny (2×128 MLP).

---

### Paper 2: Agile Flight from Multi-Agent Competitive Racing (Pasumarti et al., 2024)
- **ArXiv:** 2512.11781
- **Core technique:** Sparse reward + competitive self-play. Explicitly REJECTS dense progress rewards.

**Exact reward function (sparse):**
```
r_t = r_pass + r_lap - r_cmd - r_crash
r_pass  = 10.0 (leading) or 5.0 (trailing) when passing gate
r_lap   = 50.0 for first to complete lap
r_cmd   = 0.15(ω²_roll + ω²_pitch) + 0.05·ω²_yaw
r_crash = 2.0 (terminal collision) or 0.1 (contact)
```

**Why they reject progress reward:** Dense progress reward discourages deviations needed for obstacle avoidance and tactical racing. Sparse + competition naturally induces agile behaviors.

**Observation space (42D):**
- Linear velocity in body frame (3D)
- Attitude rotation matrix (9D)
- Current gate corners in body frame (12D)
- Next gate corners in body frame (12D)
- Opponent position in body frame (3D)
- Opponent velocity in body frame (3D)

**Action space (4D):** Thrust + body rates, mapped via tanh to [-1,1].

**Network:** Actor MLP [512, 512, 256, 128] ELU. Critic [512, 512, 256, 256, 128, 128] ELU.

**Training:** IPPO (independent PPO), Isaac Sim v4.5, Crazyflie 2.1, policy at 50Hz / PID at 500Hz.

**Results:** 91.17% win rate vs baselines, outperforms dense single-agent on complex tracks. Zero-shot sim-to-real transfer.

**Key insight for us:** (1) Gate corners in body frame appear in EVERY competitive paper — 12D per gate is the standard. (2) Sparse rewards + competition outperform dense rewards. (3) Much bigger networks than Swift (512-node layers vs 128). (4) Progress reward can actually HURT in complex environments.

---

### Paper 3: Dream to Fly — Model-Based RL for Drone Racing (Romero et al., 2025)
- **ArXiv:** 2501.14377

**Exact reward function:**
```
r(k) = -4.0                                     if collision
      = +10.0                                    if gate passed
      = b₁(||g_k - p_{k-1}|| - ||g_k - p_k||)  otherwise (progress)
        - b₂||ω_k||                             (body rate penalty)
b₁=1.0, b₂=0.01
```

**Observation:** 64×64 RGB pixels. **Action (4D):** thrust + body rates.

**DreamerV3 arch:** Large config, 768-unit MLPs, 2048 RSSM units. 20M steps to converge (~240hr on RTX 8000).

**PPO comparison:** PPO achieves consistently low rewards across all tracks (Circle, Kidney, Figure-8) even after 10M steps. DreamerV3 converges to 8 m/s gate passage.

**Key insight for us:** PPO fails from pixels, not from state. Our setup uses state obs (not pixels), so PPO should work IF given the right obs. Dream to Fly confirms progress reward (distance delta) is the standard — same as Swift, same as our PBRS.

---

### Paper 4: Vision-Based Agile Gate Navigation (RSS 2024, UZH)
- **Ref:** roboticsproceedings.org/rss20/p082.pdf
- **Core technique:** Asymmetric actor-critic with privileged information. Maps pixels → control at 40 km/h.

**Key technique — Asymmetric Actor-Critic:**
- **Critic** receives full privileged state (drone pose + gate positions + velocities)
- **Actor** receives only sensor observations (gate edge detections or images)
- During training, critic's privileged info improves value estimates → better policy gradient
- At deployment, only actor runs (no privileged info needed)

**Gate sensor abstraction:** Uses inner edges of gates as compact representation instead of full images. Reduces compute while preserving task-relevant information.

**Key insight for us:** We could use asymmetric actor-critic — give the critic full state (position + gate positions) while the actor gets our current obs space + gate-relative features. This improves learning without changing deployment requirements.

---

### Paper 5: Curriculum-Based RL for Quadrotor Stabilization (2025)
- **ArXiv:** 2501.18490

**Three-stage curriculum:**
1. Hover from fixed position (0,0,0) → target (0,0,1)
2. Reach target from random positions (2m radius, ±15° attitude)
3. Add random velocities [-1,1] m/s and angular velocities [-1,1] rad/s

**Transfer mechanism:** Policy weights continue from previous stage. Each stage trains ~6-7M steps.

**Reward:** R(t) = 25 - 20·T_e - 100·E + 20·S - 18·w_e (target error, exploration bounds, stability, angular velocity)

**PPO:** MLP 12→64→64→4, lr=3e-4, γ=0.99, batch=128, epochs=10, 5s episodes.

**Results:** Curriculum converges in 20M total steps (163 min). Single-stage fails to converge in same budget.

**Key insight for us:** Our exp_039 (short episodes) was a crude curriculum. This paper shows proper curriculum: start simple (hover), then add complexity (random starts, velocities). We could structure: Stage 1=hover at gate altitude (exp_026 solved this), Stage 2=navigate to nearby gate from hover, Stage 3=full gate racing with randomization.

---

### Paper 6: Mastering Diverse Cluttered Tracks (IEEE RA-L, 2025)
- **ArXiv:** 2512.09571
- **Core technique:** Adaptive noise-augmented curriculum + asymmetric actor-critic.

**Two-phase training:**
1. **Soft-collision phase:** Preserves policy exploration for high-speed flight (collisions don't terminate)
2. **Hard-collision phase:** Enforces robust obstacle avoidance (collisions terminate)

**Key insight for us:** Their soft-collision phase maps directly to our problem — if we make crashes non-terminal initially, the policy can explore navigation without the binary hover-or-crash trap. Then tighten to hard crashes after navigation is learned.

---

### Paper 7: Teacher-Student for Drone Racing (CoRL 2024, UZH)
- **Core technique:** Three-phase training:
  1. Train teacher with RL (PPO) using privileged state (full gate positions)
  2. Distill teacher → student via imitation learning (student uses only sensor obs)
  3. Fine-tune student with RL

**Key insight for us:** The teacher-student pipeline is how the best teams bridge the gap between what info is available at training vs deployment. We could train with full state access first, then distill.

---

## Synthesis: What the Literature Collectively Says

### 1. Gate observations are non-negotiable
EVERY successful drone racing RL system includes gate information in the observation space:
- Swift: 12D (4 gate corners × 3D, body frame)
- Competitive Racing: 24D (current + next gate corners, body frame)
- Song et al.: gate-relative state representation
- RSS 2024: gate edge detections

Our agent has 73 obs dims but ZERO gate info. **This is the #1 fix.**

### 2. The standard reward is progress (distance delta) + gate bonus
```
r_progress = d_{t-1} - d_t  (toward gate center)
r_gate     = bonus on passage (5-20 points)
r_smooth   = -penalty on body rates and action deltas
r_crash    = -penalty (2-5 points)
```
Our PBRS (exp_032) already implements this correctly. The reward isn't the problem.

### 3. Network architectures vary but are all MLPs
- Swift: 2×128 (tiny, fast)
- Competitive: [512, 512, 256, 128] (much larger)
- Curriculum: [64, 64] (minimal)

Our architecture should work. Consider trying larger networks if gate-obs doesn't solve it.

### 4. Asymmetric actor-critic is the state of the art
Give the critic privileged info (full gate positions) while the actor gets sensor-realistic obs. This dramatically improves learning without changing deployment. **Low-hanging fruit for us.**

### 5. Curriculum design matters more than reward tuning
- Soft collisions → hard collisions (2-phase)
- Simple hover → random starts → full racing (3-stage)
- Short episodes → longer episodes
Our exp_039 (short episodes) was on the right track but needs the full 2-3 stage structure.

### 6. Sparse rewards can outperform dense in complex environments
The competitive racing paper explicitly rejects progress rewards. Dense rewards can create local optima (like our hover trap). With proper exploration (competition, curriculum), sparse gate-passage rewards alone are sufficient.

---

## Proposed Experiments (Priority Order)

### exp_040_gate_obs (HIGHEST PRIORITY)
- **Hypothesis:** Adding gate-relative observations is the single most impactful change based on ALL reviewed papers.
- **What to change:** Add to obs: vector to target gate in body frame (3D), distance (1D), gate normal in body frame (3D), next gate same (7D). Total: ~14 new dims. Code change to `_preprocess_obs()`.
- **Expected outcome:** Policy can "see" gates → navigate toward them.
- **Paper basis:** Swift, Competitive Racing, Song et al., RSS 2024 — ALL use gate obs.

### exp_041_soft_collision_curriculum
- **Hypothesis:** Two-phase training — soft collisions first (crashes don't terminate, just penalty), then hard collisions. Eliminates the binary hover-or-crash trap because the policy can explore through crashes.
- **What to change:** Phase 1: set crash penalty but don't terminate episode on collision. Phase 2: switch to normal termination after navigation is learned.
- **Paper basis:** Mastering Diverse Tracks (RA-L 2025) — soft→hard collision curriculum.

### exp_042_asymmetric_critic
- **Hypothesis:** Give the critic full privileged state (drone position + all gate positions) while the actor keeps current obs + gate-relative features. Better value estimates → better policy gradient → escape hover trap.
- **What to change:** Separate critic obs from actor obs in PPO. Critic gets privileged 42D state (like Competitive Racing paper). Actor gets standard obs + gate features.
- **Paper basis:** RSS 2024, CoRL 2024, RA-L 2025 — all use asymmetric actor-critic.

### exp_043_larger_network
- **Hypothesis:** Our network may be too small. Competitive Racing uses [512,512,256,128] vs our likely [64,64] or similar.
- **What to change:** Increase hidden layers to [256,256,128] or [512,512,256,128].
- **Paper basis:** Competitive Racing paper uses much larger networks than Swift.

### exp_044_sparse_reward
- **Hypothesis:** Remove all dense rewards. Only: +20 gate passage, -5 crash, -0.15(ω²). Let the curriculum and gate obs do the work instead of dense reward shaping.
- **What to change:** Set all *_coef to 0 except gate_bonus and crash penalty. Requires gate obs (exp_040) first.
- **Paper basis:** Competitive Racing paper shows sparse outperforms dense in complex environments.
