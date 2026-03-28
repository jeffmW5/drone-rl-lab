# Research: Stochastic-to-Deterministic Deployment Gap

**Date:** 2026-03-27
**Plateau context:** exp_056-060 all achieve 25-38 training reward but 0 benchmark gates. The
stochastic PPO training policy navigates toward gates, but the deterministic mean policy crashes
at 0.5-1.2s during deployment. Reward design, obs transforms, and curriculum are exhausted.
**Search queries used:**
- "PPO deterministic mean policy crashes stochastic exploration works"
- "drone racing reinforcement learning policy deployment stochastic vs deterministic"
- "PPO action distribution collapse mean policy unstable continuous control"
- "PPO entropy annealing log_std schedule deterministic convergence quadrotor"
- "max entropy misleads PPO quadrotor continuous control"

## Papers Reviewed

### Paper 1: Swift — Champion-level drone racing (Kaufmann et al., Nature 2023)
- **ArXiv:** PMC10468397 | **HF:** https://huggingface.co/papers/PMC10468397
- **Core technique:** PPO with 2×128 MLP, 31D obs (15D state + 12D gate corners in body frame + 4D prev action), 4D action (thrust + body rates). Progress reward + crash penalty.
- **Key insight for us:** They deploy the **deterministic mean** successfully — but train for **100M steps** with 100 parallel envs (~50 min on RTX 3090). We train for only 1.5M steps. They also use NO entropy scheduling — exploration comes purely from initial condition diversity.
- **Proposed experiment:** Train for much longer (10-20M+ steps). Our 3600s budget only gets ~1.5M steps. Budget needs to increase 10x, or we need more efficient training.

### Paper 2: Dynamic Entropy Tuning for Quadcopter Control (2512.18336)
- **ArXiv:** 2512.18336 | **HF:** https://huggingface.co/papers/2512.18336
- **Core technique:** SAC with dynamic entropy tuning. Target entropy = -dim(A). Entropy coefficient adjusted via gradient descent during training.
- **Key insight for us:** **Stochastic policies generalized to unseen positions where deterministic agents crashed.** Dynamic entropy tuning prevented catastrophic forgetting. External noise hurts — internal entropy is better.
- **Proposed experiment:** Deploy our trained model with stochastic action sampling (sample from distribution instead of using mean). Also try reduced-temperature sampling.

### Paper 3: When Maximum Entropy Misleads Policy Optimization (2506.05615)
- **ArXiv:** 2506.05615 | **HF:** https://huggingface.co/papers/2506.05615
- **Core technique:** Analysis of SAC vs PPO for quadrotor control. Shows MaxEnt corrupts Q-values for precision tasks where optimal actions are a "narrow corridor."
- **Key insight for us:** PPO is the right algorithm for precision control — confirms our choice. But entropy should be **minimized at critical states** and increased for exploration. Static entropy coefficients can't capture state-dependent needs. Our ent_coef=0.01 with max_logstd=-1.0 might be overly constraining the mean.
- **Proposed experiment:** Try removing max_logstd clamp entirely. The tight clamp forces the distribution to be narrow, but the mean may not have converged yet at 1.5M steps.

### Paper 4: Entropy Annealing for Policy Mirror Descent (2405.20250)
- **ArXiv:** 2405.20250 | **HF:** https://huggingface.co/papers/2405.20250
- **Core technique:** Theoretical framework for entropy decay schedules. For continuous actions: τ = 1/√(s+1).
- **Key insight for us:** Start with HIGH entropy to explore, then anneal to LOW entropy for convergence to deterministic-like policy. This is the opposite of what we do (we clamp logstd to be narrow from the start).
- **Proposed experiment:** Start with high entropy (ent_coef=0.05, no logstd clamp), anneal both over training.

### Paper 5: Action Collapse in Policy Gradient (2509.02737)
- **ArXiv:** 2509.02737 | **HF:** https://huggingface.co/papers/2509.02737
- **Core technique:** ETF structure for action layer to prevent mode collapse. Primarily for discrete actions.
- **Key insight for us:** Theoretical explanation of why mean becomes unstable — gradient concentration on single actions without exploration regularization. Less directly applicable (discrete focus).

## Synthesis

Three key findings change our approach:

1. **We are massively undertrained.** Swift trains for 100M steps; we train for 1.5M.
   Our reward is still climbing at budget end (28.7 and accelerating in exp_060).
   The mean policy may simply not have converged yet.

2. **Stochastic deployment works for quadcopter control.** The dynamic entropy paper
   shows stochastic policies generalize better than deterministic for quadrotors.
   We should test stochastic deployment immediately — it's a zero-cost experiment
   (just change the controller, no retraining needed).

3. **Our logstd clamp is counterproductive.** max_logstd=-1.0 forces narrow distributions
   from the start. With only 1.5M steps, the mean hasn't converged, so forcing
   narrow distributions just makes the unconverged mean crash. Either train much
   longer to let the mean converge, or remove the clamp and deploy stochastic.

## Proposed Experiments

### exp_061 -- Stochastic Deployment of exp_060 model
- **Hypothesis:** The stochastic training policy navigates (28.02 reward) but the
  deterministic mean crashes. Deploying with stochastic sampling (sample from
  the learned distribution) should recover the navigation behavior seen in training.
- **What to change:** Modify attitude_rl_race.py to support body_frame_obs in
  stochastic mode (sample action = mean + std * noise instead of just mean).
  No retraining needed — use exp_060's model.ckpt.
- **Expected outcome:** Flight time > 2s, possibly gate passages.
- **Paper basis:** Dynamic Entropy Tuning (2512.18336) — stochastic deployment
  generalizes where deterministic crashes.

### exp_062 -- Temperature-Scaled Deployment
- **Hypothesis:** Full stochastic (std=1.0 of learned distribution) may be too noisy
  for deployment. Scaling temperature (std *= 0.5 or 0.3) reduces noise while
  keeping the policy away from the unstable mean.
- **What to change:** Add temperature parameter to controller. Test T=0.1, 0.3, 0.5, 1.0.
- **Expected outcome:** Sweet spot between deterministic crash and full stochastic noise.
- **Paper basis:** Standard RL practice + entropy annealing theory (2405.20250).

### exp_063 -- Extended Training (10M+ steps, no logstd clamp)
- **Hypothesis:** Swift achieves deterministic deployment after 100M steps. Our
  1.5M is 67x less. Remove max_logstd clamp and train for 10M+ steps (longer budget
  or checkpoint saving). The mean should naturally converge with enough training.
- **What to change:** Remove max_logstd=-1.0, budget_seconds=7200 or more,
  keep body_frame_obs + bilateral_progress + soft_collision from exp_060.
- **Expected outcome:** Mean policy stabilizes after 5-10M steps. Deterministic
  deployment should improve.
- **Paper basis:** Swift (Nature 2023) — 100M steps, deterministic deployment works.

### exp_064 -- Entropy Annealing Schedule
- **Hypothesis:** Start with high entropy (ent_coef=0.05, no logstd clamp) for
  exploration, then anneal both to low values over training. This lets the mean
  converge naturally to the optimal mode.
- **What to change:** Implement ent_coef annealing (0.05 → 0.001 over training).
  Remove max_logstd clamp. Train for 10M+ steps.
- **Expected outcome:** Smoother convergence, mean finds the navigation mode.
- **Paper basis:** Entropy Annealing (2405.20250) — τ = 1/√(s+1) for continuous.
