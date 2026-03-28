# Research: Closing the Deterministic Deployment Gap

**Date:** 2026-03-28
**Plateau context:** exp_069 (2×128 MLP) produced the first deterministic gates (2/15) and 5/15 at T=0.3, but the deterministic mean policy still crashes 87% of the time despite training reward of 52.39 peak. The stochastic policy navigates during training but the mean fails at deployment. Network capacity helped (2×64→2×128 doubled gate passage) but didn't close the gap.
**Search queries used:**
- "PPO deterministic policy deployment gap continuous control mean policy convergence"
- "policy distillation reinforcement learning drone racing quadrotor"
- "network architecture PPO continuous control hidden size MLP drone"
- "deterministic actor critic deployment stochastic training gap"
- "batch normalization layer normalization PPO continuous control policy network"
- "reward shaping gate passage drone racing reinforcement learning 2024 2025"
- "low variance policy PPO entropy penalty deterministic deployment"
- "mean action regularization reinforcement learning continuous control"
- "sim-to-real transfer drone agile flight PPO SAC 2024 2025"

## Current Architecture Notes

From inspecting the installed `Agent` class:
- **Activation:** Tanh (matches Swift/literature recommendations)
- **Init:** Orthogonal with gain=sqrt(2), actor final layer uses gain=0.01 (good)
- **log_std:** State-independent `nn.Parameter`, initial `[-1, -1, -1, 1]`
- **Observation normalization:** NONE — observations go raw into the network
- **Action smoothness:** d_act_th_coef and d_act_xy_coef exist in reward code but are set to 0.0 in exp_069/070 configs

## Papers Reviewed

### Paper 1: Learning Optimal Deterministic Policies with Stochastic Policy Gradients
- **ArXiv:** 2405.02235 | **Venue:** ICML 2024 / JMLR
- **Core technique:** Formal framework for the stochastic-to-deterministic deployment practice. Proves global convergence to the best deterministic policy under gradient domination. Shows how to tune exploration levels.
- **Key insight for us:** PPO optimizes expected return under the stochastic policy — the mean is a side effect, not the objective. If entropy is too high, the mean drifts from the optimal deterministic policy. Action-space exploration degrades the mean more than parameter-space exploration.
- **Proposed experiment:** Anneal entropy/log_std toward zero late in training to force the mean to converge.

### Paper 2: Smooth Exploration for Robotic RL (gSDE)
- **ArXiv:** 2005.05719 | **Venue:** CoRL 2022
- **Core technique:** Generalized State-Dependent Exploration. Noise is a learned function of policy features, re-sampled periodically (not per timestep). Produces temporally correlated, state-dependent exploration.
- **Key insight for us:** With i.i.d. Gaussian noise, the policy can learn to rely on noise as implicit dithering/recovery. When the mean is deployed alone, that recovery vanishes. gSDE forces the mean itself to encode recovery behavior because the noise is structured. Available in Stable-Baselines3.
- **Proposed experiment:** Implement state-dependent exploration in our PPO loop.

### Paper 3: CAPS — Regularizing Action Policies for Smooth Control
- **ArXiv:** 2012.06644 | **Venue:** ICRA 2021
- **Core technique:** Two regularization terms: (i) temporal smoothness — penalizes ||a_t - a_{t-1}||, (ii) spatial smoothness — penalizes action difference for nearby states. Tested on a real quadrotor: 80% reduction in power consumption.
- **Key insight for us:** Spatial smoothness forces the mean to be a Lipschitz-smooth function of state. This prevents the mean from being erratic while noise compensates. Temporal smoothness penalizes jittery actions. Both push the mean toward physically coherent actions.
- **Proposed experiment:** Re-enable d_act penalties or add CAPS-style spatial regularization.

### Paper 4: What Matters in Learning Zero-Shot Sim-to-Real RL (SimpleFlight)
- **ArXiv:** 2412.11764
- **Core technique:** Identifies 5 critical factors for sim-to-real quadrotor control. Factor 3: action smoothness regularization via ||u_t - u_{t-1}||_2 is critical. Also: rotation matrix (not quaternion) for actor input, large batch sizes.
- **Key insight for us:** Action-difference penalty forces the mean to produce smooth commands. Consecutive actions must be similar, so the mean cannot rely on noise-based corrections. This single factor was identified as critical for real-world transfer.
- **Proposed experiment:** Add action-difference penalty (we already have d_act_th_coef/d_act_xy_coef in the reward code — re-enable them).

### Paper 5: What Matters In On-Policy RL? A Large-Scale Study
- **ArXiv:** 2006.05990 | **Venue:** ICLR 2021
- **Core technique:** 250,000+ agents tested across 50+ design choices. Key findings: (1) Always use observation normalization. (2) Small final policy layer init (100x smaller). (3) Tanh activations. (4) Softplus for std. (5) GAE lambda=0.9.
- **Key insight for us:** **No observation normalization is a likely major issue.** Unnormalized observations cause the mean to be poorly conditioned. Our Agent has no running normalization. Our init and activation are good (orthogonal, Tanh, 0.01 final layer), but missing obs normalization could explain much of the deployment gap.
- **Proposed experiment:** Add observation normalization (running mean/std or VecNormalize wrapper).

### Paper 6: Structured Control Nets for Deep RL
- **ArXiv:** 1802.08311 | **Venue:** ICML 2018
- **Core technique:** Splits the policy into parallel nonlinear + linear sub-modules. The linear branch provides a stable baseline controller; the nonlinear branch handles complex maneuvering.
- **Key insight for us:** A linear control pathway would prevent catastrophic failures when the nonlinear network outputs degenerate actions in unfamiliar states. This is a safety net for the mean.
- **Proposed experiment:** Add a linear bypass in the actor architecture.

### Paper 7: Colored Noise in PPO
- **ArXiv:** 2312.11091 | **Venue:** AAAI 2024
- **Core technique:** Replaces white Gaussian noise with temporally correlated colored noise. Improves exploration and final performance in continuous control.
- **Key insight for us:** If the stochastic policy works because noise provides temporal smoothing (correlated perturbations smooth out jerky mean actions), colored noise could help the mean learn smoother actions. Related to gSDE but simpler.

### Paper 8: Regularization Matters in Policy Optimization
- **ArXiv:** 1910.09191
- **Core technique:** L2 regularization on the policy network brings large improvements, especially on harder tasks. Regularizing value networks less effective.
- **Key insight for us:** L2 on the policy constrains weight magnitudes, indirectly constraining how far the mean can drift. Simple baseline addition.

### Paper 9: Bootstrapping RL with Imitation for Vision-Based Agile Flight
- **ArXiv:** 2403.12203 | **Venue:** CoRL 2024
- **Core technique:** Three-stage: (1) RL teacher with privileged state, (2) distill to student via imitation, (3) fine-tune student with RL under performance constraint.
- **Key insight for us:** Distilling the stochastic teacher into a deterministic student via behavioral cloning could strip out noise-dependent features. The fine-tuning step ensures the student doesn't degrade.

### Paper 10: Competitive Multi-Agent Racing
- **ArXiv:** 2512.11781
- **Core technique:** Sparse "win the race" reward with multi-agent competition. Multi-agent policies transfer more reliably than single-agent progress-based.
- **Key insight for us:** Dense progress rewards may enable stochastic-stumbling-through. Sparser objectives could force more robust mean behavior.

## Synthesis

The papers collectively point to several explanations for our deployment gap:

1. **The mean was never optimized to be good.** PPO optimizes expected return under the stochastic policy. The mean is a side effect. Without mechanisms to force mean quality, the gap is expected (Paper 1).

2. **Missing observation normalization.** The "What Matters" study found this critical. Our Agent has NO obs normalization. Unnormalized inputs make the mean poorly conditioned (Paper 5).

3. **Noise-dependent navigation.** The stochastic policy may rely on noise for implicit dithering/recovery that vanishes at deployment. Action smoothness penalties (Papers 3, 4) and state-dependent exploration (Paper 2) address this directly.

4. **No late-training mean convergence.** Without entropy annealing, the policy never transitions toward deterministic behavior. The mean never has to carry all the performance (Papers 1, 7).

## Proposed Experiments

### exp_071 -- Observation Normalization
- **Hypothesis:** Missing observation normalization causes poorly conditioned mean actions. Adding running-mean/std normalization (as recommended by the 250K-agent "What Matters" study) should improve the mean policy's quality at deployment.
- **What to change:** Add VecNormalize or equivalent running observation normalization to the training environment wrapper. Single change from exp_069 baseline.
- **Expected outcome:** Improved deterministic benchmark gates, potentially large improvement given this is identified as a critical missing ingredient by the largest PPO design-choices study.
- **Paper basis:** "What Matters In On-Policy RL?" (2006.05990), Swift (both normalize observations)

### exp_072 -- Action Smoothness Penalty
- **Hypothesis:** The mean policy produces jerky, unstable commands because there is no incentive for temporal coherence. Re-enabling action-difference penalties (d_act_th_coef, d_act_xy_coef) forces the mean to produce smooth action sequences, reducing reliance on stochastic corrections.
- **What to change:** From exp_069 baseline, set d_act_th_coef=0.4, d_act_xy_coef=1.0 (the default values in train_racing.py). These penalties already exist in the reward code but are disabled (set to 0.0) in recent configs.
- **Expected outcome:** Smoother deterministic actions, potentially more stable flight and gate passage. Training reward may decrease slightly but benchmark should improve.
- **Paper basis:** CAPS (2012.06644), SimpleFlight (2412.11764)

### exp_073 -- Entropy Annealing (Late Mean Convergence)
- **Hypothesis:** The mean doesn't converge because entropy stays constant throughout training. Annealing ent_coef from 0.01→0.001 over training forces the distribution to collapse onto the mean, making the mean carry all performance by end of training.
- **What to change:** From exp_069 baseline, add ent_coef annealing schedule: 0.01 for first 70% of training, linearly anneal to 0.001 over remaining 30%.
- **Expected outcome:** Mean policy improves in late training as variance shrinks. Deterministic benchmark should improve.
- **Paper basis:** "Learning Optimal Deterministic Policies" (2405.02235), "Entropy Annealing" (2405.20250)
- **Risk note:** exp_064 failed with ent_coef=0.03 start, but that was 3x higher initial value. This starts from the working ent_coef=0.01 and only anneals late.

### exp_074 -- Obs Normalization + Action Smoothness (Combined)
- **Hypothesis:** If both obs normalization and action smoothness independently help, combining them from the exp_069 baseline should produce the best deterministic deployment yet.
- **What to change:** Both exp_071 and exp_072 changes applied together.
- **Expected outcome:** Best deterministic gate passage rate. If it works, follow up with clean ablations to attribute.
- **Paper basis:** Combined evidence from Papers 3, 4, 5

### [DEFERRED] exp_075 -- State-Dependent Exploration (gSDE)
- **Hypothesis:** State-dependent exploration forces the mean to encode recovery behavior, directly reducing the deployment gap.
- **What to change:** Replace i.i.d. Gaussian noise with gSDE in the PPO loop. Requires code changes to train_racing.py Agent class.
- **Expected outcome:** Better mean-policy quality, smoother deployment.
- **Paper basis:** gSDE (2005.05719)
- **Scope note:** Requires non-trivial code changes. Defer until simpler interventions (obs norm, action smoothness) are tested.
