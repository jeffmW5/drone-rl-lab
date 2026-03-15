# Drone RL Lab — Learning Report

How our reinforcement learning drone racing system works, what every variable means, and where we stand vs the competition.

---

## 1. How PPO Works (Plain English)

**Reinforcement Learning** = an agent learns by trial and error. It takes actions in an environment, gets rewards, and updates its strategy to get more reward over time.

**PPO (Proximal Policy Optimization)** is our specific RL algorithm. Here's the loop:

```
1. Fly the drone (take actions) for 8 steps across 64 parallel simulations
2. Collect: what did I see? what did I do? what reward did I get?
3. Compute "advantages" — was each action better or worse than expected?
4. Update the neural network weights to make good actions more likely
5. Repeat from step 1
```

The "proximal" part means PPO is careful — it clips how much the policy can change per update (controlled by `clip_coef = 0.26`). This prevents the agent from making a big change that accidentally destroys what it already learned.

**Two networks work together:**
- **Actor** (policy): "Given what I see, what should I do?" → outputs action probabilities
- **Critic** (value): "Given what I see, how much total reward do I expect?" → outputs a single number

**GAE (Generalized Advantage Estimation):** Computes "how much better was this specific action compared to what I normally do in this state?" This is what actually drives learning — positive advantage = "do more of this", negative = "do less."

---

## 2. The Neural Network

```
Input (73 dims) ──→ [Linear 64] ──→ [Tanh] ──→ [Linear 64] ──→ [Tanh] ──→ Output
                    (both actor and critic have this structure)
```

**Actor head:** outputs mean of 4 actions + learned log-standard-deviation
- Log-std initialized to `[-1, -1, -1, 1]`
- This means: low exploration on roll/pitch/yaw (already bounded), high exploration on thrust (needs more experimenting)
- Final layer uses Tanh to bound outputs to [-1, 1]

**Critic head:** outputs single value estimate (expected future reward)

**Size:** ~9,000 trainable parameters. This is tiny by modern standards — the network is small because the observation space is small and the physics are relatively low-dimensional.

**Weight initialization:** Orthogonal initialization (helps with gradient flow in Tanh networks). The actor's final layer uses std=0.01 (start with near-zero actions = hover), the critic's final layer uses std=1.0.

---

## 3. What the Agent Sees (73 Observation Dimensions)

Every simulation step, the agent receives a vector of 73 numbers:

| Component | Dims | Description |
|-----------|:----:|-------------|
| `pos` | 3 | Drone position [x, y, z] in meters |
| `quat` | 4 | Drone orientation as quaternion [x, y, z, w] |
| `vel` | 3 | Linear velocity [vx, vy, vz] m/s |
| `ang_vel` | 3 | Angular velocity [wx, wy, wz] rad/s |
| `local_samples` | 30 | Relative positions to next 10 trajectory waypoints (3 coords each) |
| `prev_obs` frame 1 | 13 | Previous step: [pos, quat, vel, ang_vel] |
| `prev_obs` frame 2 | 13 | Two steps ago: [pos, quat, vel, ang_vel] |
| `last_action` | 4 | What the agent did last step [roll, pitch, yaw, thrust] |
| **Total** | **73** | |

**Important detail:** The raw environment also provides `gates_pos`, `gates_quat`, `obstacles_pos`, etc. But the `RandTrajEnv` wrapper consumes gate positions to compute a trajectory, then gives the agent `local_samples` (relative positions to the next 10 trajectory waypoints). So the agent sees "where should I fly next" rather than "where are the gates."

**With n_obs=0:** No `prev_obs`, total drops to 47 dims. Faster to train but agent can't infer acceleration from position history.

---

## 4. What the Agent Does (4 Action Dimensions)

| Action | Range | What it controls |
|--------|:-----:|-----------------|
| Roll | [-1, 1] | Tilt left/right (bank for turning) |
| Pitch | [-1, 1] | Tilt forward/back (controls forward speed) |
| Yaw | [-1, 1] | Rotate around vertical axis (blocked to 0 by wrapper) |
| Thrust | [-1, 1] | Vertical force (up/down, also affected by tilt) |

The `NormalizeActions` wrapper maps [-1, 1] to the drone's actual control ranges. The `AngleReward` wrapper blocks yaw to 0 (the drone doesn't need to rotate, just bank and pitch).

---

## 5. The Reward Function

Every step, the agent gets a reward signal that tells it how well it's doing:

```
total_reward = base_reward + angle_penalty + energy_penalty + smoothness_penalties
```

### Base Reward: Follow the Trajectory
```python
reward = exp(-2.0 * distance_to_next_waypoint)
```
- Distance = 0m → reward = 1.0 (perfect)
- Distance = 0.5m → reward = 0.37
- Distance = 1.0m → reward = 0.14
- Distance = 2.0m → reward = 0.02 (basically zero)
- Crashed → reward = -1.0

This exponential shape means the agent gets much more reward for being very close vs. roughly close. It creates a strong gradient toward the trajectory.

### Penalties (Subtract from Base Reward)

| Penalty | Formula | Weight | Purpose |
|---------|---------|:------:|---------|
| Angle | -rpy_coef * norm(euler_angles) | 0.06 | Don't tilt excessively |
| Energy | -act_coef * thrust^2 | 0.02 | Don't waste energy hovering hard |
| Thrust smoothness | -d_act_th_coef * (thrust - prev_thrust)^2 | 0.4 | Don't jerk the throttle |
| Roll/pitch smoothness | -d_act_xy_coef * sum((roll - prev_roll)^2 + (pitch - prev_pitch)^2) | 1.0 | Don't jerk the attitude |

**Key insight:** The reward is about trajectory-following, NOT about passing through gates. The trajectory happens to pass through gates, but the agent only knows "follow these waypoints." This is why Level 2 (randomized gates) breaks things — the trajectory might not go through the moved gates.

---

## 6. Every Hyperparameter Explained

### PPO Algorithm Knobs

| Parameter | Value | What It Does | Turn Up | Turn Down |
|-----------|:-----:|--------------|---------|-----------|
| `learning_rate` | 0.0015 | Size of weight updates | Faster learning, risk oscillation | Slower, more stable convergence |
| `gamma` | 0.94 | Discount factor for future rewards | Values rewards further in the future | Myopic, focuses on immediate reward |
| `gae_lambda` | 0.97 | Advantage estimation smoothing | Lower bias, higher variance estimates | Higher bias, lower variance (more stable) |
| `clip_coef` | 0.26 | Max policy change per update | Allows bigger strategy shifts | Smaller, safer updates |
| `ent_coef` | 0.007 | Entropy bonus (exploration) | More random exploration | Exploits current best strategy |
| `vf_coef` | 0.7 | Value function loss weight | Better value predictions | Prioritizes policy improvement |
| `max_grad_norm` | 1.5 | Gradient clipping | Allows larger parameter changes | Prevents training instability |
| `update_epochs` | 10 | Reuse each batch N times | More learning per data collection | Less risk of overfitting to one batch |
| `num_minibatches` | 8 | Split batch into N chunks | More gradient updates, smaller each | Fewer but larger gradient updates |

### Scale Knobs

| Parameter | CPU | GPU | What It Does |
|-----------|:---:|:---:|--------------|
| `num_envs` | 64 | 1024 | Parallel drone simulations. More = smoother gradient estimates + faster wall-clock |
| `num_steps` | 8 | 8 | Steps collected before each PPO update. Short = faster updates, less temporal context |
| `total_timesteps` | 500k | 3-10M | Total training budget. More = more learning, diminishing returns |
| `n_obs` | 0 or 2 | 2 | Observation history stacking. 2 = can infer velocity/acceleration from position deltas |
| `budget_seconds` | 600 | 1800-7200 | Wall-clock training cutoff |

**Batch size** = `num_envs * num_steps`
- CPU: 64 * 8 = 512 samples per PPO update
- GPU: 1024 * 8 = 8,192 samples per PPO update (16x more signal per update)

### Reward Shaping Knobs

| Parameter | Value | Effect of Increasing |
|-----------|:-----:|---------------------|
| `rpy_coef` | 0.06 | Drone stays more level (less aggressive banking through turns) |
| `act_coef` | 0.02 | Uses less thrust (more efficient but potentially slower) |
| `d_act_th_coef` | 0.4 | Smoother thrust changes (less jerky altitude control) |
| `d_act_xy_coef` | 1.0 | Smoother roll/pitch changes (less aggressive maneuvering) |

**Trade-off:** Higher smoothness penalties = safer, smoother flight but potentially slower lap times. The Kaggle winners likely tuned these down to allow more aggressive flying.

---

## 7. Our Experiments So Far

### Experiment 010 — Racing Baseline (Level 0, CPU)
- **Config:** n_obs=0, 500k steps, 64 envs, 600s budget
- **Result:** reward = 7.36, completed all 500k steps
- **Sim test:** 5/5 finishes, average 13.36s (within 0.024s of reference model)
- **Lesson:** The pipeline works. With n_obs=0, the agent converges in 10 min on CPU.

### Experiment 013 — n_obs Fix (Level 0, CPU)
- **Config:** n_obs=2, 500k steps, 64 envs, 600s budget
- **Result:** reward = 5.02, only completed 297k/500k steps (hit time limit)
- **Sim test:** 0/5 finishes (crashes on every run)
- **Lesson:** n_obs=2 makes observations 55% larger (47 → 73 dims), training is much slower on CPU. Agent was severely undertrained. Needs more compute (GPU).

### Level 2 Benchmark — All Controllers
Everyone was tested on the competition level (randomized gates + physics):

| Controller | Type | Avg Time | Finishes | Gates |
|-----------|------|:--------:|:--------:|:-----:|
| State controller | Trajectory following | 5.96s | 1/5 | 0-4 |
| PID attitude | Classical PID | 8.59s | 2/5 | 0-4 |
| Their RL (pre-trained) | RL (trained on L0) | 7.25s | 0/5 | 0-3 |
| Our RL (exp_013) | RL (undertrained) | 4.30s | 0/5 | 0-2 |

**Nobody reliably finishes Level 2.** All controllers follow trajectories built from fixed waypoints that don't adapt to randomized gate positions.

---

## 8. Kaggle Competition Leaderboard

**Competition:** lsy-drone-racing-ws-25 (TUM, Winter Semester 2025)
**Task:** Complete a drone racing course on Level 2 (randomized gates + physics)
**Scoring:** Average lap time across evaluation runs

| Rank | Team | Score (seconds) | Submissions |
|:----:|------|:---------------:|:-----------:|
| 1 | Team Y | **3.39** | 7 |
| 2 | Group6 | **4.89** | 17 |
| 3 | Limo | **5.02** | 17 |
| 4 | Liangyu Chen, Tuo Yang | 5.61 | 9 |
| 5 | Jai Seth | 9.56 | 10 |
| 6 | Elena Kuznetsova | 22.51 | 1 |
| 7 | RandomUsername2374 | 24.29 | 1 |
| 8 | Marcel Rath | 27.07 | 1 |
| 9 | Radu Cristian | 28.19 | 1 |
| 10 | Yufei Hua | 29.99 | 1 |

**Our target: sub-5s (top 3)**

The top teams submitted 7-17 times, suggesting significant iteration. The bottom teams submitted once (likely the default controllers).

---

## 9. The Gap: Where We Are vs Where We Need to Be

### Current status
- Our best model (exp_010) completes Level 0 in 13.36s but can't handle Level 2
- Level 2 breaks ALL existing controllers because trajectories don't adapt to moved gates
- We've only trained on CPU with 64 envs and 500k steps

### What the Kaggle winners likely did
1. **Trained directly on Level 2** — so the agent experiences randomized gates during training
2. **Used GPU + many envs** — 1024+ parallel simulations for faster, smoother training
3. **Trained for millions of steps** — 3-10M+ timesteps
4. **Tuned reward coefficients** — possibly reduced smoothness penalties for faster, more aggressive flight
5. **Possibly used curriculum learning** — start on easy levels, progressively add difficulty

### Our GPU plan
| Experiment | Level | Steps | Envs | Purpose |
|-----------|:-----:|:-----:|:----:|---------|
| exp_014 | 0 | 1.5M | 1024 | Validate GPU works, converge n_obs=2 |
| exp_015 | **2** | 3M | 1024 | First competition attempt |
| exp_016 | **2** | 10M | 1024 | Extended training if 3M isn't enough |

**Why GPU changes everything:**
- 1024 envs × 8 steps = 8,192 samples per update (vs 512 on CPU)
- GPU parallelizes the matrix math across all envs simultaneously
- Estimated: 3M steps in ~5-10 minutes on RTX 3090 (vs hours on CPU)
- n_obs=2 won't be a bottleneck — GPU handles the larger obs easily

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **PPO** | Proximal Policy Optimization — the RL algorithm |
| **Actor** | Neural network that decides actions |
| **Critic** | Neural network that estimates expected future reward |
| **GAE** | Generalized Advantage Estimation — measures how good an action was |
| **Rollout** | A sequence of steps collected before updating the network |
| **Batch size** | num_envs * num_steps = total samples per PPO update |
| **Epoch** | One pass through the entire batch during PPO update |
| **Entropy** | Measure of randomness in the policy (higher = more exploration) |
| **Clipping** | Limiting how much the policy can change per update |
| **Advantage** | How much better an action was vs the average for that state |
| **Discount (gamma)** | How much to value future vs immediate rewards |
| **Observation stacking** | Including past observations so the agent can see trends |
| **Level 0** | Perfect knowledge — fixed gates, no randomization |
| **Level 1** | Randomized physics (mass, inertia) but fixed gates |
| **Level 2** | Competition level — randomized gates + physics + obstacles |
