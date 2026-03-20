# INBOX -- Experiment Queue

> Written by Windows Claude (orchestrator). Linux Claude processes tasks top-to-bottom.
> Mark each task [DONE] when complete. Orchestrator cleans up completed tasks on next pass.

---

## Queue

### [DONE] exp_028 -- High Speed Reward (fine-tune exp_026, no random gates)
**Result:** Reward 16.95 (highest ever), benchmark 0/5 finishes, **0.2 avg gates** (FIRST GATE PASSAGE in run 4!), avg 0.94s.
speed_coef=1.0 taught navigation but destroyed hover stability. Sweet spot is 0.3-0.5.

---

### [DONE] exp_029 -- Balanced Speed Reward (hover + navigation)
**Result:** Reward 16.52 ± 2.18 (very stable training), benchmark 0/5 finishes, 0 gates, avg 29.98s.
Hover perfectly preserved (29.98s = max episode!) but speed_coef=0.4 wasn't enough to break out of
hover local optimum. The phase transition between hover-only and navigation is between 0.4 and 1.0.

---

### [DONE] exp_030 + exp_031 -- Speed Coefficient Phase Transition Sweep
**exp_030 (0.55):** Reward 15.64, 0 gates, avg 25.98s (4/5 hover, 1/5 crash at 9.98s — edge of transition)
**exp_031 (0.70):** Reward 10.44, 0 gates, avg 2.02s (hover destroyed, no navigation)
**Conclusion:** Sharp phase transition between 0.55 and 0.70 — goes straight from hover to crash with no
navigation sweet spot. exp(-k*dist) proximity reward is fundamentally flawed. PBRS needed.

---

**exp_030 config** — create `configs/exp_030_speed_055.yaml`:

```yaml
name: exp_030_speed_055
backend: racing
hypothesis: "Phase transition sweep: speed_coef=0.55 fine-tuned from exp_026. exp_028 (1.0) first gate
  pass but crashes (0.94s), exp_029 (0.4) hover only (29.98s). 0.55 is lower half of transition zone.
  Very conservative LR=0.0001 to protect pretrained hover skill. Success = >5s flight AND >0 gates."
budget_seconds: 5400
racing:
  env_type: race
  level: level2
  race_config: level2_attitude.toml
  pretrained_ckpt: /workspace/drone-rl-lab/results/exp_026_racecore_vzpenalty/model.ckpt
  total_timesteps: 8000000
  num_envs: 512
  num_steps: 8
  learning_rate: 0.0001
  anneal_lr: true
  gamma: 0.97
  gae_lambda: 0.97
  update_epochs: 10
  num_minibatches: 8
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  max_grad_norm: 1.5
  n_obs: 2
  cuda: true
  seed: 42
  gate_bonus: 20.0
  proximity_coef: 1.0
  speed_coef: 0.55
  rpy_coef: 0.06
  act_coef: 0.02
  d_act_th_coef: 0.4
  d_act_xy_coef: 1.0
  max_episode_steps: 1500
  alt_coef: 1.5
  survive_coef: 0.5
  vz_coef: 0.5
  vz_threshold: 0.5
  oob_coef: 8.0
  z_low: -0.05
  z_high: 1.3
  random_gate_start: false
```

---

**exp_031 config** — create `configs/exp_031_speed_070.yaml`:

```yaml
name: exp_031_speed_070
backend: racing
hypothesis: "Phase transition sweep: speed_coef=0.70 fine-tuned from exp_026. Upper half of the
  transition zone (0.4->1.0). If 0.55 still hovers, 0.70 may be the minimum that forces navigation.
  If 0.55 achieves gate passages, 0.70 tests whether we can get more. LR=0.0001 same as exp_030."
budget_seconds: 5400
racing:
  env_type: race
  level: level2
  race_config: level2_attitude.toml
  pretrained_ckpt: /workspace/drone-rl-lab/results/exp_026_racecore_vzpenalty/model.ckpt
  total_timesteps: 8000000
  num_envs: 512
  num_steps: 8
  learning_rate: 0.0001
  anneal_lr: true
  gamma: 0.97
  gae_lambda: 0.97
  update_epochs: 10
  num_minibatches: 8
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  max_grad_norm: 1.5
  n_obs: 2
  cuda: true
  seed: 42
  gate_bonus: 20.0
  proximity_coef: 1.0
  speed_coef: 0.70
  rpy_coef: 0.06
  act_coef: 0.02
  d_act_th_coef: 0.4
  d_act_xy_coef: 1.0
  max_episode_steps: 1500
  alt_coef: 1.5
  survive_coef: 0.5
  vz_coef: 0.5
  vz_threshold: 0.5
  oob_coef: 8.0
  z_low: -0.05
  z_high: 1.3
  random_gate_start: false
```

---

### [DONE] exp_032 -- PBRS Delta-Progress Reward (replaces exp(-k*dist) proximity)
**Result:** Reward 11.87 ± 3.76, benchmark 0/5 finishes, 0 gates, avg 2.92s. PBRS broke hover optimum
(model attempts lateral movement) but crashes before reaching gates. Slightly better than exp_031 (2.02s).
**Depends on:** exp_030 and exp_031 complete

**Context:** The proximity reward `exp(-proximity_coef * dist)` is NOT potential-based (Ng, Harada,
Russell, ICML 1999). It rewards absolute closeness to a gate, not movement toward it. The hover local
optimum is high-reward because at z≈0.7 near the start, the drone already gets alt_reward + survive +
some proximity from being ~1.5m from gate 0. There is almost no per-step incentive to move laterally
because the proximity gradient is tiny (Δexp ≈ 0.005/step for 0.01m movement at 1.5m distance).

**Fix:** Replace `exp(-proximity_coef * dist)` with `max(0, prev_dist - curr_dist) * progress_coef`.
This is a valid potential-based shaping function Φ(s) = -dist, F = γΦ(s') - Φ(s). When hovering in
place, prev_dist ≈ curr_dist → progress ≈ 0 (no fake reward for proximity). When approaching gate,
prev_dist > curr_dist → positive reward. Cannot be hacked by hovering near gate.

Reference: Ng et al. 1999 (ICML), Skalse et al. NeurIPS 2022 ("Defining and Characterizing Reward
Hacking" arXiv:2209.13085), Kaufmann et al. Nature 2023 (Swift uses progress reward not proximity).

**Code change authorized** — modify `/media/lsy_drone_racing/lsy_drone_racing/control/train_race.py`:

**Change 1** — In `RaceRewardAndObs.__init__()`, add `progress_coef` parameter and `_prev_dist` tracker.

Find this block in `__init__` signature:
```python
        random_gate_start: bool = False,
        random_gate_ratio: float = 1.0,
```
Replace with:
```python
        random_gate_start: bool = False,
        random_gate_ratio: float = 1.0,
        progress_coef: float = 0.0,
```

Find this block in `__init__` body (the attribute assignments near the end of `__init__`):
```python
        self._prev_target_gate = None
        self._was_done = None  # Track terminated|truncated for autoreset detection
```
Replace with:
```python
        self._prev_target_gate = None
        self._prev_dist = None
        self.progress_coef = progress_coef
        self._was_done = None  # Track terminated|truncated for autoreset detection
```

**Change 2** — In `reset()`, initialize `_prev_dist` from the initial observation.

Find this block in `reset()`:
```python
        self._prev_target_gate = jp.array(obs["target_gate"])
        self._was_done = None
        return self._preprocess_obs(obs), info
```
Replace with:
```python
        self._prev_target_gate = jp.array(obs["target_gate"])
        self._was_done = None
        # Initialize prev_dist for delta-progress reward
        if self.progress_coef > 0.0:
            safe_t = jp.clip(jp.array(obs["target_gate"]), 0, self.n_gates - 1)
            idx0 = jp.arange(self.num_envs)
            t_pos = jp.array(obs["gates_pos"])[idx0, safe_t]
            self._prev_dist = jp.linalg.norm(t_pos - jp.array(obs["pos"]), axis=-1)
        return self._preprocess_obs(obs), info
```

**Change 3** — In `step()`, replace the proximity computation to support both modes.

Find this block in `step()`:
```python
        proximity = jp.exp(-self.proximity_coef * dist)
```
Replace with:
```python
        if self.progress_coef > 0.0 and self._prev_dist is not None:
            # Potential-based reward shaping: F = gamma*Phi(s') - Phi(s), Phi(s) = -dist
            # Only reward positive progress (approaching gate), not retreating
            proximity = self.progress_coef * jp.maximum(self._prev_dist - dist, 0.0)
        else:
            proximity = jp.exp(-self.proximity_coef * dist)
```

**Change 4** — In `step()`, update `_prev_dist` at the end. Note: for autoreset envs, `dist` here is
already computed from the post-reset obs (the VecEnv returns reset obs on done), so storing it is
correct — the next episode starts fresh with accurate prev_dist.

Find this block at the end of `step()`:
```python
        self._prev_target_gate = jp.array(target_gate)
        self._was_done = np.array(terminated | truncated)
```
Replace with:
```python
        self._prev_target_gate = jp.array(target_gate)
        if self.progress_coef > 0.0:
            self._prev_dist = dist
        self._was_done = np.array(terminated | truncated)
```

**Change 5** — In `make_race_envs()`, pass `progress_coef` through to `RaceRewardAndObs`.

Find this block in `make_race_envs()`:
```python
        random_gate_start=coefs.get("random_gate_start", False),
        random_gate_ratio=coefs.get("random_gate_ratio", 1.0),
```
Replace with:
```python
        random_gate_start=coefs.get("random_gate_start", False),
        random_gate_ratio=coefs.get("random_gate_ratio", 1.0),
        progress_coef=coefs.get("progress_coef", 0.0),
```

**After code change**, create `configs/exp_032_pbrs_progress.yaml` and run it.
Fine-tune from whichever of exp_030/031 achieved the best gate count (if tied, use exp_031).
If neither passed gates, fine-tune from exp_026.

```yaml
name: exp_032_pbrs_progress
backend: racing
hypothesis: "Replace exp(-k*dist) proximity with PBRS delta-progress reward (Ng et al. ICML 1999).
  progress_coef * max(prev_dist - curr_dist, 0) only rewards approaching the gate, not hovering near
  it. The hover local optimum gets ~0 progress reward per step (not moving). Set progress_coef=50 to
  make approaching gate at 0.01m/step = 0.5 reward/step, comparable to survive_coef=0.5. proximity_coef
  is still present but unused when progress_coef>0. Fine-tune from best of exp_030/031 (or exp_026)."
budget_seconds: 5400
racing:
  env_type: race
  level: level2
  race_config: level2_attitude.toml
  pretrained_ckpt: /workspace/drone-rl-lab/results/exp_031_speed_070/model.ckpt  # or best available
  total_timesteps: 8000000
  num_envs: 512
  num_steps: 8
  learning_rate: 0.0001
  anneal_lr: true
  gamma: 0.97
  gae_lambda: 0.97
  update_epochs: 10
  num_minibatches: 8
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  max_grad_norm: 1.5
  n_obs: 2
  cuda: true
  seed: 42
  gate_bonus: 20.0
  proximity_coef: 1.0   # unused when progress_coef > 0, kept for reference
  progress_coef: 50.0   # 0.01m/step approach -> 0.5 reward/step, on par with survive_coef
  speed_coef: 0.3       # lighter speed signal since progress already handles direction
  rpy_coef: 0.06
  act_coef: 0.02
  d_act_th_coef: 0.4
  d_act_xy_coef: 1.0
  max_episode_steps: 1500
  alt_coef: 1.5
  survive_coef: 0.5
  vz_coef: 0.5
  vz_threshold: 0.5
  oob_coef: 8.0
  z_low: -0.05
  z_high: 1.3
  random_gate_start: false
```

---

### [DONE] exp_033 -- Truncation vs Termination Fix (Pardo et al. ICML 2018)
**Result:** Reward 14.37 ± 4.11, benchmark 0/5 finishes, 0 gates, avg 24.58s. Truncation fix restored
hover stability (4/5 hover @ 29.98s vs exp_032's 2.92s crash), but back in hover trap. Better value
estimates made model conservative — the PBRS progress reward isn't strong enough with good GAE.

---

### [DONE] exp_034 -- PBRS + Higher Speed (break hover with directional reward)
**Result:** Reward 17.26 ± 5.50, benchmark 0/5 finishes, 0 gates, avg 29.98s (perfect hover).
PBRS eliminated the crash at speed_coef=0.7 (exp_031 was 2.02s) but model still hovers. High training
rewards came from stochastic exploration, not learned navigation. Hover optimum persists because
alt_coef(1.5)+survive_coef(0.5)=2.0/step dominates any intermittent progress/speed reward.
**Config:** `configs/exp_034_pbrs_speed.yaml`
**Depends on:** exp_033 complete

**Goal:** The speed sweep (exp_028-031) showed a sharp phase transition between hover (≤0.55) and crash (≥0.70) because the proximity reward `exp(-k*dist)` rewarded hovering near gates. PBRS replaces that with delta-progress reward — no hover trap. speed_coef=0.7 should now create a third regime: stable navigation (>5s flight AND >0 gates). The truncation fix from exp_033 provides accurate value estimates, and PBRS provides directional incentive. Speed adds the urgency to actually move.

**What's new:**
- speed_coef increased from 0.3 (exp_033) to 0.7
- All other reward coefficients unchanged from exp_033
- Fine-tune from exp_033 checkpoint (has PBRS + truncation fix + hover skill)
- No code changes needed — only config change

**Training:** 512 envs, 8M steps on GPU (RTX 3090)

**Success criteria:**
- avg flight time >5s (not just crashing like exp_031's 2.02s)
- avg gates >0 (any gate passage proves PBRS eliminated the phase transition)
- Ideal: >0.5 avg gates with >10s flight time

**If it doesn't work:**
- If crashes (like exp_031): try speed_coef=0.55 with PBRS (exp_035) — more conservative
- If hovers (like exp_033): try speed_coef=1.0 with PBRS (exp_035) — if still hovering, the problem isn't speed
- If crashes AND no gates: reduce survive_coef from 0.5 to 0.2 to lower hover incentive

---

### [DONE] exp_035 -- Remove Survive Reward (break hover anchor)
**Result:** Reward 11.32 ± 5.52, benchmark 0/5 finishes, 0 gates, avg 0.96s. Removing survive_coef
destroyed hover — crashes instantly. survive_coef brackets: 0.5=hover(29.98s), 0.0=crash(0.96s).
**Config:** `configs/exp_035_no_survive.yaml`
**Depends on:** exp_034 complete

**Goal:** survive_coef=0.5 gives 2.0/step guaranteed reward (with alt_coef=1.5) for hovering. This
is the highest reward-to-risk ratio, anchoring the policy mean at hover regardless of PBRS or speed
signals. exp_034 proved PBRS eliminates crashes at speed=0.7 but the mean policy still hovers.
Removing survive_coef forces all reward to come from altitude (passive, ~1.5/step) + progress/speed
(active, requires movement). Navigation at 1 m/s yields ~1.5 (alt) + 1.0 (progress) + 0.7 (speed) =
3.2/step vs hover at ~1.5/step (alt only). Clear 2x advantage for moving.

**What's new:**
- survive_coef reduced from 0.5 to 0.0 (single variable change from exp_034)
- All other coefficients unchanged (PBRS progress_coef=50, speed_coef=0.7, alt_coef=1.5, oob_coef=8.0)
- Fine-tune from exp_034 checkpoint (has PBRS + truncation fix + stable hover at 29.98s)
- No code changes needed — config only

**Training:** 512 envs, 8M steps on GPU (RTX 3090)

**Success criteria:**
- avg flight time >5s (model doesn't just crash without survive reward)
- avg gates >0 (model navigates toward gates instead of hovering)
- Ideal: >0.5 avg gates with >10s flight time = first "stable navigation" regime

**If it doesn't work:**
- If crashes (<3s): alt_coef=1.5 alone isn't enough to prevent crashing. Try survive_coef=0.1 (minimal)
- If still hovers (>25s, 0 gates): alt_coef itself is the trap. Try alt_coef=0.5 + survive_coef=0
- If erratic movement but no gates: progress_coef may need increasing (try 100)

---

### [DONE] exp_036 -- Binary Search survive_coef (0.15)
**Result:** Reward 28.61 ± 0.22 (HIGHEST EVER, very stable), benchmark 0/5 finishes, 0 gates, avg 0.93s.
survive_coef=0.15 crashes like survive_coef=0.0 (0.96s). Bracket narrows: 0.5=hover, 0.15=crash.
Phase transition is sharp between 0.15-0.5. Training reward misleading — stochastic exploration hit 29+
but deterministic mean policy crashes. Reward tuning alone may not solve hover-or-crash.
**Config:** `configs/exp_036_survive_015.yaml`
**Depends on:** exp_035 complete

**Goal:** We've bracketed survive_coef: 0.5=hover (29.98s, 0 gates), 0.0=crash (0.96s, 0 gates).
Binary search the midpoint at 0.15. If the survive parameter has a sweet spot, 0.15 should produce
intermediate behavior (5-20s flight with some navigation). If it shows the same binary phase
transition (either hover >25s or crash <3s), the problem is fundamental and requires curriculum
learning or architecture changes, not reward tuning.

**What's new:**
- survive_coef reduced from 0.5 (exp_034) to 0.15 (midpoint of bracketed range)
- All other coefficients unchanged from exp_034 (PBRS progress_coef=50, speed_coef=0.7, alt_coef=1.5)
- Fine-tune from exp_034 checkpoint (stable hover baseline with PBRS + truncation fix)
- No code changes needed — config only

**Training:** 512 envs, 8M steps on GPU (RTX 3090)

**Success criteria:**
- Flight time 5-20s (intermediate, not hover and not crash)
- Any gate passage (>0 avg gates)
- If binary again: conclusive evidence that reward tuning alone won't solve this

**If it doesn't work:**
- If crashes (<3s): try 0.25 (closer to hover side). If THAT also crashes, the transition is very sharp
- If hovers (>25s, 0 gates): try 0.05 (closer to crash side)
- If binary at all tested values: pivot to curriculum (shorter max_episode_steps=300) or higher entropy

---

### [DONE] exp_037 -- Binary Search survive_coef=0.3 (final bracket step)
**Result:** Reward 18.10 ± 9.66 (bimodal sawtooth), benchmark 0/5 finishes, 0 gates, avg 1.62s.
4/5 crash at ~0.85s but ONE outlier at 4.64s — first intermediate flight time. survive_coef=0.3
is right at the phase transition edge. Bracket complete: 0.5=hover, 0.3=crash(edge), 0.15=crash,
0.0=crash. **Reward tuning alone cannot solve hover-or-crash.** Pivot to ent_coef/curriculum needed.
**Config:** `configs/exp_037_survive_030.yaml`
**Depends on:** exp_036 complete

**Goal:** Final binary search step on survive_coef. Bracket: 0.15=crash (0.93s), 0.5=hover (29.98s).
Midpoint = 0.3. This is the LAST pure reward-tuning experiment. If 0.3 shows binary behavior
(either crash <3s or hover >25s), we confirm there's no survive_coef sweet spot and pivot to
fundamentally different approaches (higher entropy, curriculum, or stochastic deployment). The key
insight from exp_036: training reward 28.61 (highest ever) but benchmark crashes — the stochastic
policy navigates during exploration but the deterministic mean doesn't. This is a policy optimization
problem, not a reward design problem.

**What's new:**
- survive_coef=0.3 (midpoint of narrowed bracket [0.15, 0.5])
- All other coefficients unchanged from exp_034/036 (PBRS, speed=0.7, alt=1.5)
- Fine-tune from exp_034 checkpoint (stable hover with PBRS + truncation fix)
- No code changes needed — config only

**Training:** 512 envs, 8M steps on GPU (RTX 3090)

**Success criteria:**
- Flight time 5-20s (intermediate behavior = sweet spot found!)
- Any gate passage (>0 avg gates)
- If binary: conclusive evidence to pivot strategy

**If it doesn't work:**
- If crashes (<3s): phase transition is between 0.3-0.5, very sharp. Pivot to higher ent_coef (0.05)
- If hovers (>25s, 0 gates): try 0.2 (sweet spot between 0.15-0.3)
- If ALL survive values show binary: pivot to curriculum (max_episode_steps=300) or deploy stochastic policy

**Depends on:** exp_032 complete

**Context:** The current GAE computation in `train_racing.py` conflates two distinct episode-ending
signals: `terminated` (true crash, absorbing state) and `truncated` (episode timeout, not absorbing).
Pardo et al. 2018 (arXiv:1712.00378) proves that treating timeouts as terminal states biases the
value function: the agent implicitly learns to avoid end-of-episode states, creating conservative
behavior. In our case, `max_episode_steps=1500` timeouts are NOT crashes — bootstrapping should
continue at timeout boundaries, not treat them as terminal absorbing states.

Current bug in `train_racing.py`:
```python
next_done = torch.Tensor(np.logical_or(_to_np(terminated), _to_np(truncated)).astype(float))
```
Then in GAE: `nextnonterminal = 1.0 - dones_buf[t + 1]` — this zeroes out bootstrap at timeouts,
which is incorrect. The fix is to track `terminated` separately and use it (not dones) for GAE.

**Code change authorized** — modify `train_racing.py`:

**Change 1** — Add `terminated_buf` alongside the existing buffers.

Find this block (in the buffer initialization section):
```python
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
```
Replace with:
```python
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminated_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
```

**Change 2** — Store `terminated` separately in the rollout collection.

Find this block in the rollout collection loop:
```python
            next_obs, reward, terminated, truncated, info = envs.step(_action_for_env(action, jax_on_gpu))
            reward = torch.tensor(_to_np(reward), dtype=torch.float32).to(device)
            rewards_buf[step] = reward
            next_obs = torch.Tensor(_to_np(next_obs)).to(device)
            next_done = torch.Tensor(
                np.logical_or(_to_np(terminated), _to_np(truncated)).astype(float)
```
Replace with:
```python
            next_obs, reward, terminated, truncated, info = envs.step(_action_for_env(action, jax_on_gpu))
            reward = torch.tensor(_to_np(reward), dtype=torch.float32).to(device)
            rewards_buf[step] = reward
            terminated_buf[step] = torch.Tensor(_to_np(terminated).astype(float)).to(device)
            next_obs = torch.Tensor(_to_np(next_obs)).to(device)
            next_done = torch.Tensor(
                np.logical_or(_to_np(terminated), _to_np(truncated)).astype(float)
```

**Change 3** — Use `terminated_buf` (not `dones_buf`) for GAE `nextnonterminal`.

Find this block in the GAE computation:
```python
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
```

Within this loop, find and replace ONLY the `nextnonterminal` computation in the `else` branch:
```python
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
```
Replace with:
```python
                    nextnonterminal = 1.0 - terminated_buf[t + 1]
                    nextvalues = values_buf[t + 1]
```

Also find the `if t == args.num_steps - 1:` branch (which uses `next_done` for the last step).
That branch likely has:
```python
                    nextnonterminal = 1.0 - next_done
```
This is correct as-is (next_done for the final step uses `terminated | truncated` for the env's
autoreset sync, which is fine — the terminal boundary handling only matters for mid-rollout steps).
Do NOT change that line.

**After code changes**, create `configs/exp_033_trunc_fix.yaml`. Fine-tune from best checkpoint so far.

```yaml
name: exp_033_trunc_fix
backend: racing
hypothesis: "Fix truncation vs termination conflation in GAE (Pardo et al. ICML 2018 arXiv:1712.00378).
  Current code uses dones=terminated|truncated for nextnonterminal, which zeroes bootstrap at episode
  timeouts. Correct fix: use terminated only for nextnonterminal. Timeout at max_episode_steps=1500
  is NOT a crash — value function should bootstrap through it. This improves value estimates near
  episode boundaries and should reduce conservative near-end-of-episode behavior. Combined with best
  reward config from exp_030/031/032."
budget_seconds: 5400
racing:
  env_type: race
  level: level2
  race_config: level2_attitude.toml
  pretrained_ckpt: /workspace/drone-rl-lab/results/exp_032_pbrs_progress/model.ckpt
  total_timesteps: 8000000
  num_envs: 512
  num_steps: 8
  learning_rate: 0.0001
  anneal_lr: true
  gamma: 0.97
  gae_lambda: 0.97
  update_epochs: 10
  num_minibatches: 8
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  max_grad_norm: 1.5
  n_obs: 2
  cuda: true
  seed: 42
  gate_bonus: 20.0
  proximity_coef: 1.0
  progress_coef: 50.0
  speed_coef: 0.3
  rpy_coef: 0.06
  act_coef: 0.02
  d_act_th_coef: 0.4
  d_act_xy_coef: 1.0
  max_episode_steps: 1500
  alt_coef: 1.5
  survive_coef: 0.5
  vz_coef: 0.5
  vz_threshold: 0.5
  oob_coef: 8.0
  z_low: -0.05
  z_high: 1.3
  random_gate_start: false
```
