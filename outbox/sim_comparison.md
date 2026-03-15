# Sim Comparison — Level 0

All controllers tested on Level 0 (perfect knowledge, no randomization), 5 runs each.
State controller uses `control_mode=state`; attitude-based controllers use `control_mode=attitude`.

| Controller | Type | Avg Time (s) | Best Time (s) | Finished | Gates |
|-----------|------|:------------:|:-------------:|:--------:|:-----:|
| state_controller | Trajectory | 13.860 | 13.86 | 5/5 | 4/4 |
| attitude_controller | PID | 13.376 | 13.36 | 5/5 | 4/4 |
| attitude_rl (theirs) | RL (pre-trained) | 13.336 | 13.32 | 5/5 | 4/4 |
| **attitude_rl_exp010 (ours)** | **RL (exp_010)** | **13.360** | **13.36** | **5/5** | **4/4** |

## Analysis

**Our model can race.** reward=7.36 translates to a fully functional racing controller that completes all 4 gates in every run.

### Performance ranking
1. Their pre-trained RL: 13.336s (fastest)
2. Our exp_010 RL: 13.360s (+0.024s, ~0.2% slower)
3. PID attitude controller: 13.376s
4. State (trajectory) controller: 13.860s (slowest)

### Key findings
- **Our RL is competitive** — within 0.024s of their pre-trained model, faster than PID
- **All controllers finish Level 0** — this track is "solved" by all approaches
- **Near-deterministic behavior** — almost no variance across runs (Level 0 has minimal randomization, only small action/dynamics disturbances)
- **RL beats classical PID** — both RL controllers are faster than the PID attitude controller
- **Attitude > State interface** — attitude controllers (~13.35s) beat the state trajectory controller (13.86s)

### Why is our model slightly slower than theirs?
- Their model was trained with `n_obs=2` (observation history stacking), ours with `n_obs=0` (the `n_obs` config param wasn't passed through in train_racing.py)
- Their model likely trained for more timesteps and/or with more envs (1024 vs our 64)
- Despite these disadvantages, the gap is only 0.024s — our 500k-step, 64-env training on CPU is remarkably close

### Bug found
`train_racing.py` doesn't pass `n_obs` from config to `make_envs()` reward_coefs, so it defaults to 0 instead of the config value of 2. This means our agent has less information per step than theirs.

### Suggested next steps
1. **Fix the n_obs bug** in train_racing.py so `n_obs=2` is passed through
2. **Retrain with n_obs=2** and 1024 envs to match their setup — should close the 0.024s gap
3. **Test on Level 1** (randomized gates) where RL generalization matters more
4. **Level 0 is "solved"** — all approaches complete it, so the interesting comparison is on harder levels
