# Real Flight Deployment — Status (2026-04-19)

## What exists

Three files in `/media/drone-rl-lab/real_flight/`:

- **fly.py** — Main deployment script. Uses cflib directly (no ROS2/Vicon).
  Three modes: `check` (radio/battery/positioning), `hover` (PID only), `fly` (RL policy).
  Run with: `python3.11 real_flight/fly.py check|hover|fly`

- **config.yaml** — All configuration: radio URI (`radio://0/80/2M`), level2 gate/obstacle
  positions (from official LSY GitHub level2.toml), cf21B_500 drone params, geofence, safety limits.

- **analyze.py** — Post-flight log visualization. Run with: `python3.11 real_flight/analyze.py real_flight/logs/flight_*.npz`

## What was installed

- cflib 0.1.32 on Python 3.11 (`python3.11 -m pip install --user cflib`)
- torch 2.10.0+cpu was already on 3.11

## Architecture decisions

- **cflib-only** instead of the existing `RealRaceCoreEnv` (which requires ROS2 + Vicon motion capture).
  This lets us fly with just Lighthouse V2 or Flow Deck positioning.
- **Auto-detect policy architecture** from checkpoint weight shapes (obs_dim, hidden_size, act_dim).
  No need to manually specify — works with any experiment checkpoint.
- **Safety-first flight sequence**: PID takeoff → hover stabilization → switch to RL policy.
  Ctrl+C → safe landing. Geofence/battery violations → emergency stop.
- State estimation via cflib logging framework (stateEstimate.x/y/z, stabilizer.qx/qy/qz/qw,
  gyro.x/y/z) at 100Hz. Policy runs at 50Hz.

## Known issue: observation format mismatch

The checkpoint (exp_069) expects **55D** input but the standard race env obs dict
(pos/quat/vel/ang_vel/target_gate/gates_pos/gates_quat/gates_visited/obstacles_pos/obstacles_visited)
flattens to **62D**. The 7D difference comes from `make_race_envs` in
`lsy_drone_racing/control/train_race.py` which only exists on the **RunPod GPU training server**,
not on this VM.

**Current workaround**: auto-truncate the 62D obs to 55D. The first 55 dims contain all the
drone state + gate info which are the most important features.

**To fix properly**: Copy `train_race.py` from RunPod to this VM (or the lsy_drone_racing
shared folder at `/media/lsy_drone_racing/`). Then read `make_race_envs` to determine the
exact observation wrapper order. The function is imported as:
`from lsy_drone_racing.control.train_race import make_race_envs`

Alternatively, run on RunPod:
```python
envs = make_race_envs(config="level2_attitude.toml", num_envs=1, ...)
obs, _ = envs.reset()
print(obs.shape, envs.single_observation_space)
```
and record the exact obs_dim and key ordering.

## What still needs doing

1. **VirtualBox USB passthrough** for Crazyradio PA dongle:
   VBox Manager → Settings → USB → Enable USB 3.0 → Add Filter → "Bitcraze Crazyradio PA"

2. **Fix obs format** (see above) — copy train_race.py from RunPod or print obs space there

3. **Progressive flight testing**:
   - `python3.11 real_flight/fly.py check` — verify connection, battery, positioning lock
   - `python3.11 real_flight/fly.py hover` — PID hover, no RL
   - `python3.11 real_flight/fly.py fly --no-gates` — RL policy without gate tracking
   - `python3.11 real_flight/fly.py fly` — full RL with gate detection

4. **Measure and update gate positions** in config.yaml to match your physical gate layout
   (currently using nominal level2.toml positions)

5. **Sim-to-real tuning** — once flying, compare flight logs against sim benchmarks using
   analyze.py. Key gaps to watch: thrust scaling, attitude response lag, position drift.

## File locations

| What | Where |
|------|-------|
| Deployment scripts | `/media/drone-rl-lab/real_flight/` |
| Training configs | `/media/drone-rl-lab/configs/exp_*.yaml` |
| Trained checkpoints | `/media/drone-rl-lab/results/exp_*/model.ckpt` |
| Best checkpoint so far | `results/exp_069_larger_network/model.ckpt` (128 hidden, 55D obs) |
| Drone model params | `/home/jeff/.local/lib/python3.11/site-packages/drone_models/data/params.toml` (cf21B_500) |
| cfclient radio config | `/home/jeff/.config/cfclient/config.json` |
| Existing real env (needs ROS2) | `/home/jeff/.local/lib/python3.11/site-packages/lsy_drone_racing/envs/real_race_env.py` |
| Level2 config (created) | `/home/jeff/.local/lib/python3.11/site-packages/config/level2_attitude.toml` |
