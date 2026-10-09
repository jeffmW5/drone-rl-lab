# First tether-control model

The first main experiment is `configs/tether_v002.json`: SB3 PPO, four
single-process CPU MuJoCo environments, CUDA updates on the GTX 1070 Ti,
250,000 transitions or 900 training seconds, whichever is reached first.
The budget excludes Python/CUDA startup and final evaluation.

The policy produces four residual motor commands around the privileged PD
baseline. Each correction is bounded to ±0.05 normalized motor command
(±0.01 N). This is a **hybrid residual controller**, not an independent pilot,
camera policy, or real-flight-ready model.

Training observations include simulator qpos/qvel, actual motor forces, target,
wind, time, previous residual actions and gust phase. Observations and rewards
are normalized. Reset randomizes initial horizontal position, small cable
tilt and gust phase. The wind is a 0.012 N sinusoid with 2.5 s period.
Mass, motor and cable parameters are fixed in this feasibility experiment.

The objective penalizes payload/drone target error, swing, residual effort
and changes in residual command. Ground collision terminates with an extra
penalty. The smooth target trajectory is the simulator's lift/carry sequence.

## Start and inspect

Use the existing Pascal-compatible CUDA venv; do not replace its Torch wheel
with an arbitrary newer CUDA build.

```bash
cd /home/n33du/ai/drone-rl-lab
/home/n33du/ai/crazyflie-tether-venv/bin/pip install -r tether_sim/requirements-training.txt
/home/n33du/ai/crazyflie-tether-venv/bin/python -u -m tether_sim.train
```

The output folder must not exist, to protect prior runs. Use `--output` for
a separate attempt. `--smoke` creates a short derived configuration without
changing the frozen source configuration.

Artifacts under `tmp/tether-runs/tether_v002/`:

- `config.json`, `provenance.json`: exact configuration and runtime versions,
  Git revision/dirty status, PID and GPU.
- `status.json`, `progress.csv`: current state, steps, throughput and PPO metrics.
- `checkpoint_*.zip` plus matching `*_normalize.pkl`: every 20,000 steps.
- `eval_*.json`: every 50,000 steps, deterministic policy and PD evaluated on
  the same three held-out seeds. No training occurs during evaluation.
- `best.zip`: lowest evaluated PPO payload RMSE among crash-free PPO
  checkpoints; this name does **not** imply it beats PD.
- `final.zip`, `final_normalize.pkl`, `evaluation_final.json`: final artifacts.

The normalization file is mandatory for inference. Evaluate the policy in
`LiftCarryTask`, whose actions are residuals; feeding them directly to
`TetherEnv` would use the wrong action contract. Checkpoints are local files,
not automatically uploaded. The viewer displays training progress if launched
from the same checkout; its animation still uses PD, not the learned policy.

## Evidence and limits

The first `tether_v001` subprocess smoke reached 256 transitions then ended
abruptly, coincident with Windows allocation error `0xC0000017` and a WSL
restart. The precise root cause is not established. Its configuration and
partial artifacts were preserved. `tether_v002` uses one process to avoid
replicating Torch in rollout workers.

The v002 CUDA smoke completed 2,048 transitions with checkpoints and paired
evaluations. It verifies execution, not learning quality. See
`evidence/training_smoke.json`; no improvement over PD was established.

This small MLP can be faster on CPU; GPU use here satisfies the requested
1070 Ti pipeline, not a speedup claim. See the
[SB3 PPO CPU/GPU guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
Three evaluation seeds and fixed physics are insufficient for broad claims.
Larger held-out sets, parameter randomization and hardware calibration remain
necessary before relying on this controller beyond the current simulator.

## Watch a saved policy

```bash
/home/n33du/ai/crazyflie-tether-venv/bin/python -m tether_sim.viewer \
  --policy-run tmp/tether-runs/tether_v002 --checkpoint best --seed 10001
```

The viewer runs the exact LiftCarryTask observation/action contract, loads
the matching normalization file with updates disabled, and performs deterministic
CPU inference. CPU inference avoids unnecessary CUDA overhead for one live drone.
Use the PD / Best policy / Final policy buttons for a same-seed comparison;
changing controller resets the episode. Only the trained lift/carry scenario
is enabled in policy mode. The sinusoidal training wind remains active; Wind
pulse adds an extra disturbance outside the original evaluation conditions.
The controller label and inference counter show which controller actually runs.

`python -m tether_sim.check_playback --run tmp/tether-runs/tether_v002`
checks the reloaded best policy against its saved evaluation. On this run,
the three-seed RMSE reproduced as 0.205950080967 m versus 0.205950081223 m
recorded, across 1,800 inference steps.

The main run completed 250,880 transitions in 510.24 seconds. Final payload
RMSE was 0.22857 m versus PD's 0.20472 m. Best evaluated PPO RMSE was
0.20595 m at 50,000 steps. Neither beat PD; both controllers had zero crashes
on the three evaluation seeds. These are limited simulator results.

V005: six random physical obstacles, bounded 2D A* routing, coordinate goal commands, randomized dynamics. CPU smoke completed 2048 transitions with checkpoints/evaluation; 19 tests pass. CUDA V003/V004 attempts ended during WSL restarts, cause unconfirmed. Preserved artifacts. Main command: python -m tether_sim.train --config configs/tether_v005.json --output tmp/tether-runs/tether_v005
