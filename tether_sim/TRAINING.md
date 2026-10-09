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
