# Status -- Last Updated 2026-03-19

## Last Completed
- **exp_023b** -- RaceCoreEnv with hard OOB termination + grace period fix
- **Training:** Mean reward 6.42 ± 3.04, 4.06M/6M steps (budget: 3600s, z_high=1.8, oob_coef=5.0)
- **Benchmark L2:** 0/5 finishes, 0 gates, avg 0.79s crash time
- **Grace period bug FIXED:** Changed `self.sim.freq` → `self.freq` in race_core.py line 412.
  Now uses env_freq (50Hz) instead of sim_freq (500Hz). Grace = 50//5 = 10 env steps = 0.2s (was 2.0s).
- **Root cause analysis:** Model applies near-max thrust (+0.97 normalized) throughout flight.
  Flies to z=1.8 (training OOB limit) by step 46, but at benchmark continues to z=2.5+ and crashes.
  The model maximizes cumulative proximity reward (~10-15/episode) by flying fast near gates,
  rather than learning to pass gates (gate_bonus=5.0 is too weak relative to proximity reward).
- **Key insight:** gate_bonus (5.0) << cumulative proximity reward (10-15). Model optimizes for
  "fly fast, collect proximity, die at ceiling" instead of "hover carefully, pass gates."

## Currently Training
- **exp_024** -- RaceCoreEnv with rebalanced rewards (pod d54yx9n4s9i9k4)
  - gate_bonus: 5→20 (make gate passage dominant reward signal)
  - proximity_coef: 2→1 (reduce flying-near-gate reward)
  - oob_coef: 5→10, z_high: 1.8→1.5 (tighter altitude enforcement)
  - act_coef: 0.02→0.05 (penalize extreme actions more)
  - budget: 5400s, 8M steps
  - Grace period fix included in training env

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv):** exp_022 -- 0/10 finishes, 0.5 gates avg (altitude control failure)

## Queue Status
- Completed: exp_022, exp_023, exp_023b (all 0 gates at benchmark — altitude control failure)
- Training: exp_024 (reward rebalancing — gate_bonus dominates over proximity)
- **Critical finding:** Grace period bug in race_core.py now fixed (was 2.0s, now 0.2s)
- **Next if exp_024 fails:** Try curriculum (level0 → level2), larger network, or direct altitude reward
