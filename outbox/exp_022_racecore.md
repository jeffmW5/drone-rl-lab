# exp_022_racecore — Results

**Backend:** racing (lsy_drone_racing)
**Hypothesis:** Train directly on VecDroneRaceEnv (MuJoCo gate-racing env) with dense gate-proximity reward. Eliminates crazyflow→MuJoCo physics gap and trajectory-following ceiling. First experiment with the new paradigm.

| Metric | Value |
|--------|-------|
| mean_reward | 6.336 +/- 2.598 |
| timesteps_trained | 1,933,312 |
| wall_time | 1800.5s |
| level | level2 |

*(Linux Claude: write full analysis to `results/exp_022_racecore/EXPERIMENT.md`, then update this file.)*
