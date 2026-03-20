# What to Try Next

> Prioritized by Windows Claude. Strikethrough = completed.
> Linux Claude: check this for context on WHY the current INBOX task matters.

1. ~~GPU training with 1024 envs + n_obs=2~~ -- DONE (exp_014). Reward 7.29, validates n_obs=2 works with GPU.
2. ~~Train directly on Level 2~~ -- DONE (exp_015). Reward 7.53 at 3M steps.
3. ~~Dynamic trajectory generation~~ -- TESTED, doesn't work at inference time alone (see hard rule #6).
4. ~~Extended training (exp_016, 10M steps)~~ -- DONE. Reward 7.71, converged, 13.49s lap, 2/10 finishes.
5. ~~Retrieve model checkpoints from RunPod~~ -- DONE. Used base64-over-TTY. Pod stopped.
6. ~~Benchmark exp_016 on Level 2 sim~~ -- DONE. 13.49s, 2/10 finishes.
7. ~~Investigate reward shaping for gate passage~~ -- INVESTIGATED. Gate reward impossible via config; requires code changes.
8. ~~Modify RandTrajEnv.reset() for gate-aware trajectories~~ -- DONE (exp_018). Config flag: `gate_aware: true`.
9. ~~GPU gate-aware training (exp_019/020)~~ -- DONE. Both crash at gate 1->2 transition.
10. ~~Improve gate 1->2 trajectory~~ -- DONE (exp_021). Yaw-aware vectors fix it. 2-3 gates on L0, 0-3 on L2. But 0 finishes.
11. ~~Build new training pipeline on RaceCoreEnv~~ -- DONE (exp_022). VecDroneRaceEnv pipeline with dense gate reward. Mean reward 6.34, peak 10.04. Agent passes 1-2 gates. Needs benchmark + extended training.
12. ~~Investigate crazyflow->MuJoCo physics gap~~ -- RESOLVED. No physics gap. Training and benchmark both use MuJoCo via RaceCoreEnv. The 2.02s crash is caused by the 100-step grace period masking OOB during training (model never learns altitude control). Fix: add OOB penalty to reward.
11. ~~Build new training pipeline on RaceCoreEnv~~ -- DONE (exp_022-025b). VecDroneRaceEnv pipeline working. exp_025b achieved altitude awareness (thrust modulation at z~1.1) but momentum overshoots z=1.5 ceiling to z=2.3+.
12. ~~**Fix altitude overshoot (exp_026)**~~ -- DONE. vz_coef=0.5 + z_high=1.3 + gamma=0.97. Reward 9.77, flight time 28.8s, stable hover at z=0.72. Gap is now horizontal navigation, not altitude.
13. **Random gate starts (exp_027)** -- Swift-style initialization from Kaufmann et al. (Nature 2023). Reset agents at random gate with bounded perturbation instead of always gate 0. Forces policy to learn all track segments. Should dramatically improve finish rate and mid/late-gate navigation.
14. **Pass at least 1 gate** -- currently 0 gates in benchmark. Gate passage requires surviving past z=0.7+ without OOB.
15. **Add speed incentive** -- once gate passage works, add reward for fast gate-to-gate transitions.
16. **Improve finish rate** -- 20% is not competition-ready. Need >80% to be meaningful.
17. **Improve lap time** -- 13.49s vs target 5.0s. Need gate-aware training to produce competitive times.
