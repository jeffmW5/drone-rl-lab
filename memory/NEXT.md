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
13. ~~**Random gate starts (exp_027)**~~ -- DONE but FAILED benchmark. Reward 11.67 but 0 gates, 1.1s flight — 100% random spawns skipped ground takeoff. Fallback: exp_027b with 50/50 mix (random_gate_ratio=0.5).
13b. ~~**50/50 gate start mix (exp_027b)**~~ -- FAILED. Mid-air envs still dominate reward signal even at 50/50 from scratch. Reward 10.79, 0 gates, 1.18s. Same failure as exp_027.
13c. ~~**Fine-tune exp_026 with random gates (exp_027c)**~~ -- FAILED. Reward 6.34, 0 gates, 3.2s. Hover largely destroyed (3.2s vs 28.8s). Random gate starts are a dead end — all 3 variants (027/027b/027c) failed.
13d. ~~**High speed reward (exp_028)**~~ -- DONE. Reward 16.95, 0.2 avg gates, 0.94s flight. FIRST GATE PASSAGE EVER! But hover destroyed by speed_coef=1.0. Sweet spot is 0.3-0.5.
13e. **Balanced speed reward (exp_029)** -- Fine-tune exp_026 with speed_coef=0.4 + proximity_coef=1.0 + LR=0.0003 + survive_coef=0.5. Balance lateral navigation with hover stability.
14. ~~**Pass at least 1 gate**~~ -- DONE (exp_028, 1 gate in run 4). Now need consistent multi-gate passage.
15. ~~**Add speed incentive**~~ -- DONE (exp_028). speed_coef=1.0 proved navigation is learnable. Tuning needed.
16. **Improve finish rate** -- 20% is not competition-ready. Need >80% to be meaningful.
17. **Improve lap time** -- 13.49s vs target 5.0s. Need gate-aware training to produce competitive times.
