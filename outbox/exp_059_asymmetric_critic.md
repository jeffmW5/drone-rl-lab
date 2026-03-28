# exp_059_asymmetric_critic

## What we changed
- Added asymmetric actor-critic for direct racing: actor uses deployable observation dims only, critic gets 28D privileged gate pose state.
- Trained on a fresh RunPod RTX 3090 pod for the normal 3600s budget.
- Benchmarked on the matched `level2_midair` evaluator because this config trains with `random_gate_start: true`.

## What was held constant
- Reward family, level, bilateral progress, spawn settings, and 1-hour budget were held against the intended baseline family (`exp_056`).

## Results
- Training: `mean_reward = 32.502 +/- 1.149`
- Timesteps: `1,974,272`
- Wall time: `3603.3s`
- Mid-air benchmark: `0/5` finishes, `0.0` avg gates, `0.79s` avg flight

## Observations
- Startup instrumentation on the fresh pod made the first compile window visible: first training iteration completed by about `28.6s`.
- The first launch exposed an implementation bug in `AsymmetricAgent`: `std` was being passed into `nn.Linear(...)`. That was fixed before the successful run.
- The existing generic benchmark controller could not load the asymmetric checkpoint correctly until actor-only inference support was added in `lsy_drone_racing/control/attitude_rl_generic.py`.

## Inference
- Within this family, asymmetric critic improved 1-hour training reward relative to `exp_056`.
- It did **not** improve the matched deployment benchmark: still `0` gates and short flights.
- This supports asymmetric critic as a training-side improvement, but not as a sufficient fix for the deployment gap.

## Confidence
- Training result: high
- Mid-air benchmark result: high
- Broader explanation beyond this experiment family: medium

## What this does NOT prove
- It does not prove asymmetric critics are useless for racing.
- It does not prove the deployment gap is purely an inference-time problem.
- It does not verify race-start performance; only the matched mid-air benchmark was run in this turn.

## Next falsification test
- Hold the evaluator fixed and combine asymmetric critic with a training-side mean-stabilization change, then re-run the matched mid-air benchmark.

## Suggested next experiment
- Pair asymmetric critic with entropy annealing or another explicit mean-policy stabilization method instead of treating privileged value estimation as a standalone fix.
