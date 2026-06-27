# AI-GP Sim Export - 2026-06-15

## Current Artifact

The best current full-course AI-GP policy is the structured-state MLP from:

```text
results/ai_gp_036_weighted_final_approach_ppo_20m/best_policy.pt
```

Export it for the AI-GP simulator with:

```bash
/home/jeff/drones-venv/bin/python scripts/export_ai_gp_structured_policy.py \
  results/ai_gp_036_weighted_final_approach_ppo_20m/best_policy.pt \
  results/ai_gp_036_weighted_final_approach_ppo_20m/evals/nominal.json \
  results/ai_gp_036_weighted_final_approach_ppo_20m/ai_gp_structured_policy.json \
  --randomized-validation-report results/ai_gp_036_weighted_final_approach_ppo_20m/evals/random_1001.json \
  --randomized-validation-report results/ai_gp_036_weighted_final_approach_ppo_20m/evals/random_1002.json \
  --randomized-validation-report results/ai_gp_036_weighted_final_approach_ppo_20m/evals/random_1003.json
```

Keep the original BC artifact at
`results/ai_gp_030_swift_full_course_bc_50m/ai_gp_structured_policy.json` for
comparison. Keep `035` and `031` as previous baselines. Use `best_policy.pt`,
not `final_policy.pt`, for these runs.

## Validation

Best `036` nominal evaluation over 512 episodes:

- mean gates: `6.0`
- success rate: `1.0`
- collision, out-of-bounds, missed-gate, vertical-runaway rates: `0.0`
- gate crossing minimum margin: `0.7019 m`

Best `036` randomized evaluations are improved over `035` but still not
Swift-level robust:

- seed `1001`: `71.29%` success, `5.11` mean gates
- seed `1002`: `68.55%` success, `5.08` mean gates
- seed `1003`: `66.41%` success, `4.96` mean gates

Previous `035` randomized baseline:

- seed `1001`: `69.34%` success, `5.05` mean gates
- seed `1002`: `65.82%` success, `5.00` mean gates
- seed `1003`: `66.41%` success, `4.88` mean gates

This is a structured-state simulator pilot and a strong starting point. It is
not a fully generalized vision/live policy.

For the current June 27 status, target metrics, and bottlenecks, see
`docs/AI_GP_STATUS_REPORT_2026_06_27.md`.

## Follow-Up Training

`ai_gp_031_randomized_full_course_ppo_120m` fine-tuned from the nominal BC
policy and briefly improved randomized success. It was stopped early because
unanchored PPO then collapsed to `0%` randomized success. The saved best
checkpoint is still useful as a previous baseline, but `036` is now the current
export target.

`ai_gp_032_anchored_randomized_ppo_30m` added an actor-anchor penalty to prevent
that collapse. It stayed near the starting policy but did not beat `031`; its
best embedded eval was `55.27%` success with `5.04` mean gates. Do not promote
`032` over `031`.

`ai_gp_035_hard_case_final_approach_ppo_20m` used dense failure telemetry from
`031`, preserved the initial actor as the saved baseline, and focused anchored
PPO on final-approach hard cases. It improves three-seed randomized average
success from `60.55%` to `67.19%`.

`ai_gp_036_weighted_final_approach_ppo_20m` continued from `035` with weighted
near-gate replay over gates 1-5. It improves three-seed randomized average
success from `67.19%` to `68.75%`, improves mean gates from `4.98` to `5.05`,
and is the current structured export target.

## Sim Runtime Contract

The JSON export has `policy_role=structured_state_sim_teacher` and
`observation_contract=structured_teacher_v2`. The AI-GP sim runner must compute
the 26 actor features in the exported order:

1. active gate relative position in body frame, scaled by `30 m`
2. active gate normal in body frame
3. next gate relative position in body frame, scaled by `30 m`
4. next gate normal in body frame
5. body-frame velocity, scaled by `8 m/s`
6. body-frame gravity unit vector
7. body angular rates, scaled by `[3, 3, 2] rad/s`
8. previous normalized action
9. active gate index divided by `gate_count - 1`

Actions are tanh-normalized `[collective_offset, roll_rate, pitch_rate,
yaw_rate]`. The measured AI-GP command mapping is embedded in the artifact:

- thrust center: `0.295`
- thrust span up/down: `0.105 / 0.095`
- max roll/pitch/yaw rates: `0.30 / 0.20 / 0.15 rad/s`
- negative pitch command means forward

The artifact also embeds the measured six-gate NED track, the surrogate FLU
coordinate mapping, inferred upright gate normals, and test vectors for loader
verification.

## Next Training Work

To make this Swift-level rather than nominal-only, continue training with harder
domain randomization and DAgger-style aggregation from simulator failures. Do
not move to camera-only vision until the structured-state policy is robust under
randomized gate pose, dynamics, latency, mass, drag, and spawn conditions.
