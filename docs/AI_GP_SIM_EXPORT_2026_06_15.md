# AI-GP Sim Export - 2026-06-15

## Current Artifact

The first usable full-course AI-GP policy is the structured-state MLP from:

```text
results/ai_gp_030_swift_full_course_bc_50m/best_policy.pt
```

Export it for the AI-GP simulator with:

```bash
/home/jeff/drones-venv/bin/python scripts/export_ai_gp_structured_policy.py \
  results/ai_gp_030_swift_full_course_bc_50m/best_policy.pt \
  results/ai_gp_030_swift_full_course_bc_50m/nominal_telemetry.json \
  results/ai_gp_030_swift_full_course_bc_50m/ai_gp_structured_policy.json \
  --randomized-validation-report results/ai_gp_030_swift_full_course_bc_50m/random_1001.json \
  --randomized-validation-report results/ai_gp_030_swift_full_course_bc_50m/random_1002.json \
  --randomized-validation-report results/ai_gp_030_swift_full_course_bc_50m/random_1003.json
```

Use `best_policy.pt`, not `final_policy.pt`. The final checkpoint regressed
after the best evaluation.

## Validation

Best nominal evaluation over 512 episodes:

- mean gates: `6.0`
- success rate: `1.0`
- collision, out-of-bounds, missed-gate, vertical-runaway rates: `0.0`
- gate crossing minimum margin: `0.1647 m`

Randomized evaluations are not yet Swift-level robust:

- seed `1001`: `56.64%` success, `4.73` mean gates
- seed `1002`: `56.45%` success, `4.75` mean gates
- seed `1003`: `56.25%` success, `4.71` mean gates

This is a structured-state simulator pilot and a strong starting point. It is
not a fully generalized vision/live policy.

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
