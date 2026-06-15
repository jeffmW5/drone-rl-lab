# AI-GP Pilot Status - 2026-06-15

## Current State

The best current measured-dynamics policy is still not Swift-level and should
not be exported for live control. The best full-course behavior remains the
measured-track structured policy family from runs 023/025/026: it can pass gate
0 in many nominal evaluations, but it does not reliably recover into gate 1 and
does not complete the six-gate AI-GP course.

The training environment has been rebuilt around the real six-gate AI-GP track,
measured command response, real thrust-command mapping, command latency,
drag/rate-response fit, structured active/next gate state, missed-gate
termination, and replayable actuator state. Local verification is green: 37 unit
tests pass.

## What Was Trained Today

RunPod was resumed on an RTX 3090 and used for real PPO training, not local CPU
training. Throughput was about 180k-195k environment steps/sec.

- `ai_gp_028_any_gate_recovery_finetune_100m`: aggressive any-gate replay across
  all six gates with randomized approach state and actuator history. Stopped
  early at about 31.5M steps because full-course eval collapsed to `0.00` mean
  gates after initially reaching only `0.43`, below the 026 baseline.
- `ai_gp_029_any_gate_conservative_finetune_80m`: lower-rate any-gate replay to
  preserve race-start behavior. Stopped early at about 23.6M steps because eval
  trended `0.57 -> 0.55 -> 0.39 -> 0.29 -> 0.15 -> 0.00` mean gates.

The pod was stopped after these runs to avoid idle billing.

## Diagnosis

The blocker is no longer GPU access or insufficient PPO wall time. Plain PPO
with replayed active-gate states is causing catastrophic forgetting of the
launch/gate-0 behavior before it learns a generalized recovery skill. Gate-only
or gate-specific replay is also not the right target; the policy needs a
general active-gate recovery controller that can handle any gate, any approach
state, and actuator history.

## Current Work Direction

The next useful step is to add expert guidance for generalized recovery instead
of continuing blind PPO sweeps. The plan is:

1. Generate or fit expert recovery actions for randomized active-gate states
   under the measured AI-GP dynamics.
2. Train a structured-state controller with behavior cloning or DAgger over all
   gates and approach states.
3. Fine-tune with PPO only after the controller already recovers to arbitrary
   active gates.
4. Evaluate from normal race start and randomized active-gate starts before any
   live-policy export.
