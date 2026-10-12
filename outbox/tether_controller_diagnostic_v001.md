# Intermediate attitude gain diagnostic

2026-10-12 00:09 UTC heartbeat: no new training launches; rolling24h allowance
exhausted until2026-10-12 22:48:33 UTC. Default campaign tick remains at failed
trial review. Nine PD-only episodes tested three gains on matched seeds51001-3,
in the frozen recovery1ms model,90s horizon. No learned policy was changed.
Source configs/tether_controller_diagnostic_v001.json, script
scripts/probe_tether_controller_sweep.py and full domain/measurement records
tether_sim/evidence/controller_sweep_v001.json. Small reused validation set;
direct measurements have high confidence, generalization/causality low.

| Attitude gain | Mean final payload error mm | Route completions | Crashes | Numerical failures | 1mm successes |
|---:|---:|---:|---:|---:|---:|
| .015 default | 211.579 | 3/3 | 0 | 0 | 0 |
| .012 | 136.424 | 3/3 | 0 | 0 | 0 |
| .010 | 143.520 | 3/3 | 0 | 0 | 0 |

The .012 diagnostic reduced mean error35.52% against the matched PD baseline.
Most improvement came from seed51001:386.858->162.226mm; motor-command clipping
occurred on88.76% of default steps versus0% with .012. Last10s mean tilt
12.44->1.47deg and vertical error-0.368->+0.052m. Other seeds improved only
slightly in final error and retained roughly-95mm/-142mm vertical biases.
Previous .006 single-seed intervention failed route completion and remains
counterevidence against simply lowering gain as far as possible.

Hypothesis: attitude/motor-lag interaction and clipping contribute in one domain;
steady vertical biases remain a separate target. Gain sweep alone does not
prove the original numerical-instability cause or hardware behavior. No trained
policy promotion, no millimeter accuracy, no use of reserved audit seeds.

Next bounded campaign proposal after the daily allowance returns:
1. Make .012 attitude gain opt-in via versioned config; preserve legacy .015
   defaults and frozen results. Verify the configured controller matches this
   diagnostic and retains observation/action shapes and contact checks.
2. Evaluate the saved residual policy with that controller on these validation
   seeds before training; a policy learned against .015 may compensate poorly
   against .012. Record initial paired PD/PPO metrics on this new benchmark.
3. If stable, train a bounded continuation on the matching1ms/.012 benchmark;
   compare deterministic checkpoints only within that version. Follow with
   steady-bias diagnostics (motor calibration or anti-windup integral feedback)
   rather than assume more optimization will remove controller bias.
4. Keep reserved audit seeds for the final selected policy once; do not use
   those results for search. Replicate promising changes beyond3 seeds before
   stronger claims. This proposal is queued, not implemented or trained yet.

Research runner now enforces the3-launch rolling24h allowance across campaign
and recovery ledgers, including duplicate ledger references;6 harness checks
passed. Existing GPU lease and one-trainer rule remain in place.
