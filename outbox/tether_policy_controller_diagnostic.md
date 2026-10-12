# Frozen-policy controller comparison, 2026-10-12

Completed version2: configs/tether_policy_controller_diagnostic_v002.json and
tether_sim/evidence/policy_controller_probe_v002.json. Twelve deterministic
90-second episodes, validation seeds51001-51003, recovery1ms integration,
selected precisionv001 best weights, frozen observation normalization, CUDA
inference. No optimizer updates or training launches. Policy/normalizer hashes
verified unchanged after evaluation. Audit seeds untouched.

| Attitude gain | PD mean final error mm | PPO mean final error mm | Routes per policy | Crashes | Strict successes |
|---|---:|---:|---:|---:|---:|
| .015 | 211.579 | 202.693 | 3/3 | 0 | 0 |
| .012 | 136.424 | 135.478 | 3/3 | 0 | 0 |

Observation: PPO error falls33.16%, primarily seed51001 (360.323->156.875mm).
Counterevidence: seed51002 worsens103.027->104.193mm and seed51003 worsens
144.730->145.366mm. At lower gain PPO improves only0.69% over PD mean error;
it is worse than PD on two of three routes. Every precision hold remains0s.
The PD lower-gain results reproduce the prior standalone controller sweep.

Version1 is INVALID for the controller comparison: binding before reset was
discarded when reset rebuilt the simulator, so both groups used default gain.
Preserved full artifact: policy_controller_probe_v001_invalid.json. Version2
binds after every reset. No evidence from version1 supports lower-gain efficacy.

Scope/confidence: paired simulation evidence on three reused validation seeds,
provisional and dominated by one domain. Does not establish stability over long
training, generalization, hardware accuracy or millimeter tracking. No promotion
against the original2ms incumbent; integration/control benchmark has changed.

Next hypothesis: opt-in .012 attitude controller may remove saturation and make
residual learning more effective, but persistent vertical bias needs separate
diagnosis. Prepare a versioned1ms benchmark with an unchanged default controller
for legacy experiments, then a bounded campaign with matched PD/PPO comparisons.
No new training before rolling24h allowance returns2026-10-12 22:48:33 UTC.
Original failed longer trial and queued third trial remain preserved.
