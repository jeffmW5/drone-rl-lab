# Methodology review 0011

Last reviewed trial index:98. Optimizer-scale campaign88-93 complete, allbest_step0.

|Seed|LR1e-5 mm/crashes|LR3e-4 mm/crashes|
|---|---:|---:|
|17001|8514.799 /0|4438.108 /0|
|17002|5628.611 /1|2100.532 /31|
|17003|9089.541 /0|3643.422 /0|

All final24takeoffs0 sustained hover. No finalbeats initial1080.487mm. Higher LR
reduces destination error in three seeds but violates crash non-regression in one;
crash resets also complicate final-error interpretation. Reject promotion. Full
periodic evidence cuda_optimizer_batch_v001.json, original native runs preserved.

New hypothesis to measure, not fix blindly: combined-loss value gradients may
dominate global clipping. Installed PufferLib3 pufferl.py398/416-420 clips all
policy gradients together; our body is shared. gradient_diagnostics.py observes
parameter gradients via hooks before clipping, and actual parameter.grad before
optimizer.step after clipping. It reports actor/log_std, value-head and shared-body
mean/max L2norms. Shared-body gradients cannot be causally attributed to either
loss by these measurements. No gradients, optimizer state or RNG are modified.
The diagnostic source is copied into every instrumented run for reproducibility.

CPU test checks identical parameter updates with hooks on/off and clipping; actual
GPU paired trials are still required to verify behavior across the training stack.
Queuev007 pairs diagnostic off/on across seeds18001-18003,65536steps/180s,
physics2,reward2,controls2,LR1e-5,unchanged initial reference and validation.
Compare exact completed steps and deterministic metrics; instrumented throughput
may differ. Preserve diagnostics even if no imbalance is found. Do not change
value loss or architecture based solely on a gradient-size ratio.

Sources read: HF full2311.13081v2 IV https://arxiv.org/html/2311.13081v2 discusses
different actor/critic information, but uses TD3;2501.18490 V original fallback
https://arxiv.org/html/2501.18490 describes PPO actor/critic structure. Neither
establishes the hypothesized clipping problem here. This is source-inspection-led
measurement informed by literature, not a paper replication or hardware result.
Audit seeds untouched, original strict1mm separate, contact response and attachment
offset remain unresolved. No healthy trainer interrupted or concurrent GPU job.
