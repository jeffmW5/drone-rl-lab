# Methodology review 0009

Last reviewed trial index:85. Contact campaign79-84 complete; allbest_step0,
24 takeoffs0 sustained hover. Final outcomes:

|Seed|Physics1 mm/crashes|Physics2 mm/crashes|
|---|---:|---:|
|16001|9291.599 /0|9291.599 /0|
|16002|5923.057 /0|5923.057 /0|
|16003|2367.458 /2|2372.039 /2|

Source native tmp/cuda-research/trials79-84/result.json and periodic evaluations,
compact tether_sim/evidence/cuda_segment_batch_review_v001.json. Geometry checks
support the segment detector, not learned improvement. Third-pair difference is
small but preserved as counterevidence. Particle/segment crash events can include
resets. Current takeoff cases do not establish obstacle contact response fidelity.
Reserved audits untouched; no original strict1mm outcomes or hardware claims.

Next diagnosis: policies remain close to their constant initial commands despite
262k-step tests and multiple reward/reset changes. Current LR1e-5 is30x smaller
than the PPO LR3e-4 in2501.18490 tableI, original fallback read at
https://arxiv.org/html/2501.18490 (HF404 here). Learning to Fly in Seconds2311.13081v2
https://arxiv.org/html/2311.13081v2 sectionsIV/V uses TD3 and300k interactions;
it does not establish a PPO learning rate for this simulator. Published settings
motivate an optimizer-scale sanity check, not an expected convergence guarantee.

Hypothesis: tiny update size prevents useful corrective control within our bounded
trials. Queuev006 pairs LR1e-5 and3e-4 with physics2 in both arms,262144steps/600s,
eval_every65536,reward2,controls2,observation98,same initialreference,three matched
seeds17001-17003. No variance override or region reset. Architecture/checkpoint
shapes unchanged; physics version is recorded in each evaluation and source snapshot.
Require no crash/takeoff regression and better sustained hover/paired destination
behavior. Keep all periodic/final failures and compare common checkpoints, not
reward magnitude. Do not keep cycling narrow LR sweeps if this sanity check fails:
inspect actor versus critic gradient contribution and rollout/action semantics.

Centre-offset attachment torque and obstacle contact response remain open fidelity
work; this optimizer test neither supplies nor proves them.
