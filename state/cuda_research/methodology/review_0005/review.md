# Methodology review 0005

Last reviewed trial index: 51. Sample budget ablation complete: all six requested
step counts were reached. All selected policies remain initial best_step0,
0 crashes24 takeoffs0 hover1080.487mm. No promotion.

|Seed|65536 final mm|262144 final mm|
|---|---:|---:|
|14001|6774.731|5590.326|
|14002|7519.725|1761.109|
|14003|8530.726|8916.855|

All final evaluations0 crashes24 takeoffs0 sustained hover. More training improves
two final means but worsens one and every final remains worse than its initial.
Evidence: native tmp/cuda-research/trials45-50/result.json, periodic checkpoints,
and tether_sim/evidence/cuda_sample_budget_batch_v001.json. Final first-domain
trajectories show motors typically near0.12N with small differentials despite
large drift. This observation does not prove insufficient exploration.

Sources read: HF full markdown Learning to Fly in Seconds2311.13081v2, sectionIV
exploration scheduling and motor-delay discussion, https://arxiv.org/html/2311.13081v2 .
Prior original-source fallback2501.18490 sectionsIV/V:
https://arxiv.org/html/2501.18490 . Neither specifies our Gaussian PPO variance.
Current initial log_std=-4 implies action std0.0183; controls2 collective command
std0.00916 and each differential action contributes std0.000366 before mixing.
Exploration may be too narrow to discover materially different corrective action.
Confidence high for algebra and recorded metrics; low for causal explanation.

Next matched ablation: initial_log_std=-4 vs-2, giving7.389x initial action std.
Identical policy mean weights and deterministic initial evaluation; apply variance
override after loading the same reference checkpoint. Same262144steps/600s,
LR1e-5,reward2,controls2,observation98,no region resets, eval_every65536.
Three paired training seeds15001-15003. This is exploration, not a PID helper or
physics change. Evaluate crashes, takeoffs, sustained hover and paired error,
preserve initial/periodic/final outcomes; reject regressions. Audit seeds reserved.
No claim that paper results or simulation hover establish hardware accuracy.
