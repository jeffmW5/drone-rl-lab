# Methodology review 0013: gradient bound partial review

Last reviewed trial index:112. Trials113-115 still pending. Preserve the healthy
campaign and wait for matched three-seed outcomes before another hypothesis.

Seed19001 final error8983.163mm atbound0.5 versus8802.196mm atbound5; both0crashes,
24takeoffs0hover and selectedbest_step0. Seed19002 bound0.5 final6845.122mm,
0crashes24takeoffs0hover; matching bound5 not yet complete. These are worse than
initial1080.487mm. One small paired error reduction does not establish improvement.
Source native trials110-112/result.json and gradient_diagnostics.json; compact
evidence cuda_gradient_bound_partial_v001.json. Process healthy, not paused.

New literature read: Implementation Matters in Deep Policy Gradients: A Case Study
on PPO and TRPO2005.12729v1 section3, https://arxiv.org/html/2005.12729 . HF metadata
and markdown returned404; original HTML used. It explicitly distinguishes global
gradient clipping from policy-ratio clipping and identifies reward scaling,
initialization and normalization as implementation choices worth controlled tests.
Its reported four-factor ablation did not isolate every listed optimization;
do not cite it as proof that changing our bound from0.5 to5 will help.
Author code https://github.com/MadryLab/implementation-matters (not executed).
Prior Learning to Fly in Seconds2311.13081v2 IV/V
https://arxiv.org/html/2311.13081v2 motivates curriculum/actuator observations but
uses TD3 and a different simulator. Both task/algorithm differences remain explicit.

After complete results, if bound5 cannot promote a learned policy, stop further
clipping sweeps. A future diagnostic should measure rollout credit horizon and
observation/return scale before modifying them. Current instrumentation observes
combined gradients only; no evidence proves value-loss dominance. Original strict
1mm benchmark, hardware/audit seeds and fidelity limitations remain separate.
No new trial, compiler, pause or restart issued by this review.
