# Methodology review 0004

Last reviewed trial index: 43. Reward compression did not produce a promotable
policy. Trials36-41 all selected the initial checkpoint (best_step0):0 crashes,
24 takeoffs,0 sustained hover,1080.487mm. Provisional reused-seed results only.

|Training seed|Reward2 final mm|Reward3 final mm|
|---|---:|---:|
|13001|1279.719|3195.448|
|13002|9366.276|9260.749|
|13003|8640.350|8570.790|

All final evaluations have0 crashes24 takeoffs0 hover. Two small error reductions
and one large regression are insufficient for promotion. Full periodic metrics
are preserved in tether_sim/evidence/cuda_reward_batch_review_v001.json; original
trajectories and checkpoints remain in native tmp/cuda-research/trials36-41.
Training reward magnitudes across reward2/3 are not a common leaderboard.
Confidence high for recorded outcomes, low for root-cause attribution.

Sources read: Learning to Fly in Seconds2311.13081v2 sectionV training and IV
methodology, https://arxiv.org/html/2311.13081v2 ; HF full markdown and metadata
retrieved here. Curriculum-based Sample Efficient Reinforcement Learning for
Robust Stabilization of a Quadrotor2501.18490 sectionV/VI sample budgets,
https://arxiv.org/html/2501.18490 (original HTML fallback). The first reports
about300k interactions with TD3; the latter uses millions with PPO. Neither
supports expecting our tether PPO policy to converge in65k transitions. Different
algorithm/task/simulator assumptions prohibit importing their training times.

Next hypothesis: repeated65k restarts reject policies before sufficient learning.
Matched three-seed test, reward2/control2/observation98 and initial reference
unchanged:65536 vs262144 transitions. Both arms have600s elapsed learner cap
including periodic evaluations, eval_every_steps65536. Only requested transitions
differ; actual steps/time must be reported if a budget interrupts a trial.
Keep final outcomes primary for this ablation, compare common64k checkpoint too;
long-arm extra checkpoint selection opportunities are not independent evidence.
No region resets, reward3 or unrelated optimizer changes. Require no extra crashes
or lost takeoffs and improved sustained hover/paired error; preserve failures and
all intermediate regressions. Audit seeds untouched. No hardware/1mm claim.
This tests sample budget, not a new physics or controller-fidelity assertion.
