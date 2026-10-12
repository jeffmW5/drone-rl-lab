# Methodology review 0003

Last reviewed trial index: 33. Region-reset hypothesis rejected provisionally.
All six matched trials completed65536 steps. Selected checkpoints remain initial
best_step0,0 crashes24 takeoffs0 hover1080.487mm. Final means below are outcomes,
not selected-policy improvements (three reused validation seeds,24 domains).

| Training seed | Baseline final mm | Region-reset final mm |
| --- | ---: | ---: |
|12001|8489.831|8425.578|
|12002|3595.424|10205.254|
|12003|7351.711|8838.695|

All final evaluations:0 crashes24 takeoffs0 sustained hover. Source: native WSL
tmp/cuda-research/cuda_auto_0025_takeoff through cuda_auto_0030_takeoff/result.json.
Region reset worsens2/3 seeds and produces no hovering. Do not promote or reuse
its checkpoint as a curriculum success. Trajectories show drift and attitude
departure while collective thrust stays near0.12N/motor. Confidence high for
recorded metrics; explanations remain hypotheses.

Reward diagnostic: installed PufferLib3 pufferl.py line255 hard-clamps received
rewards to[-1,1]. Current reward2 is unbounded below (speed and crash penalties).
Thus e.g.-2 and-20 have identical learner feedback. The amount of saturation in
real training has not been measured; do not claim this is the proven root cause.

Read: Learning to Fly in Seconds,2311.13081v2 sectionIV reward/curriculum:
https://arxiv.org/html/2311.13081v2 . HF full markdown retrieval in this directory.
Curriculum-based Sample Efficient Reinforcement Learning for Robust Stabilization
of a Quadrotor,2501.18490 sectionIV-B reward/truncation:
https://arxiv.org/html/2501.18490 . These support careful reward/task design;
neither prescribes our compression transform. It is a local engineering hypothesis.

Next ablation: reward3=r/(1+abs(r)) applied to the exact reward2 expression before
PufferLib sees it. Strictly monotone instantaneous transform retains differences
below-1 without changing the physical task or evaluation. It changes the cumulative
objective; keep reward versions separate, never compare training return magnitudes.
Six queued trials pair reward2 and reward3 on seeds13001-13003 with identical
initial checkpoint,LR1e-5,controls2,observation98,budget,ground starts. No region
reset. Preserve all checkpoints and failures. Reject extra crashes/lost takeoffs,
require improved sustained hover and paired destination outcomes. Audit seeds
untouched. No hidden PD, physics change, hardware claim or demonstrated improvement.
