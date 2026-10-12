# Methodology review 0010: optimizer-scale partial review

Last reviewed trial index:90. Trials91-93 remain pending at inspection. Preserve
the healthy campaign; do not launch another hypothesis before the three pairs
finish. Initial checkpoint remains selected in the three completed trials.

Seed17001: low LR final8514.799mm,standard LR4438.108mm. Both0 crashes24 takeoffs
0 sustained hover; both worse than initial1080.487mm. Seed17002 low LR final
5628.611mm,1 crash event24 takeoffs0 hover; matched standard arm not yet complete.
Source native tmp/cuda-research/trials88-90/result.json and periodic checkpoints,
compact tether_sim/evidence/cuda_optimizer_partial_v001.json. One final-error
improvement is not a policy promotion or a complete replicated finding.

Implementation inspection: installed PufferLib3 pufferl.py lines398/416-420 form
combined policy plus0.5value loss, then globally clip all policy parameter gradients
to0.5 before optimizer.step. CudaPolicy shares its128x128 body between actor and
value heads. Whether value gradients dominate clipping is unmeasured; retain it
as a diagnostic hypothesis, not a proven cause. Lines253-289 sample Gaussian
actions, save original samples/log probabilities and clip commands to action bounds.
That is an additional saturation diagnostic if wider exploration is reconsidered.
Do not infer these mechanisms from training reward or a single trajectory.

Literature: HF full markdown2311.13081v2 sectionIV critic privileged information
and actor observations https://arxiv.org/html/2311.13081v2 . Earlier primary
2501.18490 tableI and sectionV https://arxiv.org/html/2501.18490 provide PPO settings
but do not demonstrate a gradient imbalance in our code. Neither justifies changing
value-loss weight blindly. These sources motivate examining actor/critic training,
with algorithm and task differences explicit. No author code executed.

Next after campaign completion: aggregate all common and final checkpoints. If
no learned improvement emerges, instrument actor/value/shared-body gradient norms
and action clipping in a versioned diagnostic trial before architecture changes.
Keep diagnostics behavior-preserving and paired deterministically if instrumented;
measure rather than invent a root cause. Physics2 detection remains distinct from
contact response/offset attachment fidelity. Audit seeds and original1mm outcomes
remain untouched. No current new trial, compiler or restart was issued in this review.
