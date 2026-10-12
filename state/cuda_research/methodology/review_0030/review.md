# Review 0030: rollout counterevidence and input conditioning

Last reviewed trial index: 250. Reviewed 2026-10-12 UTC.

## Completed evidence

The final horizon pair completed. At fixed 16384 batch size, horizon32 versus128 final destination errors (mm) were 1456.238/2297.386, 2202.288/2991.435 and 5553.338/5025.594. All six ran 262144 steps, had zero recorded obstacle crashes, 24/24 takeoffs and zero sustained prototype hovers. Every best_step was zero: all final checkpoints regressed from their common 943.615 mm initial checkpoint. Reject longer rollout as an improvement under this budget and GAE configuration. Longer horizon alone does not extend effective GAE credit when gamma/lambda stay fixed.

Trial242 subsequently selected a genuinely updated periodic checkpoint at step32768: initial857.312 to best822.218 mm, zero crashes,24 takeoffs,zero hovers. Its final checkpoint regressed to1295.322 mm. This is provisional checkpoint selection, not replicated final-policy convergence. Trials243-250 did not displace it. Preserve all rejected outputs. completed_results.json records frozen plans, scores and sampled best trajectory endpoints.

Inspected first-domain sampled trajectories and heights: policies launch but do not hold the payload at the destination; trial242 peak drone heights are roughly1.42-1.71m despite the takeoff payload goal0.2m. Endpoint error and lack of sustained hover remain decisive. Sparse positions do not prove velocity stability or identify a unique optimizer cause.

## Original methods

HF metadata/markdown returned404 for both older papers; retrieval.json preserves the failures. Read original arXiv v1 full methods with cached original text from review0020 as fallback:

- https://arxiv.org/html/2006.05990v1 , section3.3 and appendixB.9: observation normalization affects continuous-control training; value normalization has task-dependent results. Their locomotion evidence does not establish tether-drone performance.
- https://arxiv.org/html/2005.12729v1 , section3: PPO implementation includes observation normalization/clipping and other optimizations. Its main factorial ablation covers the first four listed optimizations, so it does not isolate observation normalization causality.

No author code downloaded or executed. These sources motivate conditioning investigation, not adoption based on paper reward scores.

## Falsifiable next adaptation

Queuev019 pairs raw input conditioning1 against fixed physical-unit conditioning2 on training seeds30001-30003. Both resume the immutable trial242 best checkpoint, use physics1/reward2/controls2, shared critic, LR3e-6,65536steps180seconds,batch4096,horizon32,GAE.95 and identical evaluations. Only input conditioning differs. Divide relative positions by.6m, angular rates by10rad/s, motor forces by.2N; leave velocity,quaternion and goal scales1. These are explicit hypothesis scales, not measured domain variances. This differs from the papers' running mean/variance normalization. It tests whether optimizer conditioning helps, not whether the simulator is calibrated.

Multiply first-layer weight columns by new_scale/old_scale to preserve the initial actor and critic function before learning. Persist scales in checkpoint buffers; restore original scales on a control resume if necessary. No clipping, centering, reward change or PID is introduced. CPU verification passed on the actual incumbent and random shared/separate critics: initial-forward equivalence within2e-5, checkpoint roundtrip, scale reversal and finite gradients. GPU learning results remain pending; no compile or healthy-trainer interruption.

Reject if crashes increase or takeoff/hover regress; require improvements from each arm's own initial checkpoint and consistent paired destination improvements across seeds before promotion. Reused validation51011-51013 is provisional; reserve audit/hardware seeds. Record initial/periodic/final metrics separately and retain failed trials. Current model lacks validated tether bending and attachment torque/contact fidelity; no original strict1mm navigation or hardware accuracy claim.
