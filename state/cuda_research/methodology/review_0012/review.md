# Methodology review 0012

Last reviewed trial index:107. Gradient diagnostics101-106 completed. Paired off/on
final flight metrics exactly identical for all three seeds; allselectedbest_step0.
Final errors8817.919,9670.825,6329.761mm; crash events0,0,1; all24takeoffs0hover.
Evidence cuda_gradient_batch_v001.json and native trials101-106 preserved.

Measured mean L2gradient norms before/after global clipping:

|Seed|Actor|Value head|Shared body|
|---|---|---|---|
|18001|11.534 /0.391|5.614 /0.182|7.033 /0.220|
|18002|12.478 /0.398|6.016 /0.184|6.952 /0.207|
|18003|12.330 /0.412|5.154 /0.170|6.208 /0.198|

192 optimizer steps each. Actor head norm exceeds value head in these averages;
demote the earlier value-head-dominance hypothesis. Shared-body combined gradients
remain unattributed. All groups shrink heavily, but this does not prove clipping
causes learning failure. Adam normalization/epsilon and policy-ratio clipping also
matter. Do not interpret group norm averages as gradient fraction or causal effect.

Concrete next ablation: max_grad_norm0.5 vs5 with otherwise identical262144step/600s
trials,physics2,reward2,controls2,LR1e-5,initialreference and validation. Three paired
seeds19001-19003; diagnostics on in both arms. Only global gradient bound differs.
Require no crash/takeoff regression and improved sustained hover/paired error;
preserve all intermediate and final failures. If unsuccessful, stop tweaking this
clipping parameter and inspect rollout horizon/task curriculum instead.

Sources: HF full2311.13081v2 IV/V https://arxiv.org/html/2311.13081v2 and prior
primary2501.18490 V https://arxiv.org/html/2501.18490 discuss actor/critic learning
and PPO configuration but do not prescribe5 as a correct bound here. The bound is
an explicitly local measured-gradient hypothesis, not paper replication. Source
inspection of installed PufferLib3 establishes global clipping before Adam.step.
No hardware/audit seed use, no original1mm result. Contact response/offset attachment
remain open, despite successful segment-detection diagnostics.
