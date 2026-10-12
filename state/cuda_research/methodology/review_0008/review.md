# Methodology review 0008

Last reviewed trial index: 76. No new learned hover success or promotion observed
in the continuous default trials. Focus changes to the reproduced segment-contact
gap from review0007 rather than another blind optimizer sweep.

Physicsv002 preservesv001 and its binary. Adds exact finite segment-to-AABB squared
distance via slab-crossing intervals and quadratic minima, tested against cable
radius1.5mm. Contact marks a physical crash; no sliding/friction response is claimed.
No CUDA classes/templates; C11 host API is unchanged. Source-hash guard covers
each selected kernel. Old configs/checkpoints/results are not rewritten.

Measured: scripts/check_cuda_segment_v002.py passes midpoint,near-grazing,miss,
corner-miss,endpoint cases. v001 misses midpoint/grazing;v002 detects them. Both
retain misses and endpoint hits. Collision-free256domain10step trajectory difference
is exactly0m in this check. Six-box synchronized physics-only throughput5713.770
v001 vs5296.373v002 transitions/s, single short measurement(~7.3% lower); excludes
PPO and is not a robust speed ranking. Evidence cuda_segment_checks_v002.json.

Literature: HF2202.09714 returned404; fallback original2202.09714v1 II-A
https://arxiv.org/html/2202.09714 describes distance/orientation constraints and
material identification. DiffXPBD2301.01396v3 4.5
https://arxiv.org/html/2301.01396 describes collision thickness/constraints and
distinguishes detection strategies from response. Our local geometric detector
is not their XPBD solver or a claimed replication. No author code executed.

Next bounded campaign queuev005 pairs physicsv001/v002 across seeds16001-16003.
65536steps/90s,reward2,controls2,observation98,LR1e-5,initialreference unchanged.
Evaluation selects the declared physics version too; benchmark condition keys
separate versions. More detected crashes can reflect stricter physics and must
not be disguised as policy improvement. Compare initial/periodic/final trajectories
and flight metrics, never reward alone. No contact-freeze/hardware accuracy claim.
Centre-offset attachment torque, continuous-time tunneling and contact response
remain unimplemented. Reserve hardware/audit seeds; no original1mm outcome.
