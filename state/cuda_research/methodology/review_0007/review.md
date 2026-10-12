# Methodology review 0007

Last reviewed trial index: 66. Exploration batch54-59 finished, allbest_step0.
All initial references0crashes24takeoffs0hover1080.487mm. Final outcomes:

|Seed|std log=-4 mm/crashes|std log=-2 mm/crashes|
|---|---:|---:|
|15001|3350.845 /0|9899.313 /0|
|15002|8485.475 /0|4472.607 /18|
|15003|8451.979 /0|7883.784 /0|

Allfinal24takeoffs0sustainedhover. Crash counts are events, not distinct domains.
Reject wider exploration for crash regression and no superior selected policy.
Source native tmp/cuda-research/trials54-59/result.json, all periodic checkpoints
preserved. Do not conflate lower destination error after crash resets with success.

New focus: validate reduced physics before further blind optimizer sweeps.
Hypothesis: node-only collision can miss a thin obstacle between cable particles;
centre attachment can translate the drone without torque under angled cable load.
scripts/check_cuda_fidelity_limits.py tests both on the existing built GPU backend,
under the same GPU lease after a healthy trainer drains. No compiler or duplicate.
This is a counterexample diagnostic, not a hardware-accuracy benchmark. A centre
attachment is internally consistent with a zero offset; actual hardware offset
has not been measured. No missing-offset magnitude may be called calibrated.

Literature retrieval: HF returned404 for2202.09714 and2301.01396; original HTML
fallback read. Differentiable Robotic Manipulation of Deformable Rope-like Objects
Using Compliant Position-based Dynamics2202.09714v1 sectionII-A constraints and
II-B identification https://arxiv.org/html/2202.09714 . It explicitly models
orientation-dependent stretch/shear and bend/twist constraints, beyond this
backend's distance-only particle chain. DiffXPBD2301.01396v3 sections3.1 and4.5
https://arxiv.org/html/2301.01396 treats compliance/time-step dependence and contact
as solver constraints. These motivate fidelity checks, not importing material
parameters or claiming our backend already implements XPBD. Author rigid-body
PDF fetch failed(size/tool error); do not claim that text was read.

Next concrete recovery hypothesis, if the midpoint diagnostic confirms the gap:
add versioned segment contact detection with a conservative finite cable radius,
then validate grazing/midpoint/endpoint cases and benchmark cost under the GPU
lease. Preserve old physics kernel and checkpoint contracts; do not silently
replace training dynamics or claim contact response from detection alone.
No new learned candidate is promoted by these diagnostics. Audit seeds untouched.

Measured diagnostic result: fidelity_limits_v001.json records midpoint intersection
with crash_flag0, while moving the box onto a particle gives crash_flag1. An angled
cable translates the centre0.197181m over0.2s of freefall with exactly zero angular
velocity. The latter is the zero-offset model's behavior, not proof of erroneous
hardware torque. These reproduce the stated limits; no contact or attachment fix
has been implemented in this review. Confidence high for this deterministic check.
