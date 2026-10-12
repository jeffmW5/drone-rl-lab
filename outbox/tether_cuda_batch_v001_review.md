# Batch review and controls2 hypothesis

Sources: tether_sim/evidence/cuda_batch_v001_review.json, retained runtime
tmp/cuda-research/cuda_auto_0001..0003_takeoff/result.json and selection.json.
Three reused seeds x8 domains, provisional prototype5s takeoff evaluation.
All three final checkpoints rejected; initial network reference retained.

| Trial | Final crashes | Takeoffs | Hover successes | Final error mm | Parent error mm |
|---|---:|---:|---:|---:|---:|
| 0001 | 8 | 24 | 0 | 5745.132 | 1985.926 |
| 0002 | 0 | 24 | 0 | 10077.873 | 1985.926 |
| 0003 | 0 | 24 | 0 | 3554.714 | 1985.926 |

Trial2 trajectory, first domain: angular speed rises from near zero to
roll0.274/pitch-0.231rad/s around2.5s, drone reaches1.26m, then descends and slides
on ground. Motor imbalances are small but accumulate with low inertia.
No sustained hover. This does not prove that action scaling is the cause.
Ground friction/contact and missing attachment torque remain fidelity limitations.

Next versioned hypothesis: learned collective and limited differential authority
may improve attitude stability. controls2 maps one learned collective channel
to0..1 and three learned roll/pitch/yaw channels through fixed four-motor geometry
with differential amplitude.02. No PD or state-feedback baseline. All four control
channels remain learned. Initialize collective only, differential means zero.
Old direct-motor policies are incompatible; do not resume them or compare their
weight meanings. Separate incumbents keyed phase|controls2. Fresh initial/periodic/
final comparisons use the same new action contract. Rewardv2 and physical backend
unchanged, initial motor state zero, ground takeoff maintained.

Existing healthy child drains before daemon replacement; the continuous old
scheduler may already have started0004. That trial remains controls1 and is not
retroactively reclassified. New controls2 plans explicitly record control_version=2.
No original1mm success, obstacle-route completion or hardware accuracy claim.
