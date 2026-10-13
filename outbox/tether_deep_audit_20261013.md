# Drone research deep audit — 13 October 2026

We found a real learner boundary defect and demonstrated that the current policies have weak attitude recovery. There is still no reliable takeoff-to-hover result.

## Tests completed

| Test | Result | What it establishes |
|---|---|---|
| 12 CPU production checks | All passed after repairing one stale test fixture | Reward signs/bounds, reset geometry, motor initialization, checkpoint restoration and diagnostic integrity |
| Exact installed advantage kernel | Timing, terminal bootstrap-stop and gradient signs passed | No simple global action/reward or ascent-sign inversion |
| Final rollout-cell invariant | Original learner gives uncredited cells nonzero gradients; masked learner matches independent equations exactly | A concrete loss/normalization defect; 3.125% of sampled cells in the captured update |
| 25 CPU translated-physics flights | Five saved GPU ground-failure times reproduced; frame differences 0.23–4.18 mm | Useful diagnostic replay, not calibrated physical truth |
| 24 production GPU recovery cases | All eventually failed; signed angle/rate disturbances failed rapidly | Recovery remains poor beyond the launch phase |
| Validation reset comparison | Three seeds × eight agents have identical physical starting states | The nominal 24 checks represent one deterministic condition |
| Failure/timeout overlap | Real failure on step 500 is also marked truncated | A separate reporting corner case; current 250-step evaluation is unaffected |

## What the GPU recovery test showed

Two fixed learned policies were tested with the full tether, unchanged physics, and no helper control. A 10° roll or pitch disturbance led to zone escape after approximately 1.7 seconds. A ±1 rad/s roll-rate disturbance led to escape after 1.12 seconds. Recovery from actually visited slow and fast flight states also failed.

An ideal airborne start briefly held the hover criterion for 1.70–1.76 seconds before failing. Its position and motor forces were initialized artificially; this does not demonstrate learned takeoff and hover. Ground starts still failed after 4.42–5.88 seconds.

Motor mixing and thrust/torque signs passed. The learned feedback around hover is weak; a local derivative analysis estimates only about 0.0028/s roll-rate damping for checkpoint 759. That estimate is an approximation, not a full tether stability proof.

## Changes and controlled experiments

The opt-in learner correction excludes the final rollout column from advantage normalization and actor/value/entropy losses, while retaining that column as a bootstrap value. It uses the exact audited learner source and leaves the installed package and default learner intact.

A serial 2×2 campaign compares the original and corrected learner at 65,536 and 262,144 transitions, across three training seeds. Other policy, reward and physics settings are fixed. Longer trials retain optimizer and environment state inside one run; the 65k prefix must match the corresponding short run before attributing an effect to duration. A 600-second cap may censor a trial, which must be reported.

The old short runs provided only 10.24 seconds of experience per environment—roughly one maximum-length episode—before optimizer and environment restart. Longer runs test that limitation on the current zone task rather than assuming duration solves it.

GPU work remains serial. Healthy training was not interrupted. Reserved audit seeds remain unused. All evidence, rejected outcomes and frozen plans are preserved in methodology review 0105.

## Remaining limits

No reliable takeoff-to-hover result, original strict 1 mm obstacle navigation, or hardware accuracy has been demonstrated. The tether’s material, contact, attachment and shortened-contour model remain uncalibrated. The boundary correction is real; its contribution to hover failure still needs the paired training results.
