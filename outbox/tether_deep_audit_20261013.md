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
| Failure/timeout overlap | Real failure on step 500 is also marked truncated | An opt-in version now fixes the reporting corner case; current 250-step evaluation is unaffected |

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

An opt-in termination version also makes physical failures take priority over simultaneous timeouts. It passed a mixed-batch production-logic test with GPU visibility disabled. The training factorial retains the old termination version to keep its two factors separate.

## First training comparison

The first short original/corrected pair completed on GPU with zero sustained hover in both arms. Final altitude errors were 59.83 and 56.99 mm; that small difference is not a success claim. The longer original-learner run has passed its 65k prefix: all checkpoint tensors match the short run exactly. Longer outcomes and the remaining training seeds are still pending.

The first 262k control has now completed without hitting its budget cap. Its final policy had 24 zone escapes and zero sustained hover, versus zero escapes at its matched 65k prefix. Longer training alone regressed in this seed. The corrected long arm and remaining seeds continue serially; no factor is promoted.

## Completed first factorial seed and improved failure evidence

Both262144-transition original/corrected learner arms completed their requested steps within the600second budget. Both final policies escaped the zone and produced zero sustained hover. Corrected learning briefly reduced escapes at several intermediate checkpoints, then regressed. Both65k prefixes exactly match their short-run policies. This is one training seed and one repeated physical validation condition; the remaining two seed groups continue. The second seed's original short run briefly achieved0.286mm altitudeerror but only0.14seconds of stable hover. That result does not establish millimeter tracking or sustained flight.

Saved sparse trajectory frames occur after automatic reset and omit the actual escape state. A new opt-in terminal recorder captures the pre-reset policy observation, exact step counters and original failure flags. CPU tests of the actual evaluator function show all existing score/trajectory fields and RNG unchanged, and verify detection of mismatched or tampered reference evidence. The recorder preserves nonfinite evidence explicitly and does not claim fullFP64 solver-state precision.

Two fixed-source GPU reevaluations are queued after the frozen12-armcampaign, using the same250-step benchmark. They will compare every old field against the preserved850/851 final evaluations while adding real terminal records. Those GPU checks remain pending; their4096-transition plumbing updates will not be treated as learned progress. Review0107 preserves the evidence, literature sources, failed outcomes and validation versions. Healthy serial training continues with no restart or concurrent GPU test.
