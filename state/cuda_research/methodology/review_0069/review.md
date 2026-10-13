# Review0069: reset geometry and thrust integration

Last reviewed completed trial index:590. UTC2026-10-13. Daemon284940 unpaused; corrected-thrust591 training, GPU checks passed. No concurrent build/trainer, interruption or duplicate launch. Initial/periodic/final evaluations remain immutable. This review covers completed corrected-reset comparisons584–589 and baseline thrust trial590; pending physics3 trials are not results.

## Observations

Matched final destination errors(mm), legacy/corrected reset:55001 1632.118374/1851.252573;55002 1167.983797/1499.330572;55003 994.185801/822.295930. Every arm recorded24 takeoffs,0 crashes,0 sustained hover,0 airborne acquisition; best_step0 in all six, initial822.218054mm. Two corrected arms worsened final errors; one improved relative to its control but did not beat the initial checkpoint. Reject a learned-controller improvement claim from reset3. Exact rest-length initialization remains a geometry correction, separately from policy promotion. Evaluation uses legacy starts, so training/evaluation reset mismatch is a limitation.

Compact first-domain payload/drone trajectories and metrics are preserved in paired_reset_results.json. The samples show continuing motion rather than successful target settling; trajectories do not establish one unique learning cause. Full domains/checkpoints remain in native immutable run directories584–590. Baseline590 final2576.058364mm, best0, no hover. Do not compare this single baseline with an unfinished candidate.

Physics3 GPU check591 passed updated-quaternion mass-weighted impulse for positive/negative roll/pitch and yaw. Maximum impulse discrepancy3.643386111161109e-11Ns; zero-rotation position difference with physics1 exactly0 in this test. Evidence: thrust_order_checks.json, native591 frozen source manifests. This proves the tested force-order behavior, not navigation improvement, general convergence or hardware fidelity.

## Literature review and limits

HF read-only search200 is preserved in retrieval.json/source_00.txt. Original author-hosted fallback papers read: XPBD: Position-Based Simulation of Compliant Constrained Dynamics (MiG2016), https://matthias-research.github.io/pages/publications/XPBD.pdf sections3/4, Algorithm1 and equations17–18; Small Steps in Physics Simulation (SCA2019), https://matthias-research.github.io/pages/publications/smallsteps.pdf introduction and substep method discussion. Retrieval date2026-10-13. No author code downloaded/executed. Older author PDFs have no HF metadata/fulltext evidence here; do not substitute search summaries for their methods.

XPBD introduces compliance and accumulated constraint multipliers; its zero-compliance limit reproduces the underlying hard projection. Our solver currently uses bilateral fixed-length links, not calibrated compliant rope. Adding the name XPBD with zero compliance would not fix unilateral slack/contact/attachment deficiencies. Small Steps investigates splitting an interval into smaller integration steps with few constraint iterations; its numerical results concern their solvers and scenes, not this tether. Both support checking convergence instead of trusting one solver budget. Existing diagnostic32 versus128 iterations differ by roughly8cm in the corrected motor-off launch, so numerical convergence remains unresolved.

## Current hypothesis and next decision

The explicit user-requested single-factor physics3 correction is already queued as three seed pairs56001–56003 against physics1, with reset3/reward12/landing1 fixed. Keep that bounded comparison serial and unchanged; no additional reward/learner sweep is justified while it runs. Reject policy promotion if it adds crashes, loses takeoffs/hover or fails replicated paired behavior. Physics-specific evaluation changes mean these are dynamics comparisons, not a common-physics policy leaderboard.

After these pairs, a defensible next diagnostic is full-tether fixed-input convergence at matched20ms control intervals:1ms versus0.5ms substeps,128/256 constraint iterations, unchanged motor/contact configuration. Measure entire carrier/payload trajectories, link errors and failures, not just endpoint reward. This is a proposed diagnostic, not a launched adaptation. Physical reference, ground contact velocity consistency, unilateral rope and off-center attachment torque remain missing. Reused validation seeds provisional; audit seeds unused. Original strict sustained1mm obstacle-navigation and hardware millimeter accuracy unachieved.
