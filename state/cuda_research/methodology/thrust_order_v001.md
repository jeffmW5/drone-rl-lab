# Thrust orientation order v001

Observed: physics1 computes body-Z thrust axis before quaternion integration, then applies that old axis after rotation updates. This is an explicit integration choice with a one-substep phase difference, not evidence of the sole cause of failed learning.

Physics3 is a copy of physics1 with exactly one dynamics change: compute thrust axis from the updated, normalized quaternion. Physics1/2 and existing configs/results are preserved. No reward, PID, tether, contact, motor-lag or observation changes.

The first physics3 trainer builds its own versioned binary under the existing GPU lease before creating the training environment. Compilation cannot overlap a healthy trainer. It checks mass-weighted total impulse against the updated orientation for positive/negative roll/pitch and yaw, quaternion normalization, and no-rotation equality with physics1. Constraint impulses cancel in total momentum, allowing a full-tether one-step force check. A failed check aborts the trial rather than train on an unverified candidate.

Three matched seed pairs56001–56003 compare physics1/3 with corrected reset3, reward12, landing1 and identical learner settings. Candidate evaluations use physics3; baseline evaluations use physics1, so this is a dynamics comparison, not a common-physics policy ranking. Reused validation seeds51011–51013 remain provisional. Compare crashes, takeoffs, sustained hover and initial/periodic/final destination errors. Audit seeds remain reserved. No improvement or convergence claim before completed results.
