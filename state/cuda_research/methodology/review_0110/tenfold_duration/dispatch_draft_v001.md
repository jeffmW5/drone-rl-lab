# Tenfold training extension - 13 October 2026

The requested longer campaign is active:655360 transitions per run,10times the prior65536, with128 parallel GPU environments. Each run has a30minute wall-clock cap and evaluation every16384 transitions. Optimizer and environment state continue throughout each long run. The long control/treatment pairs share source weights, training seed, physics, ground reset, observation/action versions and learning settings.

The new priority queue is methodology_queue_v000_user_abca_0_tenfold_reward.json, with three reward15/16 seed pairs. The first long control dispatched as trial884. Already-running883 completed naturally; no trainer was interrupted or duplicated. Pending short and recovery plans remain frozen. Actual first65k checkpoint equality must be verified before interpreting a duration effect.

The first short reward15/16 pair881/882 is counterevidence: both produced zero sustained hover, while reward16 added24 zone escapes against0forcontrol. First reward16 escape was at4.78s with horizontal speed2.116m/s and tilt8.56degrees. Its reward is numerically more graduated, but learned improvement has not been demonstrated. Both selected the initial checkpoint; post-reset altitude is not convergence.

This extension implements the user's explicit duration request. It does not promote reward16 or imply duration will fix the failure. Training/evaluation remain uncalibrated full-tether simulation, with provisional reused identical validation and untouched hardware/audit seeds. Review0110 preserves results through882, original methods and the new frozen duration plans. No original strict1mm obstacle navigation or hardware precision result is claimed.
