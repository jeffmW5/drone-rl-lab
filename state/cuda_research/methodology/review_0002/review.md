# Methodology review 0002

Last reviewed trial index: 22. No learned controls2 checkpoint has beaten its
initial policy through trial22. Trial22 final evaluation: 8 crashes,24 takeoffs,
0 sustained hover,5467.874mm mean destination error. Evidence in native WSL
tmp/cuda-research/cuda_auto_0022_takeoff/eval_65536.json. Among the first-domain
trajectories from three validation seeds,8/30 saved samples have drone horizontal
distance from goal >1m or drone height >1.6m. This sparse sample count is not a
fraction of rollout time or all domains. Confidence high for saved samples.

Read sources: HF markdown for Learning to Fly in Seconds (2311.13081v2), section
IV methodology, https://arxiv.org/html/2311.13081v2 ; original arXiv fallback for
Curriculum-based Sample Efficient Reinforcement Learning for Robust Stabilization
of a Quadrotor (2501.18490), sections III, IV-B episode truncation,
https://arxiv.org/html/2501.18490 . Prior retrieval0001 records HF404 for the latter.
The first motivates staged training; the second explicitly bounds exploration
episodes. Neither establishes our thresholds or fidelity of our tether simulator.

Hypothesis: a training-only region reset reduces time collecting unrecoverable
flight states and improves learned stability. Experimental region: radius1m
about goalXY, drone ceiling1.6m; reset is truncation, never a physical crash.
Observation98, controls2, reward2, physics and deterministic evaluation unchanged.
Six trials pair baseline and candidate across seeds12001,12002,12003 with identical
initial reference checkpoint,LR1e-5,65536steps/90s. No candidate is resumed into
another replicate; benchmark keys separate conditions. Freeze queued settings.
All evaluations retain ground takeoff and original prototype thresholds. Require
no extra crashes/lost takeoffs and improved sustained hover/paired error; report
all seeds and periodic regressions. No reward-only promotion or hardware claim.
This is a hypothesis test, not a demonstrated methodology improvement.
