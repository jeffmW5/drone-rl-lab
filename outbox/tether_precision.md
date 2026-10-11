# Millimeter destination target

2026-10-11: User requested millimeter accuracy. New frozen experiment
configs/tether_puffer_precision_v001.json uses NavigationPrecisionTask,
90-second episodes and the existing geometry-aware randomized simulator.
Pass: full 3D payload error <=1 mm continuously for 3 seconds, speed <=5 mm/s,
completed route, no crash. A single close sample cannot pass. Smooth logarithmic
destination shaping adds incentive at centimeter and millimeter scales. This is
an uncalibrated privileged-state simulation objective, not hardware accuracy.
Existing V2 task/config/results stay unchanged. Reward and success rates are not
directly comparable across versions. Eight environments/four CPU workers;
PufferLib policy, learning and advantage calculation on CUDA. Resume previous
best checkpoint with fixed observation/reward statistics; 200000 new transitions
or 1200 training seconds, plus evaluation. Validation seeds 51001-51003;
20 held-out test seeds 61001-61020. Best checkpoint ranks crashes, strict successes,
then mean final payload destination error in mm.

Verification: precision residence boundary/reset, motion/collision/unfinished
route/NaN rejection and real task shape/metric integration: 3 tests passed.
Existing four adapter/kernel checks passed alongside two initial residence tests.
Precision training queued at commit; progress/results belong in run status and
evaluation JSON, not inferred from these tests.

Launch verified after commit c0f049a: run tether_puffer_precision_v001, PID 11033,
GTX 1070 Ti, compiled CUDA advantages true, source best policy 200704 steps.
Status evaluating initial transfer, zero new transitions at this observation.
Actual resume_run/provenance fields identify v003 best; the inherited base note
mentioning V007 describes ancestry rather than the immediate source checkpoint.

Previous v003 final evaluation (tmp/tether-runs/tether_puffer_cuda_v003/
evaluation_final.json, one 20-seed test): PPO 5/20 success vs PD 3/20, both zero
crashes; mean payload tracking RMSE 0.158266 m vs 0.158572 m. Modest success
improvement with almost unchanged tracking error; no millimeter performance.
