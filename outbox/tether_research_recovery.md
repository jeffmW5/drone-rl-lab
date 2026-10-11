# Numerical failure and recovery diagnostic

2026-10-11 heartbeat: longer-training job had a dead Process-3 (PID19466),
MuJoCo warning `Nan, Inf or huge value in QACC at DOF6`, simulated time55.104s,
and FloatingPointError. Parent19414 kept waiting with stale progress237568.
Stopped only its verified process group; retained every checkpoint and log.
Last saved pair checkpoint_200704; no final evaluation. Marked invalid/failed,
no promotion. Scope: one stochastic rollout; exact seed/domain and physical
cause unavailable, so finer integration is a hypothesis rather than a diagnosis.
Evidence: tether_sim/evidence/research002_numerical_failure.json and runtime
tmp/tether-research/log_tether_research_002_more_steps.txt.

Added worker watchdog: unexpected worker exit raises in the parent, saves
interrupted artifacts and terminates its own children. Research runner stops
at failed trials rather than skip them. Optional numerical-failure capture
saves pre-step qpos/qvel, motor state, wind, residual action, domain and physics;
invalid transitions still raise, never silently reset or count as successes.
Twelve harness/adapter/watchdog checks passed, including real failed child,
failure serialization, 1ms physics /20ms control clock and duplicate lease.

Frozen recovery config configs/tether_research_recovery_v001.json resumes
checkpoint_200704 with integration1ms (previous2ms), 100000 transitions or600
training seconds plus3-seed validation. This changes the physics benchmark;
compare its initial/final results within this version only, no promotion
against the original2ms campaign. New failures capture their pre-step state.
Reserved audit seeds untouched. Recovery is the third research training launch
in the rolling24h window; defer the original long-horizon trial and new training
until the limit allows. Review recovery evidence before preparing campaignv002.
