# Local destination training update -- 2026-10-11

V006 simulator/transfer setup validated: 25 model checks, native .NET integration,
2,048 additional smoke transitions and pendulum timestep check passed. Main V006
launch follows this setup commit. Hardware confirmed: CF2.1 + Flow + AI decks.
Physical fidelity remains uncalibrated. See [V006 report](tether_v006.md).

# Status -- Local pipeline update 2026-10-08

Tether v002 completed 250,880 steps / 510.24 s. Final payload RMSE 0.22857 m
versus PD 0.20472 m; best checkpoint 0.20595 m. No improvement on the three
evaluation seeds; zero crashes for both. Saved-policy playback now reproduces
the best checkpoint's recorded evaluation and supports same-seed PD/best/final
comparison in Flight Studio. See `tether_sim/TRAINING.md`.

First tether training pipeline: [`tether_sim/TRAINING.md`](../tether_sim/TRAINING.md).
The CUDA smoke passed 2,048 transitions with saved checkpoints and paired PD
evaluation. No improvement was established. A first main run is configured
for 250,000 transitions / 900 seconds using bounded motor residual PPO on
the 1070 Ti; consult its live status rather than treating this as completion.

New owner-requested simulator: [`tether_sim`](../tether_sim/README.md), with live
3D/MJPEG Flight Studio, official Bitcraze visual meshes, lift/carry PD baseline,
Gymnasium API and synthetic camera demonstrations. Twelve model checks pass;
baseline evaluation is in `tether_sim/evidence/baseline.json`. This is an
uncalibrated research starting point, not validated SOTA or a real-flight result.

The local GTX 1070 Ti / WSL AI-deck pipeline is now in
[`real_flight/tether`](../real_flight/tether/README.md). See
[`tether_pipeline_2026_10_08.md`](tether_pipeline_2026_10_08.md) for integration
checks and remaining physical validation. Square detection/calibration remains
under `real_flight`. The racing report below is historical as of 2026-04-19;
no new racing training or benchmark was run in this integration.

## Latest Result

**exp_071 -- Observation Normalization** completed 2026-04-19.
- Training: **45.63 mean reward** in **4.63M** steps over the full **7200s** budget
- Benchmark: **not verified** — the current generic benchmark/controller path produced zero parsed runs for `exp_071`
- **Key finding:** observation normalization improved training reward relative to `exp_069`, but no deployment conclusion should be drawn yet

## Current Read On exp_071

- `exp_071` is a clean training-side positive on reward, not yet a racing benchmark result.
- The right status is **mixed / unresolved**, not success or failure.
- Until a parsed benchmark exists, `exp_069` remains the last benchmark-backed positive signal in this direct-racing line.

## Throughput Finding (same pod / same config family)

Measured after `exp_071` on the same RTX 3090 pod:

- `env_samples_per_s`: **803.54**
- `rollout_samples_per_s`: **750.71**
- `policy_samples_per_s`: **11549.66**
- `ppo_update_samples_per_s`: **14443.32**

Interpretation: env stepping still dominates. Low GPU utilization is expected under the current stack because rollout is env-bound, not optimizer-bound.

## Research: Deployment Gap Literature Review (2026-03-28)

Literature review of 10+ papers targeting the stochastic-to-deterministic deployment gap.
See `research/deployment_gap_mean_policy.md` for full details.

**Key finding:** Two critical issues identified in our setup:
1. **No observation normalization** — the "What Matters" study (250K agents, ICLR 2021) identifies this as critical. Our Agent has NONE. Swift normalizes. SimpleFlight normalizes.
2. **No action smoothness penalty** — CAPS (ICRA 2021) and SimpleFlight both show action-difference penalties force the mean to be physically coherent. We have d_act_th/d_act_xy in the reward code but disabled (0.0) since exp_069.

**Theoretical explanation:** Montenegro et al. (ICML 2024) formally prove that PPO optimizes stochastic expected return — the mean is a side effect, not the objective. Without mechanisms to force mean quality (entropy annealing, smoothness penalties, obs normalization), the gap is expected.

## Next Steps

1. Repair or replace the generic benchmark/controller path so `exp_071` can be evaluated with real parsed runs.
2. Keep `exp_069` as the last benchmark-backed comparison point until `exp_071` is actually benchmarked.
3. Continue with `exp_072` if queue momentum matters more than immediate evaluation repair, but keep `exp_071` explicitly unresolved.
4. Use `docs/throughput_findings_2026_04_19.md` as the current reference for rollout bottleneck discussion.

2026-10-08: Added randomized tether navigation and HTML guide. 19 tests pass; CPU V005 smoke saved/evaluated policies. CUDA smoke interrupted by WSL restarts, cause unconfirmed. Known geometry, static horizontal routes, no vision/hardware validation. Main V005 launch follows commit.


2026-10-11: V006 saved at 43,964 steps; V007 CUDA continuation launched on GTX 1070 Ti with eight simulator workers. See tether_v007_cuda.md.
PufferLib3 CUDA migration: four adapter/kernel checks passed; 4096-step CUDA smoke complete, no broad improvement claim. See tether_puffer.md.
Main tether_puffer_cuda_v003 active: 40960 steps at 743.8/s in first training segment; CUDA policy/updates/advantages verified. Native monitor and duplicate guard verified. Broader held-out performance pending.

2026-10-11 update: v003 completed 200704 transitions; final 20 seeds PPO 5 successes, PD 3, both zero crashes. Mean payload RMSE 15.83 vs 15.86 cm. New millimeter precision task/config verified and queued; see tether_precision.md. Prior active snapshot above is superseded.

Local research loop installed for 1070 Ti: configs/tether_research_1070ti_v001.json
queues three adaptive trials; system-Python runner persists state and excludes
duplicates. Five harness checks and three precision checks passed. Thread
heartbeat 1070-ti-drone-research ACTIVE every20 minutes; reviews evidence and
advances one job at a time. First lower-learning-rate trial launched, CUDA
policy/advantage kernel confirmed; initially evaluating transfer. Precision
v001 completed: 0/20 strict successes, 0 crashes, final payload destination
error188.025 mm PPO vs195.346 mm PD. No millimeter performance claim.

2026-10-11 23:09 UTC research review: lower-learning-rate trial completed200704
transitions. Best validation payload destination error114.284 mm vs incumbent
116.890 mm (~2.23% reduction), zero crashes, zero strict successes and3/3 routes
complete on seeds51001-51003. Below5% promotion gate; incumbent retained. Second
trial (400000 transitions /1800 training seconds plus validation) launched from
that incumbent. Reserved audit seeds untouched. Source: state/tether_research/
v001.json and trial eval_200704.json; single small validation set, provisional.

23:29 UTC review: research002 invalid due dead simulator worker /MuJoCo QACC
instability at55.104s; last progress237568, saved checkpoint200704 retained.
Stopped only failed job, incumbent unchanged. Worker watchdog, numerical state
capture and failed-campaign stop added;12 checks passed. Separate1ms recovery
diagnostic prepared (100k transitions /600s), third research launch allowance;
further training deferred. See tether_research_recovery.md.

23:49 UTC:1ms recovery completed102400 steps, no recorded numerical failure;
0/3 strict successes,0 crashes, destination error199.116->194.025mm. Short-run
stability only; selected2ms policy unchanged. Single-seed diagnostics retain
large vertical error after removing wind/motor imbalance; weaker attitude gain
improves height/tilt but fails route completion and is rejected. Record includes
counterevidence and next controller diagnostics. Daily launch allowance returns
2026-10-12 22:48:33 UTC. Reserved audit seeds untouched in this review.

2026-10-12 00:09 UTC: paired PD-only3-seed gain sweep (.015/.012/.010)
completed9 episodes. .012 reduces mean final error211.579->136.424mm with3/3
routes,0 crashes/numerical failures; improvements mainly one domain, no1mm
successes. Learned policy unchanged. Motor clipping on problem seed88.76%->0%.
Next opt-in controller + matched residual-policy evaluation proposed, not yet
trained. Rolling24h launch cap now enforced across recovery/campaign ledgers;
6 harness checks pass. See tether_controller_diagnostic_v001.md.

2026-10-12 controller-policy diagnosticv002 complete:12 deterministic episodes,
1ms model, frozen precisionv001 weights/normalizer. Gain.015->.012 reduces PPO
mean final error202.693->135.478mm (33.16%), all routes complete,0 crashes,
0 strict successes. Two routes slightly worsen; improvement mainly one seed.
PPO lower gain is only0.69% better than PD mean and worse on two routes.
Diagnosticv001 invalid (reset discarded override); preserved and repaired in
new version. No original2ms promotion or training launch. See
tether_policy_controller_diagnostic.md for evidence/scope/next hypothesis.

2026-10-12 user removed daily training-launch cap. Runner and heartbeat updated; six harness checks pass, including readiness after twenty recent launches. One trainer at a time and failed-run review remain. Historical allowance dates no longer apply.

2026-10-12 01:30 UTC: tether_controller_research_001 launched, PID35638, GTX1070Ti CUDA advantages confirmed. Separate1ms/.012 controller benchmark, 100k transitions/600s, learning rate1e-5, source precisionv001 best. Initial transferred evaluation precedes optimization. No duplicate job; original2ms incumbent preserved. New active default planv002. Six harness checks and controller reset/allocation checks passed. WSL provenance reports older HEAD plus dirty tree; owned source changes synced and committed in Windows ca3cde0, not misrepresented as clean WSL HEAD.
