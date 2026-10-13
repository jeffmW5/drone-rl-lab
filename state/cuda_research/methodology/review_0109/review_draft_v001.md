# Review 0109: completed learner audit, terminal evidence and fine distance reward

Last reviewed completed trial index: 879. Date: 2026-10-13 UTC. Review 0108 ended at 855. This is a catch-up review of 856-879, not a claim that intervening three-trial reviews occurred on time. The deep audit overlapped continued serial execution; default continuation trials 862-877 are retained explicitly, including failures. The daemon remains active and unpaused, with one trainer and no restart, duplicate GPU experiment or concurrent build.

## Observations and counterevidence

The 12-arm duration/learner-mask factorial is complete. All requested 65,536 or 262,144 transitions completed within their budget. Each corresponding 65k prefix matches exactly: 10 policy tensors, 29,931 elements, zero difference. All final and periodic checkpoints have zero ground-start one-second sustained hover. The longest temporary hold was 0.56 s at trial 859 step 180224, followed by regression.

| Training seed | Transitions | Original mask: escapes / final peak hold | Corrected mask: escapes / final peak hold |
|---|---:|---:|---:|
| 70401 | 65536 | 0 / 0.36 s | 0 / 0.30 s |
| 70401 | 262144 | 24 / 0.24 s | 24 / 0.06 s |
| 70402 | 65536 | 0 / 0.06 s | 0 / 0.14 s |
| 70402 | 262144 | 0 / 0.36 s | 24 / 0.16 s |
| 70403 | 65536 | 0 / 0.28 s | 0 / 0.30 s |
| 70403 | 262144 | 24 / 0 s | 0 / 0 s |

Each mask's long trials escaped in two of three training seeds. The six short trials were escape-free only within the five-second evaluation. Longer training does not reliably solve the current failure. The independent final-column credit defect is real; its correction has not demonstrated a consistent hover benefit. No learning method is promoted. Three validation seeds times eight lanes repeat one physical starting condition, not 24 independent domains.

The opt-in pre-reset trace passed its actual GPU equivalence test in trials 860/861. Initial fixed-source replay of 850/851 matched every old aggregate, per-run metric and sparse trajectory exactly. Reference hashes were verified. Added records capture finite first failures at 4.34/4.36 s, at X approximately 2.003/2.015 m, with horizontal speeds approximately 1.652/1.625 m/s and tilt approximately 10.30/10.37 degrees. Later near-ground samples are auto-reset states. The observed terminal pitch torque does not brake the outward drift, but does not prove a universal sign defect. The decoded trace has policy-observation precision, not full FP64 solver-state precision. The separate 4096-transition plumbing updates are not equivalence checks or learned progress.

Default weight-continuation trials 862-877 remained on their own observation/reward/mask configuration and did not solve hover. They are not extensions of the fixed-source factorial. Trial 874's small post-reset altitude endpoint is particularly misleading alongside 24 zone escapes. The corrected and superseded catch-up collectors are retained; an earlier Markdown display used the wrong mask key, while its JSON retained the actual configuration. Numeric evaluation-file filtering was also repaired after a failed first collector. These analysis failures were not trainer or physics failures.

The first neutral-warm versus signed-small-tilt reset pair, 878/879, completed under the serial lease, reaching 262144 transitions in 429.22/434.17 s. Both final ground policies have 24 escapes and zero hover. Treatment delays its first escape from 4.44 to 4.90 s relative to control, but both regress from the shared initial extended-ground failure at 5.88 s. Final peak ground holds are 0.12/0.18 s. All 16 recovery cases in each arm eventually fail before 20 s; final signed 2-degree holds remain under one second. Artificial ideal-airborne holds decline from 1.70 s initially to 1.56/1.44 s. Slight ground-survival improvement is not proof of learned angle recovery. Full initial/periodic/final and probe results are retained in the addendum. Do not extrapolate one pair to the remaining two training seeds. The current reward comparison takes priority after this pair; the remaining four frozen recovery arms are retained unchanged.

## Reward diagnosis and requested adaptation

Production reward15 pays raw reward 1.05 at perfect upright stationary hover. The actual learner clips incoming reward to [-1,1]. Tests of the real function show all tested 0, 1, 5 and 10 mm X, Y and altitude errors clipping to 1. Therefore smaller errors in this neighborhood were indistinguishable to the learner. This is a concrete reward/clip mismatch. It does not alone establish the cause of lateral instability.

Reward16 is a separately versioned, pure reward calculation, with positions in metres:

```
s(d) = sqrt(d*d + 0.01**2) - 0.01
D = max(abs(dx), abs(dy))
C = 4*s(z - 0.8) + s(dx) + s(dy) + 2*speed**2
A = clamp((z - 0.05)/0.25, 0, 1)
core = 0.95*A/(1 + C) + 0.05*upright
ground_cost = 0.8*(z < 0.15)
D <= 1: reward = (1 - D)*core - ground_cost
1 < D < 2: reward = -(D - 1) - ground_cost
physical failure or D >= 2: reward = -1
```

The 1 cm smooth-absolute scale and inward taper retain measurable 1 mm/centimetre improvements after clipping. At stationary upright target height, X offsets 0/1/10/50 mm receive 1.000000/0.998953/0.986120/0.914463. Altitude offsets receive 1.000000/0.999811/0.984516/0.866179. Airborne reward is continuous at the 1 m and 2 m square boundaries. Ground reward in the outer band may still clip below -1; no claim that every clipping effect is removed. Speed and upright terms remain; angular-rate penalty stays zero. This dense reward is paid at 50 Hz, with no repeated boundary-entry bonus or hidden controller.

Independent review notes that the square abs/max taper has derivative corners: boundary continuity is C0, not differentiable smoothness everywhere. The helper assumes finite normalized quaternions, as provided by the versioned physics. In a sufficiently fast, inverted state, core can become negative; moving outward within the square can then make this component less negative. This is preserved counterevidence to any global all-attitude monotonicity claim. The current near-upright terminal-drift states differ from that counterexample. Inspect actual visited and pre-reset tilt/speed before any long reward16 campaign; do not dismiss observed exploitation merely because current tests resolve fine distance near hover.

Twelve CPU invariant groups pass, including exact reward14/15 parity against the saved pre-change source, real CudaVector dispatch with CPU fake physics, pre-reset observations/terminal reward, failure/reset/timeout handling, invalid-version rejection, unit/sign symmetry, square boundaries, scalar reference, input/RNG integrity, and strict clipped millimetre/centimetre gradation. The test cannot establish learned stability or calibrated physics.

## Original methods and applicability

The installed official huggingface-papers skill was used. HF search, 2311.13081 metadata and markdown succeeded; 2006.05990's two HF 404s are preserved. Original arXiv methods provide the fallback. Retrieval URLs, exact versions, sections and inspected author-code links are recorded in retrieval.json and original_sources.json. No author code was downloaded and executed.

- Learning to Fly in Seconds, arXiv 2311.13081v2, methods IV and experiments V/VI: direct learned control uses varied initial states and an actuator model. This supports checking recovery coverage and matching actuator assumptions. It does not validate our tether, PPO implementation, reset curriculum or hover outcome. The pending signed 2-degree curriculum isolates initial attitude exposure from the neutral-warm baseline; previous unsuccessful warm-start trials remain counterevidence.
- Learning to Control a Quadrotor through Reinforcement Learning, arXiv 2006.05990v1, sections 3.2-3.5: architecture, initial action distribution, observation normalization and temporal data collection motivate explicit contracts rather than diagnosing physics from a failed training run. Its reported performance is not our simulator's performance. Our actor already observes simulator orientation, angular rates and motor forces; a missing-tilt or missing-actuator-observation explanation is unsupported. A separate action-history or asymmetric-critic change is not bundled into this reward comparison.

The fine-distance adaptation is directly motivated by local production reward evidence and the user's request, not a claim that a paper proves this reward will converge. Material/contact/attachment fidelity remains uncalibrated. Existing leased signed thrust/rotation/slack-gravity checks passed; no additional physics version is justified by these reward/learning failures alone.

## Frozen test and rejection rule

The priority queue methodology_queue_v000_user_abca_fine_reward.json compares reward15/16 across training seeds 70601-70603. Each pair starts from checkpoint 759 best with identical hashes, fresh Adam, ground reset3, observation2, control2, physics5, corrected mask2, 128 parallel environments, batch4096, horizon32, learning rate 3e-6, 65,536 transitions and a 600 s budget. Checkpoints are evaluated initially, every 16,384 transitions and at final, using unchanged 250-step ground evaluation and pre-reset trace1. Recovery probe2 is disabled because its validation contract is reward15-specific.

Only reward_version and identifying labels differ within a pair. The recovery curriculum is a separate frozen campaign and its remaining arms are not edited. The current healthy run drains naturally before queue priority changes. Preserve source snapshots and rejected trials. Reject the reward as a hover improvement without replicated gains in genuine ground sustained hover, or at least motion/hold improvement with no added true failures and no lost takeoffs. Larger reward, a better initial checkpoint or a reset endpoint is insufficient. If short trials show a consistent signal, a separately reviewed duration/robustness test is needed before promotion; do not silently repeat failed coefficient sweeps.

Evidence is grouped under factorial/, terminal_trace/, catchup/, recovery_curriculum/ and fine_reward/. Reserved hardware/audit seeds remain untouched. All reused validation is provisional. Original strict sustained1mm obstacle navigation and real hardware precision remain unachieved.
