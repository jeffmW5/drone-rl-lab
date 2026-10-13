# Review 0108: second matched duration/mask seed

Last reviewed completed trial index:855. Review date2026-10-13UTC. Review0107 ended at852;853-855 are the next three completed trials. Healthy daemon554137 remains unpaused with one GPU trainer; no restart, duplicate, manual GPU probe or compilation was launched. Third-seed factorial trials continue; the opt-in terminal diagnostics remain queued after the frozen12-arm batch.

## Observed results

Seed70402 completes the second2x2 comparison. All four arms retain24 deterministic takeoffs and achieve zero sustained zone-hover successes. The three validation seeds/eight lanes are still one identical physical condition. Both long trials reach262144 transitions within600s (431.737s original,430.577s corrected), so neither is budget-censored. Both masks' short final checkpoint tensors AND complete65k evaluations exactly match their respective long prefixes (10 tensors/29931 elements each, maxdelta0, finite complete/stable/CRC-verified CPU reads). Source hashes and results are retained.

|Training seed|Transitions|Original learner: escapes / peak hold|Corrected learner: escapes / peak hold|
|---|---:|---:|---:|
|70401|65536|0 /0.36s|0 /0.30s|
|70401|262144|24 /0.24s|24 /0.06s|
|70402|65536|0 /0.06s|0 /0.14s|
|70402|262144|0 /0.36s|24 /0.16s|

These are250-step/five-second evaluations, not demonstrated indefinite survival. Every row has zero1s sustained hover. Twenty-four escapes are repeated deterministic behavior in one physical condition, not24 independent domains.

For70402, short original/corrected final altitude errors are9.411/35.795mm, but sampled4.52s speeds are1.037/1.303m/s, above the0.2m/s hover limit; XY extents0.780/0.978m. Original selected32768 checkpoint has0.286mm endpoint altitude error, yet0.14s peak hold and0.664m/s sampled motion. Altitude alone overstates stable precision. Its final9.411mm error and later long65.113mm endpoint retain regression from that intermediate.

Long original854 has no counted escapes at any saved checkpoint; final hold0.36s and sampled speed0.800m/s still fail hover. Corrected855 has24 escapes at every saved checkpoint from147456 onward, ending24 escapes/0.16s hold/535.497mm endpoint altitude error; best_step0. Its245760 last saved near-boundary sample has height0.846m, speed2.045m/s, XY extent1.969m and tilt10.745degrees. Final sparse frames instead show post-reset ground, so neither exact escape state nor time is present. The terminal observer diagnostics will test this directly. No reset endpoint is interpreted as braking or recovery.

Counterevidence: corrected mask had temporary fewer escapes on70401, but is worse on70402. The boundary-loss defect remains established by independent CPU equations; fixing it is not demonstrated to improve flight reliability. Longer duration also has mixed failure outcomes across seeds. Two seeds do not establish the full three-seed effect or justify promotion. No cause is inferred from reward alone, and no simple thrust/gradient sign fault is established.

## Physics checks

The existing per-trial leased GPU invariants passed for853-855 withphysics5: signed roll/pitch/yaw, normalized quaternion, updated-axis impulse and zero-rotation baseline, plus slack free fall. For855 maximum impulse error3.645e-11N*s, zero-rotation difference3.511e-10m and slack freefall error1.124e-12m. These are narrow one-substep regressions; they do not establish full convergence, material/contact calibration or attachment-torque fidelity. Probe4 and later versioned convergence/counterevidence remain in prior reviews; no further physics change is justified by these learning failures alone.

## Original methods read

HF paper search and1707.06347 metadata/markdown succeeded.1712.00378 HF404 responses are preserved; original fallback was read. Retrieval/source URLs, versions, sections and dates are recorded in retrieval.json and original_sources.json.

- [Proximal Policy Optimization Algorithms v2](https://arxiv.org/pdf/1707.06347), sections2.1,3Eq7,5Eq9-12/Algorithm1: clipped sampled-action optimization and combined value loss motivate checking actual credited updates and fixed rollout boundaries. Local exact signed-gradient and boundary tests already exist. This method does not guarantee monotonically better deterministic hover after each checkpoint. Published locomotion/Atari performance is not this tether simulator's performance. HF metadata has no author-code link; no author code was executed.
- [Time Limits in Reinforcement Learning v1](https://arxiv.org/html/1712.00378v1), section3Eq5-6 and4: physical terminations and artificial time limits have different bootstrap meaning; short episode success does not establish indefinite behavior. Initial-state diversity can aid coverage, but previously rejected warm-start/timeout changes are retained counterevidence. [Author project](https://sites.google.com/view/time-limits-in-rl) was identified, not executed. These ideas do not justify changing the frozen factorial's termination, reset or horizon factors mid-batch.

## Decision and falsifiable test

Continue the remaining third-seed arms unchanged. Do not promote the mask as a hovering method or repeat a reward-coefficient sweep. No additional learner/physics change is defensible from these two seeds before the complete factorial and actual first-terminal evidence.

The already-frozen post-batch adaptation from review0107 remains the concrete falsifiable next test: evaluation_trace_version1 captures the decoded98D pre-reset policy observation and exact control/episode step while retaining every old metric/trajectory. The two source850/851 reevaluations verify reference hashes and write equality/counter summaries. CPU production-evaluator and reference-contract tests passed (review0107 terminal_trace v006); GPU equivalence remains pending. Reject/investigate any old-field mismatch before causal interpretation. The4096-transition plumbing updates are separate from learned progress. No duplicate experiment is queued merely to repeat that test.

Evidence: results.json contains853-855 initial/selected/final outcomes and immutable periodic hashes, alongside full physics checks; evaluation_manifest.json records source hashes. trajectory/ preserves an independent analysis and reusable CPU source. prefix_70402.json/source proves matched continuation. Initial improvements and all rejected finals remain accessible. Reserved audit/hardware seeds are untouched; all reused validation is provisional. Original strict1mm obstacle navigation and hardware accuracy remain unachieved in this uncalibrated reduced simulator.
