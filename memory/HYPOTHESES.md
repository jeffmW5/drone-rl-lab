# Hypotheses

> Explanations we are actively testing. These are not facts.

## HYP-001
- **Statement:** A real stochastic-to-deterministic deployment gap exists in the direct-racing line.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` policies around `exp_056` to `exp_062`
- **Supported by:** `exp_056`, `exp_060`, local `exp_061`, local `exp_062`
- **Counterevidence:** Stochastic and temperature-scaled deployment still fail to produce reliable gates.
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Show a deterministic evaluation during training that tracks benchmark while stochastic deployment adds no benefit.

## HYP-002
- **Statement:** The direct-racing line is still materially undertrained, and mean-policy convergence has not completed by the current 3600s budget.
- **Type:** hypothesis
- **Scope:** GPU direct-racing runs with current 3600-7200s budgets
- **Supported by:** `exp_060` reward still climbing at budget end; literature comparison in `research/stochastic_deployment_gap.md`
- **Counterevidence:** `exp_068` doubled budget (7200s) to 42.84 reward but deterministic benchmark was flat (1.67s, 0 gates vs exp_067's 1.70s, 0 gates). Training reward gains did NOT translate to benchmark improvement.
- **Confidence:** low (weakened by exp_068)
- **Last reviewed:** 2026-03-28
- **Next falsification test:** If exp_069 (larger network, same budget) shows benchmark gates, then capacity was the bottleneck, not training duration.

## HYP-003
- **Statement:** Asymmetric critic improves training efficiency in this family, but by itself is insufficient to produce matched mid-air benchmark gains.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` line with privileged critic observations
- **Supported by:** `exp_059` training vs `exp_056` baseline; literature on asymmetric actor-critic
- **Counterevidence:** `exp_059` still achieved 0 gates on the matched mid-air benchmark
- **Confidence:** medium
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Combine asymmetric critic with a separate mean-policy stabilization change and show that benchmark performance then moves.

## HYP-004
- **Statement:** Tight logstd helps late deterministic deployment but may hurt early exploration and attribution if applied from the start.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` runs using `max_logstd`
- **Supported by:** `exp_046` as best deterministic benchmark reference; stochastic and temperature-scaled follow-ups suggest wider distributions still contain useful trajectories
- **Counterevidence:** none cleanly isolated yet
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Compare clamped vs unclamped training with matched budgets, checkpointing, and deployment evaluation.

## HYP-005
- **Statement:** Body-frame gate observations are still promising, but their repo evidence is confounded by simultaneous reward-strength changes.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv`, body-frame observation experiments
- **Supported by:** `exp_057` had one gate passage but changed both observations and `progress_coef`
- **Counterevidence:** `exp_060` combining body-frame observations with strong progress still failed on the benchmark
- **Confidence:** medium
- **Last reviewed:** 2026-03-27
- **Next falsification test:** Run a clean ablation where body-frame observations are toggled and reward weights are held fixed.

## HYP-006
- **Statement:** The 2×64 MLP lacks capacity for the deterministic mean to represent precise gate navigation, causing the persistent deployment gap.
- **Type:** hypothesis
- **Scope:** Direct `RaceCoreEnv` policies with 55D obs and 4D action
- **Supported by:** Swift (Nature 2023) uses 2×128 and achieves real-world gate navigation; our 2×64 (16K params) may be insufficient for 55D→4D mapping with the precision needed for gates; stochastic policy navigates (reward 43) but mean does not
- **Counterevidence:** Improvement is modest (2/15 det gates, 7/45 total); 87% deterministic failure rate remains. Average flight time is shorter (0.86s vs 1.67s), suggesting the larger network is more aggressive but not more stable.
- **Confidence:** medium-high (supported by exp_069 but not fully resolved)
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Even larger network (2×256) or longer training (14400s with 2×128) to see if the capacity trend continues. If 2×256 shows no further improvement, capacity alone is insufficient.

## HYP-AIDECK-RATE — capture half CONFIRMED (refined), promoted to FACT-026
- **Statement:** The 136ms mean inter-frame interval measured in `session_20260828_211353` is set by onboard GAP8 capture + JPEG encode, not by the WiFi link or the ESP32.
- **Resolution (capture half):** Directly measured on-chip via `pi_perf` (FACT-026): capture alone, with the same per-frame stop/start pattern `wifi-img-streamer.c` actually uses, is 69.408ms ± 0.02ms — 95.1% of the documented 73ms figure. Confirmed, not just convergent-by-inference anymore. Refinement: ~44ms of that 69ms is camera stop/start resync overhead, not pixel transfer — a continuous-streaming capture loop (never stopping between frames) measures only ~25ms. This wasn't visible in the original end-to-end split and is new, actionable information for the vision-hover capture-loop design.
- **Not yet confirmed:** the JPEG-encode half (~58ms) — not independently re-measured this session, only the capture half was instrumented.
- **Confidence:** high (capture half).
- **Last reviewed:** 2026-08-29 — see `real_flight/GAP8_PERF_RESULT.md`.

## HYP-GAP8-DORY-CRASH — CONFIRMED, promoted to FACT-023
- **Statement:** The `dorytest` custom `simple_cnn` network's hardware crash (FACT-022) was caused by a bug specific to that non-stock network/build, not a defect in the `gap_sdk`/DORY/JTAG toolchain.
- **Resolution:** DroNet (a real DORY stock `PULP.GAP8` example) built, flashed, and ran to completion on the identical `gap_sdk` install, OpenOCD, and physical board — all 15 layers, final checksum OK (FACT-023). Confirmed: the toolchain was never the problem. `simple_cnn`'s specific crash cause remains undiagnosed (not blocking — DroNet answers Track B's actual question) — see `real_flight/GAP8_DORY_RESULT.md`.
- **Last reviewed:** 2026-08-29

## HYP-MARKER-OTSU
- **Statement:** `marker_detector.detect_marker` fails to see a small printed marker not because of its area/squareness filters but because of global Otsu thresholding. When the black square is below roughly 0.5% of frame pixels, Otsu splits the histogram between the white page and the wall behind it, leaving the square on the same side as the wall where it never forms its own contour. Above that fraction Otsu splits on the square and detection works.
- **Type:** hypothesis
- **Status:** supported by simulation only; **not verified on hardware**
- **Evidence:** rendered `marker_{1,2,3,4}in.pdf` at 300dpi, composited onto a grey (200) wall at a simulated 87 deg FOV, 0.5m. Measured Otsu threshold and black fraction: 1in 0.10%/t=227 miss, 2in 0.37%/t=226 miss, 3in 0.79%/t=8 hit, 4in 1.54%/t=85 hit. Detected size at 1-2in was ~83-139px, i.e. the page outline, not the square.
- **Predicts:** relaxing `min_area_frac` or `min_squareness` will NOT improve detection of a small marker, because the square is not being rejected by a filter — it is not being segmented. Predicts a 4in marker detects reliably where a 1in one does not, on the same hardware and lighting.
- **Counterevidence sought:** adaptive thresholding (mean-C, blocks 31/61/101, both polarities, plus an 0.80 bbox-fill filter) was tried as the principled fix and did **not** reliably beat Otsu — it still locked onto the page at 0.3m. So "global thresholding is the whole story" is too strong; page-vs-marker competition persists under a local threshold too.
- **Scope caveat:** the 87 deg FOV is assumed, not measured — the AI Deck's actual lens FOV is unknown to this analysis, and the crossover fraction will shift with it. All numbers are simulation.
- **Next falsification test:** at the bench, print the 4in and the 1in, and try both at the same distance and lighting. If the 1in detects reliably, this is wrong. If the 4in fails too, the problem is elsewhere (lens, focus, exposure) and the marker size is a red herring.
