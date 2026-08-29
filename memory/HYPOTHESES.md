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

## HYP-AIDECK-RATE
- **Statement:** The 136ms mean inter-frame interval measured in `session_20260828_211353` is set by onboard GAP8 capture + JPEG encode, not by the WiFi link or the ESP32.
- **Type:** hypothesis
- **Supporting evidence:** `real_flight/STATUS.md` documents GAP8 console timings of 73ms capture + 58ms encode = 131ms, which is within 4% of the measured 136ms mean interval. Measured payload rate was 50.5 KiB/s, far below any plausible WiFi ceiling, so the link is not saturated.
- **Why it is not a fact:** the 73/58 split is read from prior documentation, not measured in this session, and the 136ms figure is end-to-end deck-to-PC — it includes CPX framing, WiFi, and host receive. The agreement is convergent, not conclusive.
- **Confidence:** medium-high
- **Falsification test:** instrument the GAP8 to timestamp capture start, encode end, and CPX send, and compare against the host-side interval.
- **Why it matters:** if true, disabling JPEG encode for onboard inference frees ~58ms, giving a ~73ms floor (~13.7 fps ceiling) before any network inference cost. That budget determines the model size for the onboard vision hover work.

## HYP-GAP8-DORY-CRASH
- **Statement:** The `dorytest` custom `simple_cnn` network's hardware crash (FACT-022 — semihosting runtime-error exit right after L3 buffer allocation, before any layer executes) is caused by a bug specific to that non-stock network/build (e.g. its `CORE=1` single-core cluster config, or a DORY-codegen issue for this particular net) rather than a defect in the `gap_sdk`/DORY/JTAG toolchain itself.
- **Type:** hypothesis
- **Scope:** GAP8 onboard-inference toolchain bring-up, `real_flight/GAP8_DORY_PROMPT.md` Track B.
- **Supported by:** everything upstream of network execution verifiably works on this exact hardware/toolchain combination — clean build, clean 100% JTAG flash, clean JTAG boot with live console output through three successful buffer allocations. The failure point is deep inside network-specific code (cluster/DMA startup for layer 0), not in generic boot/runtime setup.
- **Why it is not a fact:** no stock DORY reference example has been run on this same hardware to confirm the toolchain itself is clean under a known-good network. No register/PC dump was captured at the fault, so the exact exception type is unknown — a chip-side or `gap_sdk`-install-side bug that only a cluster-launching network would trigger cannot be ruled out.
- **Confidence:** medium — the reasoning is sound (isolate by process of elimination) but unverified by the one experiment that would actually confirm it.
- **Falsification test:** flash a real DORY-shipped stock example (not `simple_cnn`) through the identical `bitcraze/aideck` container pipeline and JTAG cable override. Clean run → hypothesis confirmed, `simple_cnn` is the bug. Same crash → hypothesis false, something in this `gap_sdk`/hardware pairing is broken for any cluster-launching network.
- **Why it matters:** determines whether Track B's toolchain risk is actually closed (just swap the test network) or still open (need to debug `gap_sdk`/hardware itself before any real vision model can be deployed).
