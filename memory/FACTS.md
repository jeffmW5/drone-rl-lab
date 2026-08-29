# Facts

> Direct observations only. No explanations here.

## FACT-001
- **Statement:** The current best legacy trajectory-following lap is `exp_016` at 13.49s with 2/10 Level 2 finishes.
- **Type:** fact
- **Scope:** Legacy trajectory-following racing line
- **Supported by:** `README.md`, `outbox/exp016_l2_benchmark.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** A committed legacy controller result beats 13.49s or 2/10.

## FACT-002
- **Statement:** The early direct-racing high-water mark for average gates is `exp_023`, with 0.8 average gates and 0 finishes on 10 Level 2 runs.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv` line, Level 2 benchmark
- **Supported by:** `results/exp_023_racecore_oob/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** A committed direct-racing result exceeds 0.8 average gates.

## FACT-003
- **Statement:** `exp_056` reached 28.92 mean training reward and 0 gates with 0.64s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, bilateral progress config
- **Supported by:** `results/exp_056_bilateral_progress/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-004
- **Statement:** `exp_057` reached 9.78 mean training reward and 0.2 average gates with 0.63s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, body-frame obs plus reduced progress
- **Supported by:** `results/exp_057_body_frame_obs/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-005
- **Statement:** `exp_058` reached 37.84 mean training reward and 0 gates with 1.22s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, soft-collision curriculum
- **Supported by:** `results/exp_058_soft_collision/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-006
- **Statement:** `exp_060` reached 28.02 mean training reward and 0 gates with 0.66s average mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, combined body-frame + soft collision + strong progress
- **Supported by:** `results/exp_060_combined/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-27
- **Next falsification test:** none needed; this is a recorded result.

## FACT-007
- **Statement:** `exp_059` reached 32.502 +/- 1.149 mean training reward and 0 gates with 0.79s average matched mid-air benchmark flight.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, asymmetric critic, `level2_midair` benchmark
- **Supported by:** `results/exp_059_asymmetric_critic/metrics.json`, `results/exp_059_asymmetric_critic/benchmark.json`, `results/exp_059_asymmetric_critic/EXPERIMENT.md`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-008
- **Statement:** The preexisting generic deployment controller could not load `exp_059`'s asymmetric checkpoint correctly; actor-only asymmetric loading support was required before benchmarking.
- **Type:** fact
- **Scope:** Evaluation tooling for asymmetric direct-racing checkpoints
- **Supported by:** local checkpoint load test on 2026-03-28, local patch to `lsy_drone_racing/control/attitude_rl_generic.py`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** A later benchmark path loads the same checkpoint correctly without architecture-aware logic.

## FACT-009
- **Statement:** `exp_061` stochastic deployment of exp_060 model: 1.67s avg flight (2.5x longer than deterministic 0.66s), 0 gates in 5 runs.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, stochastic vs deterministic deployment
- **Supported by:** `results/exp_061_stochastic_deploy/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-010
- **Statement:** `exp_062` temperature-scaled deployment across T=0.1-1.0: 2 gate passages in 70 total runs. No temperature value produced reliable gate passage.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, temperature-scaled deployment of exp_060 model
- **Supported by:** `results/exp_062_temperature_scaled/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-011
- **Statement:** `exp_064` with ent_coef=0.03, no max_logstd clamp, ent_coef_final=0.001, 7200s budget: 7.78 mean reward (flat ~8 throughout 2.5M steps), 0 gates, 0.52s deterministic crash.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, entropy annealing configuration
- **Supported by:** `results/exp_064_entropy_annealing/metrics.json`, `results/exp_064_entropy_annealing/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-012
- **Statement:** `exp_067` with only max_logstd removed from exp_060 (ent_coef=0.01): 29.99 mean reward (matched exp_060's 28.02), 0 gates, but 1.70s deterministic flight (2.6x longer than exp_060's 0.66s). Stochastic: 2.42s.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, logstd clamp ablation
- **Supported by:** `results/exp_067_no_logstd_clamp/metrics.json`, `results/exp_067_no_logstd_clamp/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-013
- **Statement:** exp_064's training failure (flat reward ~8) was caused by ent_coef=0.03, not by removal of the max_logstd clamp. exp_067 with ent_coef=0.01 and no clamp matched exp_060's reward trajectory.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, entropy coefficient effect at 3600s budget
- **Supported by:** exp_064 vs exp_067 comparison (same clamp removal, different ent_coef)
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Run with ent_coef between 0.01 and 0.03 to find the threshold.

## FACT-014
- **Statement:** `exp_068` with 7200s budget (no clamp, ent_coef=0.01): 42.84 mean reward (all-time high, peak 44.53 still climbing), 0/15 deterministic gates (1.67s avg), 3 gate passages in 45 total runs across stochastic/temperature modes.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, extended unclamped training
- **Supported by:** `results/exp_068_extended_no_clamp/metrics.json`, `results/exp_068_extended_no_clamp/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-015
- **Statement:** Doubling training budget from 3600s to 7200s increased training reward from 30 to 43 but did not meaningfully improve deterministic benchmark flight time (1.70s → 1.67s) or gate passage (0 → 0).
- **Type:** fact
- **Scope:** exp_067 vs exp_068 comparison, same config except budget
- **Supported by:** exp_067 and exp_068 benchmark results
- **Counterevidence:** 3 sparse gate passages in stochastic/temperature modes (vs 0 for exp_067) could indicate marginal improvement, but sample size is small
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** 4x budget training to see if trend continues or breaks through.

## FACT-016
- **Statement:** `exp_069` with hidden_size=128 (2×128 MLP, 48K params): 42.29 mean reward (peak 52.39, new all-time high), 2/15 deterministic gates (first ever in this family), 5/15 T=0.3 gates (33%), 0/15 stochastic gates. 7 total in 45 runs.
- **Type:** fact
- **Scope:** Direct `RaceCoreEnv`, mid-air benchmark, network capacity experiment
- **Supported by:** `results/exp_069_larger_network/metrics.json`, `results/exp_069_larger_network/benchmark.json`
- **Counterevidence:** none noted
- **Confidence:** high
- **Last reviewed:** 2026-03-28
- **Next falsification test:** none needed; this is a recorded result.

## FACT-017
- **Statement:** Increasing hidden_size from 64 to 128 (same 7200s budget, same config) improved deterministic gate passage from 0/15 to 2/15 and T=0.3 from 2/15 to 5/15, while matching mean reward (42.29 vs 42.84) and exceeding peak reward (52.39 vs 44.53).
- **Type:** fact
- **Scope:** exp_068 vs exp_069 comparison, same config except hidden_size
- **Supported by:** exp_068 and exp_069 benchmark results
- **Counterevidence:** Sample sizes are small (15 runs per mode); the rates could be noise. Average deterministic flight time was shorter (0.86s vs 1.67s), suggesting different behavior.
- **Confidence:** medium (directional signal is clear, exact rates may be noisy)
- **Last reviewed:** 2026-03-28
- **Next falsification test:** Run 50+ benchmark runs to confirm statistical significance of gate passage improvement.

## FACT-018
- **Statement:** The AI Deck WiFi stream does NOT stall when the client runs natively on Windows joined to `aideck-stream`. A 120s sustained throughput test delivered 881 frames, 8890 packets, 6.20 MB, `stall=null`, `over_stall_gap=0`. Reconnect was 6/6 clean. The ESP firmware was the clean official `58f15fa` image — the same image that stalled after one frame under VirtualBox slirp.
- **Type:** fact
- **Scope:** AI Deck ESP32/GAP8 stream, real hardware, 192.168.4.1:5000, native Windows host DESKTOP-CP5V6TB
- **Supported by:** `real_flight/aideck_logs/session_20260828_211353/` (REPORT.md and all four `summary.json`)
- **Counterevidence:** none. Direct contradiction of the prior VM baseline `packet_test_20260516_112123` (one frame then stall) and `reconnect_test_20260516_085408` (1/8 recovery).
- **Confidence:** high
- **Next falsification test:** repeat on a second Windows host, and repeat in the VM to confirm the stall still reproduces there.

## FACT-019
- **Statement:** Sustained AI Deck frame delivery measured 7.342 fps over 120s: mean inter-frame interval 136ms, median 140ms, min 47ms, max 406ms. Payload rate was 50.5 KiB/s at 324x244 JPEG, ~7.0 KB per frame.
- **Type:** fact
- **Scope:** Same session as FACT-018. End-to-end deck-to-PC rate, not an isolated onboard measurement.
- **Supported by:** `real_flight/aideck_logs/session_20260828_211353/throughput_test_20260828_211428/summary.json`
- **Counterevidence:** none
- **Confidence:** high for the end-to-end number; the attribution of that 136ms to onboard capture+encode is inference, not measurement — see HYP.
- **Next falsification test:** measure capture and encode time on the GAP8 directly and confirm they sum to ~131ms.

## FACT-020
- **Statement:** The GreenWaves AutoTiler can no longer be obtained. Bitcraze's GAP8 documentation page carries a top-of-page warning that the GreenWaves Technologies website is down, preventing fetching and compiling the autotiler, and states that deploying neural networks through `gap_sdk` is therefore not possible unless you already have the file. The AutoTiler is closed-source, was never shipped inside `gap_sdk`, and was pulled from that site at setup time via `make autotiler`. Bitcraze's builder image is titled "Builder docker image for Bitcraze AIdeck for the GAP8 sdk (without autotiler)", so Docker does not route around it. Bitcraze documents DORY as the alternative NN deployment path.
- **Type:** fact
- **Scope:** External toolchain availability as observed 2026-08-29. Constrains GAP8 onboard-inference work only; does not touch the sim RL line or the AI Deck WiFi stream.
- **Supported by:** bitcraze.io GAP8 dev page; github.com/bitcraze/docker-aideck repo title; github.com/GreenWaves-Technologies/gap_sdk
- **Counterevidence:** none found. Note this is an availability observation from vendor documentation, not a bench measurement.
- **Confidence:** high that the documented outage is real; unknown whether it is permanent or temporary.
- **Consequences:** `VISION_HOVER_PLAN.md` Track B and Phase 4 retargeted from GAPflow/nntool to DORY. Track B's original "flash Bitcraze's stock NN example" is blocked, because the facedetection and classification examples are the AutoTiler-dependent ones.
- **Not verified:** whether a previously-pulled AutoTiler copy already exists in a container on the Linux VM; whether DORY's pinned `gap_sdk` 3.6 and Bitcraze's tested 3.8.1 can be satisfied by one tree.
- **Next falsification test:** search the VM's Docker volumes for an existing AutoTiler library. If one is found, the original path is unblocked locally and this fact's practical consequence narrows to "cannot be re-obtained" rather than "cannot be used".

## FACT-021
- **Statement:** A prior local DORY+`gap_sdk` bring-up already existed at `~/projects/gap_sdk` on the Linux VM (`dorytest`/`dorytest_old`), untracked in any repo. After fixing a chip-target install path (`install/GAP8_V2` was missing; symlinked to the already-populated `install/GAP8_V3`, which is chip-generic) and building `gap8-openocd` from source (`riscv/riscv-openocd` @ `1449af5bd`, per `gap_sdk`'s own pinned Makefile target), the existing DORY-generated network (`dorytest`, a custom 19-conv-layer + FC int8 CNN, not a DORY stock example) built clean against the AI Deck's actual board config (`configs/ai_deck.sh`, `TARGET_CHIP=GAP8_V2`, `gap_rev1` chip). Link-time memory: L2 63,176 B / 512 KB (12.05%), L1 28 B / 65,532 B (0.04%), fc_tcdm 5,068 B / 16,380 B (30.94%).
- **Type:** fact
- **Scope:** GAP8 onboard-inference toolchain, Linux VM, `~/projects/gap_sdk`. Resolves the SDK-version-mismatch risk in FACT-020; does not resolve a hardware inference measurement.
- **Supported by:** `real_flight/GAP8_DORY_RESULT.md`, build log in this session, `~/projects/gap_sdk/dorytest/BUILD/GAP8_V2/GCC_RISCV_PULPOS/main.size`
- **Counterevidence:** none. GVSoC simulation of the same build crashed (SIGABRT, no output) — not diagnosed, not required for the exit criterion.
- **Confidence:** high for the build result; not applicable to runtime performance (none measured).
- **Not verified:** hardware inference rate, free L2 at runtime, per-layer checksum correctness — all require the physical AI Deck on JTAG, which is not connected to this VM (`lsusb` shows no Olimex adapter present).
- **Next falsification test:** connect the Olimex JTAG adapter + AI Deck/Crazyflie via VirtualBox USB passthrough, run `make all` in `~/projects/gap_sdk/dorytest` (builds, flashes, runs), read `cycle_network_execution` off UART.

## FACT-022
- **Statement:** With the Olimex ARM-USB-TINY-H JTAG adapter connected to the Linux VM, the `dorytest` DORY-generated `simple_cnn` network built for `ai_deck`/`GAP8_V2` (FACT-021) flashed to the AI Deck's hyperflash successfully over JTAG (498,556 bytes, 100%, via the `bitcraze/aideck` Docker image's OpenOCD — the only local OpenOCD build with GreenWaves' custom `gap8` target driver; the `gap_sdk`-recipe vanilla `riscv/riscv-openocd` build has no such driver and cannot flash or run GAP8 at all). Loading the flashed ELF via JTAG and starting it produced three live prints over GAP8's JTAG debug console ("L3 Buffer alloc initial ... Ok" ×3), then the target issued semihosting `SYS_EXIT` with reason `0x20023` (`ADP_Stopped_RunTimeErrorUnknown`) — a runtime fault, not a normal exit — before any layer executed or any `print_perf`/checksum output appeared. Reproduced twice, identically.
- **Type:** fact
- **Scope:** GAP8 onboard-inference hardware run, this specific `dorytest` build only.
- **Supported by:** `real_flight/GAP8_DORY_RESULT.md`, session build/flash/run logs.
- **Counterevidence:** none. Not yet tested: whether a DORY *stock* reference example (not this custom `simple_cnn`) crashes the same way.
- **Confidence:** high that build→flash→boot works on this hardware; high that this specific network crashes deterministically right after L3 buffer setup; zero confidence on root cause (network bug vs. hardware/config issue) or on any fps/free-L2 number, since none was reached.
- **Not verified:** whether a DORY stock example runs cleanly on the same hardware/toolchain (would isolate network-bug vs. toolchain-bug, exactly the ambiguity `GAP8_DORY_PROMPT.md` warned this scope deviation would create); the specific exception/fault type (only a generic semihosting exit code was captured, no register/PC dump).
- **Next falsification test:** flash and run a real DORY-shipped example network through the same pipeline; if it also crashes at the same point, suspect the `gap_sdk`/hardware combination rather than the network.

## FACT-023
- **Statement:** DroNet (DORY's own `PULP.GAP8` regression-suite network, NEMO frontend, `config_NEMO_dronet.json`) built via a fresh `pulp-platform/dory` clone, flashed over JTAG (993,456 bytes, 100%), and ran to completion on the physical AI Deck's GAP8: all 15 layers executed, final output checksum OK, clean exit. Measured: 35,989,730 cycles at the app's configured 100MHz FC/cluster clock = 359.9ms per inference (≈2.78 fps), single core (`CORE=1`), 41,103,104 MACs (1.14 MAC/cycle). Static L2 footprint 55,184 B / 512KB (10.53%); full 1.29MB of weights paged through L3/hyperflash per layer via DORY's tiler, no allocation failure.
- **Type:** fact
- **Scope:** GAP8 onboard-inference hardware measurement — this is the Track B exit criterion (`GAP8_DORY_PROMPT.md`), now met.
- **Supported by:** `real_flight/GAP8_DORY_RESULT.md`, this session's build/flash/run logs.
- **Counterevidence:** none. Resolves FACT-022's ambiguity: DroNet ran clean on the identical `gap_sdk` install, OpenOCD, and physical board that the earlier `simple_cnn` crashed on — confirms the crash was that network's own bug, not the toolchain.
- **Confidence:** high.
- **Not verified:** multi-core performance (this run used 1 of GAP8's 8 cluster cores — the realistic deployment number is likely much faster and wasn't measured); root cause of the `simple_cnn` crash (FACT-022, now known to be network-specific, not further diagnosed); root cause of odd per-layer "Checking L2 input/weights: Checksum Failed" lines in the DroNet run (final output checksum passed, so likely a false-positive in the generated harness, not investigated).
- **Next falsification test:** rebuild DroNet with multi-core enabled and re-measure fps for the real frame-budget number `VISION_HOVER_PLAN.md` needs.

## FACT-024
- **Statement:** DroNet rebuilt with `CORE=8` (identical network, identical 100MHz clock, only core count changed) ran 6,381,688 cycles = 63.82ms/inference (15.67 fps), 6.441 MAC/cycle, final output checksum OK. Speedup vs. the CORE=1 baseline (FACT-023, 359.90ms): 5.64×.
- **Type:** fact
- **Scope:** GAP8 onboard-inference throughput, `real_flight/GAP8_PERF_PROMPT.md` Task 1.
- **Supported by:** `real_flight/GAP8_PERF_RESULT.md`, this session's build/flash/run logs.
- **Counterevidence:** none.
- **Confidence:** high.
- **Next falsification test:** none needed for this number; multi-core + higher clock combined is the open next step (blocked on FACT-025's clock ceiling).

## FACT-025
- **Statement:** Raising GAP8's cluster clock to 150MHz (from the working 100MHz) hangs reproducibly, identically, across 3 attempts that varied FC clock (100 vs 150MHz) and voltage (1000mV default vs 1200mV, the documented `DCDC_DEFAULT_NV` max) independently. All three stop at the exact same content point (same layer, same checksum values) with no crash/exit code — the run simply stops producing output. `gap_sdk` source (`pmu_driver.c`) documents the cluster domain's spec range as 87–175MHz, voltage-coupled; 1200mV should cover 150MHz per that spec's linear FV-slope model, but did not in practice.
- **Type:** fact
- **Scope:** GAP8 clock scaling, this hardware/`gap_sdk` install. Verified-correct clock ceiling remains 100MHz (matching FACT-023/024).
- **Supported by:** `real_flight/GAP8_PERF_RESULT.md`, 3 build/flash/run attempts this session.
- **Counterevidence:** none — all 3 attempts failed identically.
- **Confidence:** high that 100MHz is safe and 150MHz is not, with voltage/FC-clock ruled out as the sole cause; low confidence on the actual root cause or the true ceiling between 100–150MHz (untested).
- **Next falsification test:** binary-search 100–150MHz; re-attempt with `gdb_port` enabled to capture a register/PC dump at the hang.

## FACT-026
- **Statement:** HM01B0 camera capture time measured directly on GAP8 via `pi_perf` cycle counting (30 frames, 324×244 QVGA grayscale, 100MHz FC, no JPEG encode), two configurations: (1) camera started once, captured repeatedly without stopping between frames — mean 25.12ms, range 15.75–29.33ms (frame 0 excluded as a 69.2ms warm-up outlier); (2) camera stopped and restarted every frame, exactly matching `wifi-img-streamer.c`'s actual loop — mean 69.408ms, stdev 0.0205ms (extremely tight), min 69.354ms, max 69.448ms.
- **Type:** fact
- **Scope:** AI Deck camera capture cost, real hardware, resolves HYP-AIDECK-RATE's capture half.
- **Supported by:** `real_flight/GAP8_PERF_RESULT.md`, `capture_timing_app/` build/flash/run logs.
- **Counterevidence:** none. The two configurations' results are mutually consistent: config 1's one-off frame-0 cost (69.2ms) matches config 2's steady-state per-frame cost (69.4ms) almost exactly, both attributable to the same camera stop/start resync, whether paid once or every frame.
- **Confidence:** high, especially for config 2 (±0.02ms spread over 30 frames).
- **Not verified:** whether continuous-streaming capture (config 1's ~25ms) holds up when combined with a real inference workload in one integrated pipeline — only measured standalone.
- **Next falsification test:** build a combined continuous-capture + inference app and re-measure end-to-end.

## FACT-027
- **Statement:** A live AI Deck distance-calibration run saved at 2026-08-29T18:45:57 produced 3 samples at 0.6096 m, 0.813 m, and 0.90805 m with fitted `k_px_m` = 18.855 and worst residual 3.95%. The wizard warned that the distance span was only 1.49× (want 2×). The saved artifact records `marker_input` 4.0 inches / `marker_size_m` 0.1016, giving `focal_px` 185.58 (FOV 82.2 deg on the 324 px axis). The physical marker was a 4 in x 4 in square throughout, switched up from 1 in because the 1 in would not detect -- confirmed by the operator 2026-08-29.
- **Type:** fact
- **Scope:** vision-hover step 2, native Windows host, this one live run
- **Supported by:** `real_flight/marker_calibration.json` (samples, k, residuals, original wizard `marker_input`); operator report of the 4 in physical size and 1 in non-detection
- **Counterevidence:** the fit is weak in two ways the headline number hides. Span is 1.49x, below the wizard's own 2x guard, so k is poorly constrained. More telling, the residuals are monotonic in distance (-3.29%, +3.10%, +3.95% at 0.610/0.813/0.908 m) rather than scattered -- the signature of a constant distance-datum offset, not noise. Refitting `size = k/(d + c)` yields c = -11.8 cm, k = 15.698, and drops the worst residual from 3.95% to 0.41%. That is suggestive only: 3 points against 2 parameters is near-interpolation and may simply be overfitting.
- **Confidence:** high for the saved samples, k, and residuals as recorded; high that the physical marker was 4 in. Low that k = 18.855 is the right constant -- if the offset is real, the correct near-field constant is 15.698 and the current k over-reads distance close in.
- **Last reviewed:** 2026-08-29 (superseded the earlier note that the wizard recorded 1.0 in; that was a transient UI state, not what was saved)
- **Next falsification test:** one sample at 0.30 m measured decides it. No offset predicts ~63 px; a -11.8 cm offset predicts ~86 px. Same reading also lifts the span past 2x. Full spread should be roughly 0.30 / 0.50 / 0.80 / 1.20 / 1.80 m, which is 6x span and uses the 4 in marker's real 2.12 m range.
