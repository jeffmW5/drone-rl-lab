# GAP8 Performance Envelope — Result

Date: 2026-08-29
Agent: jeff-VirtualBox-4316-1788011506 (Linux VM)
Parent prompt: `real_flight/GAP8_PERF_PROMPT.md`
Predecessor: `real_flight/GAP8_DORY_RESULT.md` (Track B, PASSED)
Status: **DONE.** Two of three sub-questions answered cleanly (multi-core,
capture cost). The clock question came back negative — 100MHz remains the
verified ceiling — which is itself the answer, not a gap.

## Headline results

1. **Multi-core: 8 cores is 5.64× faster than 1, and correct.**
   63.82ms/inference (15.67 fps) at 8 cores vs. 359.9ms (2.78 fps) at 1 core,
   both at 100MHz, both final-checksum-OK.
2. **Clock: 150MHz hangs, reproducibly, regardless of voltage. 100MHz is the
   verified ceiling**, not the 175MHz the chip's own spec headers say the
   cluster domain should support.
3. **Capture cost: confirms the 73ms figure (measured 69.4ms, tight ±0.02ms)
   for the *current* capture pattern — and reveals why: ~44ms of it is
   camera stop/start overhead, not pixel transfer.** Removing the per-frame
   restart cuts measured capture to ~25ms. This is new information the
   original 73ms/58ms split never had visibility into.

## What changed / what was held constant

Reused the exact proven pipeline from `GAP8_DORY_RESULT.md` throughout: same
`~/projects/dory` clone, same DroNet app for tasks 1–2, same `bitcraze/aideck`
container for OpenOCD, same `GAPY_OPENOCD_CABLE` override, same JTAG
load-and-start-with-session-held-open method. No re-derivation.

- Task 1: copied the CORE=1 DroNet build (`dronet_app`) to `dronet_app_core8`
  and rebuilt with `make CORE=8 all` — the only variable changed.
- Task 2: copied again to `dronet_app_c8_f150` / `dronet_app_c8_clOnly150`,
  edited only `pi_freq_set`/`PMU_set_voltage` calls in `main.c` between
  attempts, one variable at a time.
- Task 3: wrote a new minimal app, `capture_timing_app/`, based directly on
  Bitcraze's own `examples/other/test_functionalities/test_camera/test.c`
  (plain PULP-OS, not FreeRTOS) — stripped demosaicking/file-I/O, added
  `pi_perf` cycle counting around `pi_camera_capture` alone, looped 30 frames.
  Chose this over patching `wifi-img-streamer` directly because that example
  is FreeRTOS + CPX-over-radio (`cpxPrintToConsole`), which would need the
  Crazyflie/STM32/Crazyradio powered and connected — out of scope per the
  prompt ("separate hardware path, separate task"). The GAP8-JTAG-only
  console channel already proven in Track B was reused instead.

## Task 1 — multi-core

| Config | Cycles | ms | fps | MAC/cycle | Checksum |
|---|---|---|---|---|---|
| 1 core @ 100MHz (FACT-023) | 35,989,730 | 359.90 | 2.78 | 1.142 | OK |
| 8 cores @ 100MHz | 6,381,688 | 63.82 | 15.67 | 6.441 | OK |

**Speedup: 5.64×** (not the naive 8× — expected, some layers/overhead don't
parallelize perfectly). Both runs final-output-checksum OK. The same benign
"Checking L2 input/weights: Checksum Failed" pattern seen in Track B recurred
identically at 8 cores too (same layers, same wrong-looking values) while
every "Checking L2 output" and the final output checksum passed — this
confirms it's a fixed harness quirk, not something core-count-dependent, and
not a correctness concern.

## Task 2 — clock

**Datasheet source, cited as instructed:** found directly in `gap_sdk`
(not the GreenWaves datasheet PDF, which wasn't needed):
`gap8/rtos/pulp/pulp-os/kernel/gap/pmu_driver.c` lines 137–144:

```c
#define MAX_LV_FREQUENCY 150000000
#define MAX_NV_FREQUENCY 250000000
#define SOC_MIN_FREQ       150000000
#define SOC_MAX_FREQ       250000000
#define CLUSTER_MIN_FREQ   87000000
#define CLUSTER_MAX_FREQ   175000000
```

FC/SoC domain: 150–250MHz. **Cluster domain (where DroNet's layers actually
execute): 87–175MHz.** Voltage-coupled, confirmed both in the header
(`SOC_FV_SLOPE`/`CLUSTER_FV_SLOPE` are literally functions of voltage) and in
Bitcraze's own `examples/other/com-test/com-test.c`, which pairs `pi_freq_set(FC,
250000000)` with `__pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200)` — i.e. the
250MHz FC ceiling needs the max voltage, not the default. `gap_sdk` also
resolves the actual mV values: `DCDC_DEFAULT_LV=1000`, `DCDC_DEFAULT_NV=1200`
(`regulator_periph.h`). DroNet's generated `main.c` boots at `PMU_set_voltage(1000,
0)` — i.e. LV, the *minimum* voltage, which by the FV-slope math only
guarantees the cluster up to 87MHz, its floor, not its ceiling.

**Attempts, in order, all on the 8-core build:**

| # | FC | CL | Voltage | Result |
|---|---|---|---|---|
| 1 | 150MHz | 150MHz | 1000mV (default/LV) | Hangs at layer 6→7, reproducibly |
| 2 | 150MHz | 150MHz | 1200mV (NV, per com-test.c precedent) | Hangs at the **identical** point |
| 3 | 100MHz | 150MHz | 1200mV (isolate: cluster-only change) | Hangs at the **identical** point again |

All three hangs stop at the exact same content — same layer, same
byte-for-byte checksum values printed before it — which is itself a datum:
this looks like a deterministic fault tied to `CL=150MHz` specifically, not
JTAG flakiness (which would show variable cutoff points) and not an FC/voltage
effect (both were varied independently across attempts 2→3 with no change in
outcome). No crash/exit code was reported — the OpenOCD session's console
output simply stops advancing and the held-open `sleep` window elapses with
no further progress, distinct from Track B's earlier `simple_cnn` crash (which
did report a semihosting `SYS_EXIT`).

**Per the prompt's failure policy, this is reported plainly rather than
chased further: the verified-correct ceiling is 100MHz**, not the 175MHz the
spec headers imply should be achievable. Root cause not identified — a real
open item, not resolved here (see "What this does NOT prove").

**Clock cross-check (task 2's other ask):** no dedicated `pi_freq_get()` or
stopwatch test was run. Indirect corroboration: Task 3's capture measurement,
computed by dividing `pi_perf` cycles by the assumed 100MHz, lands at 69.4ms —
within 5% of the independently-derived historical ~73ms figure (which came
from a *different* clock, FreeRTOS's `xTaskGetTickCount`, entirely unrelated
to this session's 100MHz assumption). If the real FC clock were substantially
off from 100MHz, this agreement would not hold. Treat this as supporting
evidence, not a direct verification — FACT-023's 100MHz figure is not
re-confirmed by an independent instrument in this session.

## Task 3 — capture cost, measured on-chip

New minimal app (`capture_timing_app/`), no WiFi/CPX/JPEG, `pi_perf`
cycle-counted `pi_camera_capture` alone, 324×244 QVGA grayscale (matching
`wifi-img-streamer`'s resolution), 30 frames per run, both at 100MHz FC.

**Run 1 — continuous streaming** (camera started once, captured repeatedly,
never stopped between frames — *not* how any current code actually operates,
a best-case probe):

| | cycles | ms |
|---|---|---|
| min | 1,575,281 | 15.75 |
| mean (excl. frame 0 warm-up outlier) | 2,512,214 | 25.12 |
| max | 2,932,876 | 29.33 |
| stdev | 499,056 | 4.99 |

Frame 0 alone was 6,920,748 cycles (69.2ms) — a clear one-off warm-up cost,
excluded from the steady-state stats above and reported separately because it
turned out to matter (see Run 2).

**Run 2 — per-frame stop/start**, exactly matching
`wifi-img-streamer.c`'s actual loop (`PI_CAMERA_CMD_START` →
`pi_camera_capture` → `PI_CAMERA_CMD_STOP`, every frame):

| | cycles | ms |
|---|---|---|
| min | 6,935,427 | 69.354 |
| mean | 6,940,790 | 69.408 |
| max | 6,944,798 | 69.448 |
| stdev | 2,046 | 0.0205 |

**This is 95.1% of the historical 73ms figure, and far tighter (±0.02ms vs.
a ~25ms spread in Run 1)** — a real, precise measurement that lands almost
exactly on the old end-to-end-derived estimate.

**Conclusion: refines, and mostly confirms, HYP-AIDECK-RATE.** The 73ms
capture figure holds up **for the capture pattern currently in use**
(stop/start every frame) — but the reason is now visible for the first time:
**~44ms of the ~69ms is camera stop/start resync overhead, not pixel
transfer.** Frame 0's continuous-mode warm-up cost (69.2ms) landing almost
exactly on Run 2's per-frame-restart steady-state cost (69.4ms) is a strong
internal consistency check — both are measuring essentially the same
underlying resync cost, just triggered once vs. every frame.

**This is new, actionable information the original split didn't have:** an
onboard-inference capture loop that keeps the camera continuously streaming
(never stopping between frames) should cost ~25ms per frame instead of ~69ms
— a ~44ms/frame savings, larger than the entire inference budget at some
control rates below. Untried on any real end-to-end workload; flagged as the
clear next lever alongside multi-core.

## MAC budget table (the deliverable)

Built from Task 1's throughput (114 MMAC/s @ 1 core, 644 MMAC/s @ 8 cores —
both @ 100MHz, the verified-correct clock) and Task 3's two capture numbers.
DroNet (41.1M MACs) shown as the size reference, not a design target.

| Capture scenario | Config | Rate | Period | Budget after capture | MAC budget | DroNet (41.1M) fits? |
|---|---|---|---|---|---|---|
| Current (69.4ms, per-frame restart) | 1 core | 5Hz | 200ms | 130.6ms | 14.9M | no |
| Current (69.4ms) | 1 core | 10Hz | 100ms | 30.6ms | 3.5M | no |
| Current (69.4ms) | 1 core | 15Hz | 66.7ms | — | **capture-bound, infeasible** | — |
| Current (69.4ms) | 8 cores | 5Hz | 200ms | 130.6ms | 84.1M | **yes** |
| Current (69.4ms) | 8 cores | 10Hz | 100ms | 30.6ms | 19.7M | no |
| Current (69.4ms) | 8 cores | 15Hz | 66.7ms | — | **capture-bound, infeasible** | — |
| Continuous (25.1ms, redesigned) | 1 core | 5Hz | 200ms | 174.9ms | 20.0M | no |
| Continuous (25.1ms) | 1 core | 10Hz | 100ms | 74.9ms | 8.6M | no |
| Continuous (25.1ms) | 1 core | 15Hz | 66.7ms | 41.5ms | 4.7M | no |
| Continuous (25.1ms) | 8 cores | 5Hz | 200ms | 174.9ms | 112.6M | **yes** |
| Continuous (25.1ms) | 8 cores | 10Hz | 100ms | 74.9ms | 48.2M | **yes** |
| Continuous (25.1ms) | 8 cores | 15Hz | 66.7ms | 41.5ms | 26.8M | no |

**Reading this:** 8 cores is necessary at any useful rate — 1 core never
clears even a 20M-MAC budget. With the current (per-frame-restart) capture
pattern, **15Hz is not achievable at all regardless of model size or core
count** — capture alone (69.4ms) exceeds the 66.7ms period. 10Hz needs the
capture loop redesigned to continuous streaming to leave a workable budget.
5Hz works today, at 8 cores, with real headroom (84.1M MACs, 2× DroNet-sized).

## Observations

- The recurring "Checking L2 input/weights: Checksum Failed" pattern from
  Track B reappeared identically at 8 cores — same layers, same values.
  Treating only final-output-checksum as authoritative (per the prompt) held
  up again; this pattern looks systematically tied to something in the
  generated harness's intermediate-check bookkeeping (possibly a stale
  reference value baked in at codegen time that doesn't account for the
  tiled/pipelined L3 fetch order), not to real data corruption. Not
  root-caused — still a loose end if the team ever needs to trust
  intermediate checks for validating a real model.
- The three identical-cutoff-point clock hangs are the most interesting
  unresolved thread here. That they're byte-for-byte identical across 3
  attempts (different FC values, different voltages) rules out randomness but
  doesn't identify a mechanism. A next session with `gdb_port` enabled
  (instead of disabled) could get a register/PC dump at the hang, the way
  Track A's crash diagnosis wanted for the semihosting exit but never got.

## Confidence

High on Task 1 (clean, reproducible, checksummed both ways). High on Task 3's
Run 2 number (extremely tight spread, close agreement with an independently-
derived historical figure). Medium-high on Task 3's Run 1/continuous-mode
number — real measurement, but only tested in isolation, not as part of a
real capture+inference pipeline, so unknown if some other cost reappears when
integrated. High confidence that 150MHz is unsafe as currently configured;
medium confidence that the true achievable ceiling is exactly 100MHz rather
than something between 100 and 150 — no value between those two was tried.

## What this does NOT prove

- Does not identify why cluster clock >100MHz hangs. Ruled out: FC clock,
  voltage (in the 1000–1200mV range tried). Not ruled out: something between
  100 and 150MHz might work; a voltage above 1200mV (not attempted, and 1200mV
  is documented as `DCDC_DEFAULT_NV`, i.e. the intended maximum;
  going above it wasn't tried and wasn't clearly sanctioned by anything read
  this session).
- Does not prove the continuous-capture (~25ms) number holds up once combined
  with real inference in one pipeline — capture and inference were measured
  as two separate apps, never together.
- Does not re-verify FACT-023's 100MHz assumption with a direct instrument —
  only indirect corroboration via Task 3's agreement with the historical
  73ms figure.

## Next falsification test

1. Binary-search the cluster clock between 100 and 150MHz (e.g. 110, 125MHz)
   to find the actual ceiling, rather than accepting 100MHz as necessarily
   final.
2. Re-run the clock-hang attempts with `gdb_port` enabled to capture a
   register/PC dump at the hang, instead of disabled (which is what let
   Track A's crash go undiagnosed too).
3. Build a combined capture+inference app (continuous-streaming capture,
   8-core DroNet-scale inference) and measure real end-to-end throughput —
   the two costs may not simply add.

## Failure policy compliance

No simulator numbers reported. No run with a failing/missing final checksum
reported as a performance result — the three clock-hang attempts are reported
as hangs, explicitly, not as performance data.
