# GAP8 DORY Toolchain Bring-Up — Result (Partial)

Date: 2026-08-29
Agent: jeff-VirtualBox-4316-1788011506 (Linux VM)
Parent prompt: `real_flight/GAP8_DORY_PROMPT.md`
Status: **blocked on hardware — not done**

## What changed

Nothing was authored from scratch. A prior local DORY+`gap_sdk` bring-up already
existed on this VM at `~/projects/gap_sdk` (`dorytest`/`dorytest_old`), untracked
in any repo and unknown to `VISION_HOVER_PLAN.md` until this session found it.
This session fixed two broken pieces of that existing tree and used it to get
as far as possible without hardware:

1. **Wrong install target.** `~/projects/gap_sdk` had only ever run `make sdk`
   under the generic `gapuino_v3` board config (`TARGET_CHIP=GAP8_V3`), so
   `install/GAP8_V3/` has headers and libs for every chip variant (including
   `gap_rev1`, the AI Deck's actual chip), but nothing existed at
   `install/GAP8_V2/` — the path the AI Deck's own board config
   (`configs/ai_deck.sh`, `TARGET_CHIP=GAP8_V2`) expects. Fixed with
   `ln -s GAP8_V3 GAP8_V2` under `install/` — the content is chip-generic, only
   the directory name was wrong. After that, `dorytest` (the existing
   DORY-generated network, see below) built clean for the real `ai_deck` board
   target.
2. **Missing flasher.** `gap8-openocd` (the binary `common.sh` hardcodes as
   `GAP_OPENOCD_PATH`) did not exist; only its `share/openocd/{scripts,gap_bins}`
   had been checked out previously, install stopped there. Built it from source
   (`riscv/riscv-openocd` @ `1449af5bd`, the pinned commit in `gap_sdk`'s own
   Makefile) — `./bootstrap && ./configure --enable-jtag_dpi --disable-werror
   && make && make install` — then placed the binary at
   `install/workstation/gap8-openocd/bin/gap8-openocd` to match where
   `configs/openocd-gap8.sh` puts it on `PATH`. It runs (`gap8-openocd
   --version` → `0.10.0+dev-00841-g1449af5bd`).

## What was held constant

Reused the existing `dorytest` network as-is: a DORY-generated int8 CNN (19
BNRelu-conv layers + 1 fully-connected layer, from a custom `simple_cnn.py`,
not one of DORY's own shipped reference examples — see "Scope deviation"
below). Did not touch `dorytest_old` (an earlier, apparently-incomplete GVSoC
attempt with empty `tx_uart.log`/`power_report.csv` — left alone, not
diagnosed).

## Results

- **Build: PASS.** `dorytest` compiles and links cleanly against
  `gap_sdk`'s `ai_deck` board config (`TARGET_CHIP=GAP8_V2`, `gap_rev1`
  chip) — the config that actually matches the physical AI Deck, not the
  generic dev-board config the tree was built against before.
- **Memory footprint (from the linked ELF, `ai_deck` target):**
  - L2: 63,176 B / 512 KB = **12.05%** used
  - L1: 28 B / 65,532 B = 0.04% used
  - fc_tcdm: 5,068 B / 16,380 B = 30.94% used
- **GVSoC (simulator): FAILED — SIGABRT**, no output, no traceback, crashes
  before any UART or Python-level error. Given the exit criterion requires a
  hardware number regardless, did not spend further time root-causing this.
- **Hardware flash/measurement: NOT ATTEMPTED.** `lsusb` on this VM shows only
  the two root hubs — no Olimex JTAG adapter, no Crazyradio. The physical AI
  Deck / Crazyflie is not connected via VirtualBox USB passthrough right now.
  This is the actual blocker for the exit criterion.

## Observations

- The generated code already has the instrumentation the exit criterion
  needs: `network.c` calls `pi_perf_cyclecount`-based timing and
  `print_perf("Final", cycle_network_execution, 13513312)` (13,513,312 is the
  network's total MAC count, DORY's standard efficiency-denominator), plus
  per-layer checksums. Once flashed, cycle count → fps is a direct read off
  UART, no new instrumentation needed.
- `gap_sdk`'s install tree is universal across chip/board variants but
  namespaced by whatever `TARGET_CHIP` was active during the one-time
  `make sdk`; this is a standing trap for this repo's other GAP8 work too
  (`AIDECK_WIFI_CHECKPOINT.md` shows a *different*, working GAP8 flash flow
  using `/tmp/gap8-2025.02` via Docker — a separate, unrelated SDK checkout,
  not this one).

## Inference

The two "not verified" risks flagged in `GAP8_DORY_PROMPT.md` — DORY/`gap_sdk`
version compatibility, and whether DORY-generated code builds for the AI
Deck's actual chip target — are resolved on this VM: it builds clean. The
toolchain risk that remains is entirely the physical one: getting the deck on
JTAG.

## Confidence

High that build-for-hardware is solved. Zero confidence on real inference
rate/free-L2 — no hardware run has occurred yet, so there is no measurement
to report a confidence level for.

## What this does NOT prove

- Does not prove the network runs correctly on real silicon (checksums
  unverified — GVSoC would have checked these, it crashed instead).
- Does not prove *any* fps or memory number "in the field" — L2/L1 static
  footprint is a link-time fact, not a runtime measurement.
- Does not resolve whether this custom `simple_cnn` net is a fair proxy for
  toolchain health the way a DORY-shipped example would be (see below).

## Scope deviation from the prompt

`GAP8_DORY_PROMPT.md` says: "pick a reference network ... do not author a
novel network here." The network already built in `dorytest` is a custom
`simple_cnn`, not one of DORY's own examples. It was reused here purely
because it was already built and known to compile end-to-end — re-deriving
against a stock DORY example would mean re-resolving the same DORY-repo
dependency (DORY itself, `pulp-platform/dory`, is not present on this VM;
`~/dory` does not exist, so the code in `dorytest` was generated by DORY
running somewhere/sometime not reconstructable from this VM's current state).
If the hardware run below produces a suspicious result (e.g. wrong checksums),
that ambiguity — network vs. toolchain — is exactly what the prompt warned
about, and re-running against a real DORY example should be the next step
before trusting a bad number.

## Next falsification test

Connect the Olimex JTAG adapter (and power the Crazyflie/AI Deck) via
VirtualBox USB passthrough, then:

```bash
cd ~/projects/gap_sdk && source configs/ai_deck.sh
cd dorytest && make all    # builds + flashes over JTAG + runs
```

Read `cycle_network_execution` off UART, convert to fps, and confirm the
per-layer checksums pass. That is the actual exit criterion for this prompt —
still outstanding.
