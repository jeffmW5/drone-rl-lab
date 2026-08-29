# GAP8 DORY Toolchain Bring-Up — Result (Partial)

Date: 2026-08-29
Agent: jeff-VirtualBox-4316-1788011506 (Linux VM)
Parent prompt: `real_flight/GAP8_DORY_PROMPT.md`
Status: **hardware run attempted — network crashes before completing a single
inference; no fps/free-L2 number obtained**

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
- **Hardware flash: PASS.** Once the Olimex JTAG adapter was connected
  (VirtualBox USB passthrough), the image flashed to the AI Deck's hyperflash
  cleanly over JTAG: 498,556 bytes written, 100%, `flasher is done`.
- **Hardware execution: FAILS, reproducibly, before any layer runs.**
  Loading the ELF via JTAG and starting it (`gap8_jtag_load_binary_and_start`)
  gets three real, live prints back from the chip over GAP8's JTAG debug
  console (GreenWaves' `adv_dbg_unit`, a semihosting-style channel — not the
  physical UART):
  ```
  L3 Buffer alloc initial	@ 4388608:	Ok
  L3 Buffer alloc initial	@ 2888608:	Ok
  L3 Buffer alloc initial	@ 1388608:	Ok
  ```
  Then the target issues a semihosting `SYS_EXIT` (`op=0x18`) with reason code
  `0x20023` (`ADP_Stopped_RunTimeErrorUnknown`) and openocd's session ends.
  This happens immediately after the three allocations succeed and before any
  per-layer print, checksum, or `print_perf` output — i.e. before a single
  network layer executes. **Reproduced twice, identically**, including with a
  held-open session (`sleep 45000` before `exit`) to rule out the session
  being torn down too early — the crash itself kills the session, it isn't a
  timing artifact of my invocation.
  - No fps or free-L2 number is obtainable from this network — execution never
    reaches `print_perf`.

## Observations

- The generated code already has the instrumentation the exit criterion
  needs: `network.c` calls `pi_perf_cyclecount`-based timing and
  `print_perf("Final", cycle_network_execution, 13513312)` (13,513,312 is the
  network's total MAC count, DORY's standard efficiency-denominator), plus
  per-layer checksums. None of that ever printed — execution dies first.
- `gap_sdk`'s install tree is universal across chip/board variants but
  namespaced by whatever `TARGET_CHIP` was active during the one-time
  `make sdk`; this is a standing trap for this repo's other GAP8 work too
  (`AIDECK_WIFI_CHECKPOINT.md` shows a *different*, working GAP8 flash flow
  using `/tmp/gap8-2025.02` via Docker — a separate, unrelated SDK checkout,
  not this one).
- `configs/ai_deck.sh` defaults `OPENOCD_CABLE` to the ARM-USB-**OCD**-H
  adapter, but the bench's actual Olimex adapter is a **TINY**-H (different
  USB PID: `002b` vs `002a`). Bitcraze's own docs override this with
  `GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg` before
  sourcing `ai_deck.sh` — needed that override here too, or JTAG never finds
  the adapter at all ("unable to open ftdi device").
- The vanilla `riscv/riscv-openocd` checkout (`gap_sdk`'s own pinned Makefile
  target, which I built in the previous session) **cannot flash or run GAP8
  at all** — GreenWaves' `gap8` OpenOCD target type (`target create ... gap8`)
  is a custom C-level driver that vanilla upstream openocd doesn't have. Only
  Bitcraze's `bitcraze/aideck` Docker image (pulled fresh — the local copy had
  been pruned to save disk) has a build of openocd with that driver compiled
  in. All hardware interaction in this session went through that container,
  with the host's Olimex adapter passed through via `--device /dev/bus/usb
  --device /dev/ttyUSB0 --privileged`.
- GAP8's JTAG debug console (GreenWaves' `adv_dbg_unit`) intercepts the
  target's `printf()` via semihosting-style trap-and-resume: each print halts
  the core at a fixed address, openocd services it and resumes. This is how
  the three "L3 Buffer alloc" lines were captured live — no physical UART
  wiring was needed for it.

## Inference

The two "not verified" risks flagged in `GAP8_DORY_PROMPT.md` — DORY/`gap_sdk`
version compatibility, and whether DORY-generated code builds for the AI
Deck's actual chip target — are resolved: it builds clean, flashes clean, and
starts executing on real silicon. The toolchain end-to-end (DORY codegen →
`gap_sdk` build → JTAG flash → JTAG boot) works. What fails is the network
itself, at runtime, immediately after buffer setup and before any layer
computes — consistent with the concern the prompt raised about testing a
non-stock, "novel" network: a real hardware fault this early (before cluster
DMA or PE cores are ever touched) looks more like a bug in this specific
`simple_cnn` build (or a chip-revision/config mismatch it exposes) than a
DORY/toolchain-wide failure, since the toolchain got the code onto the chip
and running.

## Confidence

High that the toolchain (build → flash → boot) works on this hardware. High
that the specific `dorytest` network crashes deterministically right after L3
buffer setup — reproduced twice, identically, with the OpenOCD session held
open explicitly to rule out a premature-disconnect artifact. Zero confidence
on what fps/free-L2 this or any correctly-running network would show — no
measurement was obtained.

## What this does NOT prove

- Does not prove DORY-generated networks in general can't run on this GAP8 —
  only that this specific, non-stock `simple_cnn` build crashes early.
- Does not identify the crash's root cause. Semihosting reported
  `ADP_Stopped_RunTimeErrorUnknown` (generic PULP-OS fault exit), not a
  specific exception type — no register dump or PC-at-fault was captured in
  this session.
- Does not rule out a config issue specific to `dorytest`'s build (e.g.
  `CORE=1` single-core setting vs. cluster wake-up expectations) rather than
  a hardware or `gap_sdk` install problem.

## Scope deviation from the prompt — now the likely root cause

`GAP8_DORY_PROMPT.md` said: "pick a reference network ... do not author a
novel network here ... a novel net conflates 'our model is wrong' with 'the
toolchain does not work'." That's exactly what happened. The network reused
here (`dorytest`'s custom `simple_cnn`) was not a DORY stock example, and it
is now the prime suspect for the crash, precisely because everything upstream
of it (build, flash, boot) verifiably works. Re-deriving against a real DORY
example is now the load-bearing next step, not an optional nicety — but DORY
itself (`pulp-platform/dory`) is not present on this VM (`~/dory` does not
exist), so that means either re-cloning DORY and generating a known example,
or getting DORY's own reference example artifacts some other way.

## Next falsification test

1. Get a DORY-shipped reference example (not a custom net) generating C code
   the same way `dorytest` was produced, and run it through the exact same
   proven pipeline: `bitcraze/aideck` container, `configs/ai_deck.sh` +
   `GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg` override,
   flash, then:
   ```bash
   gap8-openocd -d0 -c "gdb_port disabled; telnet_port disabled; tcl_port disabled" \
     -f interface/ftdi/olimex-arm-usb-tiny-h.cfg -f target/gap8revb.tcl \
     -f tcl/jtag_boot_entry.tcl \
     -c 'gap8_jtag_load_binary_and_start "<binary>" elf 0x1c000080'
   ```
   If a stock example also crashes at the same point, the problem is this
   `gap_sdk`/hardware combination, not the network. If it runs clean, the
   `simple_cnn` build itself is the bug, and the toolchain is fully cleared.
2. Separately: capture a register/PC dump at the fault (attach with
   `gdb_port` enabled instead of disabled) to identify the actual exception,
   rather than inferring from the generic semihosting exit code.
