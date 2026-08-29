# Prompt: GAP8 Performance Envelope — multi-core, clock, and capture cost

Date written: 2026-08-29
For: the Linux VM agent
Status: **not started**
Predecessor: `real_flight/GAP8_DORY_RESULT.md` (Track B, PASSED)
Parent plan: `real_flight/VISION_HOVER_PLAN.md`

## Goal

Produce two measured numbers that together set the MAC budget for the
vision-hover model:

1. **DroNet inference time with all 8 cluster cores**, and separately, at a
   higher clock.
2. **Camera capture time measured on the GAP8 itself**, with no JPEG encode.

Exit criterion: both numbers measured on the physical AI Deck, plus a MAC
budget table in `VISION_HOVER_PLAN.md` derived from them.

## Why this, now

Track B answered "can it run a net at all" — yes, 359.9ms single-core at
100MHz for DroNet, 41.1M MACs, 1.14 MAC/cycle (FACT-023). That works out to
**114 MMAC/s**, which is the useful conversion rate: it lets a model be sized
in MACs before anyone writes it.

The problem is that the deployable rate is unknown by roughly an order of
magnitude, and the model design depends entirely on which regime we are in:

| Config | Throughput | MAC budget at 27ms | vs DroNet (41.1M) |
| --- | --- | --- | --- |
| 1 core @ 100MHz | 114 MMAC/s (measured) | ~3.1M | 7% — forces a very small net |
| 8 cores @ 100MHz | unknown | unknown | unknown |
| 8 cores + higher clock | unknown | unknown | unknown |

Every row below the first is a guess right now. Nobody should design the
vision-hover network until they are measurements. That is this task.

The 27ms figure comes from a ~10Hz control target against a ~73ms capture
floor — and that floor is itself unverified, which is task 2.

## Read first

1. `real_flight/GAP8_DORY_RESULT.md` — the working pipeline, in detail
2. `memory/FACTS.md` FACT-023 (the measurement), FACT-022 (the `simple_cnn`
   crash), FACT-020 (AutoTiler outage)
3. `memory/HYPOTHESES.md` HYP-AIDECK-RATE — the capture attribution task 2 tests
4. `real_flight/VISION_HOVER_PLAN.md` — "Frame budget"

The pipeline is already proven and documented. Reuse it exactly: fresh
`pulp-platform/dory` clone at `~/projects/dory`, `bitcraze/aideck` container
for OpenOCD, `GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg`
override, JTAG load-and-start with the session held open. Do not re-derive it.

## Task 1 — multi-core DroNet

Rebuild the same DroNet app with the cluster's 8 cores instead of 1 and
re-measure. Same network, same clock, one variable.

- Change only the core count (`CORE=1` in the DORY-generated Makefile).
- Keep the clock at 100MHz for this run so it is directly comparable to
  FACT-023.
- **Validate every run with the final output checksum.** A fast wrong answer
  is worse than a slow right one. If the checksum stops passing at 8 cores,
  that is the finding — report it, do not tune around it.
- Report cycles, derived ms, fps, and MAC/cycle. Report the speedup versus
  single-core explicitly.

Note the per-layer "Checking L2 input/weights: Checksum Failed" lines seen in
the Track B run, alongside passing output checksums. Treat only the final
output checksum as authoritative for now, but say whether the same pattern
recurs — if the harness's intermediate checks are unreliable, that matters
later when validating our own model.

## Task 2 — clock

Separately from task 1, raise the clock. `main.c` already calls
`pi_freq_set(PI_FREQ_DOMAIN_FC/CL, 100000000)`.

- **Do not guess the maximum.** Find GAP8's actual supported cluster/FC
  frequency from the GreenWaves datasheet or `gap_sdk` headers and cite the
  source in the writeup. I do not know this number and neither should you
  without checking. Voltage may be coupled to it.
- Step up, re-validate the checksum at each step, and stop at the highest
  clock that still produces a correct final output. Report the highest
  *verified-correct* clock, not the highest that booted.
- Report core count and clock as separate variables. Do not report a single
  combined "best" number without the breakdown.

**Cross-check the clock assumption itself.** FACT-023's 359.9ms is derived
from a cycle count divided by an assumed 100MHz. Confirm the configured clock
is the actual clock — `pi_freq_get`, or a wall-clock timing of a known number
of iterations. If the real clock differs, FACT-023's millisecond figure is
wrong and needs correcting. That is a valid and useful outcome.

## Task 3 — measure capture on-chip

`VISION_HOVER_PLAN.md`'s whole frame budget rests on "73ms capture + 58ms
encode", which came from splitting an end-to-end stream number. That is
inference, not measurement (HYP-AIDECK-RATE). It sets the ceiling on every
number in this document.

- Instrument the HM01B0 capture path directly on the GAP8 with the same
  `pi_perf` cycle-count method DORY's harness uses. `wifi-img-streamer` is a
  known-good starting point, or a minimal capture-only app.
- Measure capture alone, with **no JPEG encode** — that is the configuration
  onboard inference actually runs in.
- Report mean and spread over many frames, not a single sample.
- State plainly whether the result confirms, refines, or contradicts the
  73ms/58ms split. Any of the three is a good outcome.

## What "done" looks like

Write `real_flight/GAP8_PERF_RESULT.md` in `program.md`'s reporting format:
what changed, what was held constant, why, results, observations, inference,
confidence, what this does NOT prove, next falsification test.

Then:

- Add FACTs to `memory/FACTS.md` for each measured number, with log paths.
  Measurements only — reasoning about *why* belongs in `HYPOTHESES.md`.
- Resolve HYP-AIDECK-RATE against task 3's result: confirm, refine, or demote.
- If task 2 shows the real clock differs from 100MHz, correct FACT-023 rather
  than quietly superseding it. Do not delete counterevidence.
- **Rewrite `VISION_HOVER_PLAN.md`'s "Frame budget" section** with a real
  table: measured capture cost, measured inference throughput at the chosen
  config, and the resulting MAC budget at 5/10/15 Hz control rates. That table
  is the actual deliverable — it is what the model design will be drawn from.

## Failure policy

If multi-core does not speed things up, or breaks correctness, that is a real
finding and reshapes the plan — report it plainly with the error. If the
higher clock is unstable, report the ceiling you actually verified. If capture
turns out to be far more expensive than 73ms, say so; it would mean the
control rate is capture-bound and no amount of model shrinking helps, which is
exactly the kind of thing worth knowing before designing a model.

Do not report a simulator number as a hardware number. Do not report a run
whose final checksum failed as a performance result.

## Out of scope

- Designing or training the vision-hover model — that comes after this table
- Dataset capture (native Windows only, FACT-018)
- Flight testing, `fly.py`, hover tuning — separate hardware path, separate task
- Root-causing the `simple_cnn` crash (FACT-022) — known network-specific,
  not blocking
- Flashing TXQ16 — standing warning against it

## Coordination

```bash
python3 scripts/agent_lock.py claim <AGENT_ID>
python3 scripts/agent_lock.py heartbeat <AGENT_ID> --task "gap8-perf-envelope" --status "multi-core rebuild"
python3 scripts/agent_lock.py release <AGENT_ID>
```

Pull before touching shared files. Rebase before committing.
