# Prompt: GAP8 Onboard Inference via DORY (Track B)

Date written: 2026-08-29
For: the Linux VM agent
Status: **not started**
Parent plan: `real_flight/VISION_HOVER_PLAN.md`

## Goal

Answer one question with a number: **can the GAP8 on our AI Deck run a neural
network at all, and at what rate?**

Exit criterion: a known reference network runs on our physical deck, with a
measured inference rate in fps and a measured free-L2 figure.

That is the whole task. Do not design our vision-hover model, do not train
anything, do not touch the flight code. The measurement gates all of it.

## Why this is the priority

This is the project-killer question for the onboard vision plan. A robot cannot
have WiFi in its control loop, so the net must run on the GAP8. If it cannot,
or if it can only run something far too small, the task changes shape. Nothing
downstream is worth designing before this number exists.

It has also just become harder, which is the second reason to do it now.

## Read first

1. `real_flight/VISION_HOVER_PLAN.md` — especially "Toolchain constraint" and
   "Where the work runs"
2. `memory/FACTS.md` FACT-020 — the AutoTiler outage
3. `real_flight/AIDECK_WIFI_CHECKPOINT.md` — the JTAG toolchain that works on
   this bench, and the standing "do not flash TXQ16" warning
4. `real_flight/STATUS.md` — hardware inventory and firmware versions

## The toolchain situation

The obvious path is closed. Bitcraze's stock NN examples (facedetection,
classification) build through the GreenWaves AutoTiler. The AutoTiler is
closed-source, was never shipped inside `gap_sdk`, and was fetched from the
GreenWaves website at setup time via `make autotiler`. That website is down.
Bitcraze's builder image is explicitly "without autotiler", so Docker does not
help. Bitcraze documents **DORY** (`pulp-platform/dory`) as the alternative.

So: DORY, not GAPflow/nntool.

`not verified`, and worth resolving early:

- DORY is tested against `gap_sdk` 3.6 at a pinned commit; Bitcraze's tested
  version for AI-deck examples is 3.8.1. It is not known that one tree
  satisfies both.
- DORY pins Python 3.6.8 and old dependencies. It ships a Dockerfile. Use it
  rather than fighting the VM's system Python.
- DORY's README lists GAP8 support but does not mention the Bitcraze AI-deck.
  Board-level differences are unexamined.

## Step 0 — the cheap check, do this before anything else

If a container on this VM ever ran `make autotiler` successfully, the library
may already be on disk, which would unblock the original path for free.

Search Docker volumes, images, and any `gap_sdk` checkout for AutoTiler
artifacts. Report found or not found. **If found, stop and report** — that
changes the recommended path and the decision is the project owner's, not
yours. Do not proceed to DORY on your own judgment in that case.

If not found, continue.

## Steps

1. **Bring up DORY.** Clone `pulp-platform/dory`, init submodules (`pulp-nn`
   kernels), build its Docker image. Resolve the `gap_sdk` version question and
   write down what tree you ended up on and why.
2. **Pick a reference network.** Something known-good that DORY already has an
   example for. Do not author a novel network here — a novel net conflates
   "our model is wrong" with "the toolchain does not work", and the point of
   this track is to separate those.
3. **Build for GAP8.** Generate C via DORY, compile against `gap_sdk`.
4. **Flash to the deck** over Olimex JTAG, using the path in
   `AIDECK_WIFI_CHECKPOINT.md`. That path is proven on this bench for both the
   ESP32 and the GAP8, so a failure here is a new finding, not expected noise.
5. **Measure**, on the physical deck, not in GVSoC:
   - inference time per frame (ms) and the derived fps
   - free L2 after the model is resident
   - total L2 the model occupies
   If you also run it in GVSoC, report both and label which is which. A
   simulator number does not satisfy the exit criterion.
6. **Report.**

## What "done" looks like

Write `real_flight/GAP8_DORY_RESULT.md` using the reporting format in
`program.md`: what changed, what was held constant, why, results, observations,
inference, confidence, what this does NOT prove, next falsification test.

Then:

- Add a FACT to `memory/FACTS.md` for the measured rate and memory, with the
  log path as its source. Measurements only.
- Anything about *why* the number came out as it did is a hypothesis, not a
  fact. `memory/HYPOTHESES.md`.
- Update `VISION_HOVER_PLAN.md` "Frame budget" if the measured N changes the
  ceiling. That section currently reasons from an unverified attribution
  (HYP-AIDECK-RATE) — if your work bears on it, say so.
- Update `real_flight/STATUS.md` open items.

## Failure is a valid result

If DORY cannot be made to build for this deck, that is a real finding and it is
worth as much as a success. Report it plainly: what you tried, where it broke,
the exact error, and what you ruled out. Do not substitute a GVSoC number for a
hardware number to make the track look passed. Do not report a partial build as
a working net.

If the number comes back too low to support a useful model, say so. That
outcome reshapes the plan, which is exactly what this track exists to do.

## Out of scope

- Training any model
- Dataset capture (that runs on native Windows — do not capture through the VM,
  slirp breaks it; see FACT-018)
- Flight testing, `fly.py`, hover tuning
- The sim RL racing line
- Flashing TXQ16 — there is a standing warning against it

## Coordination

If multiple agents are active, claim before starting:

```bash
python3 scripts/agent_lock.py claim <AGENT_ID>
python3 scripts/agent_lock.py heartbeat <AGENT_ID> --task "gap8-dory-track-b" --status "toolchain bringup"
python3 scripts/agent_lock.py release <AGENT_ID>
```

Pull before touching shared files. Rebase before committing.
