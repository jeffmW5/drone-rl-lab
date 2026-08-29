# Prompt: Marker Distance Calibration + Capture Wizards

Date written: 2026-08-29
For: the Windows agent
Status: **not started**
Parent plan: `real_flight/VISION_HOVER_PLAN.md` — "Simplified first task"
Depends on: `real_flight/marker_detector.py` (done, self-tested, see below)

## Goal

Two small wizards, both native-Windows (per FACT-018 — capture never runs
through the VM):

1. **Distance calibration** — build the apparent-size-to-real-distance curve
   for the printed marker.
2. **Capture** — record frames (and telemetry, where it helps) during flight
   so the marker detector can auto-label them afterward.

Exit criterion: a calibration curve/table saved to disk, and a capture run
that produces frames the detector in `marker_detector.py` reliably finds.

## Why this, now

The vision-hover first task was narrowed to two numbers only: how far the
marker is from vertical-center of frame (drives "rise to level"), and the
marker's apparent size in pixels (drives "adjust distance"). Both numbers
come from `real_flight/marker_detector.py` — classical CV (threshold →
contour → bounding box), already written and self-tested on synthetic frames
(`python3 real_flight/marker_detector.py --self-test`, all pass). It has never
seen a real camera frame. That's what these two wizards are for: get real
frames in front of it, and get a real distance calibration so "apparent size"
means something in centimeters.

## Read first

1. `real_flight/VISION_HOVER_PLAN.md` — "Simplified first task" section
2. `real_flight/marker_detector.py` — the detector these wizards call into.
   Read its docstring and `detect_marker()`'s signature; it takes a grayscale
   `numpy` array and returns a `MarkerDetection` (found, center_x/y,
   vertical_offset_px, horizontal_offset_px, size_px, bbox).
3. `real_flight/windows_testbed/README.md` and `flight_watch_gui.py` — **check
   this before building a new capture tool.** Flight Watch already records
   camera frames with per-frame timing (`frames.csv`) during a real flight,
   without holding the Crazyradio (so it doesn't conflict with `fly.py`). It
   may cover most or all of wizard 2's job already. If so, the "capture
   wizard" here may reduce to: confirm Flight Watch's output format works
   with the detector, and skip building anything new.
4. `memory/FACTS.md` FACT-018 (native-host-only capture), FACT-019 (324x244
   JPEG, the existing capture resolution/format)

## The marker

Plain 1"×1" square, high contrast (black on white), hand-drawn or printed —
not a fiducial tag. `marker_detector.py` does threshold + contour + bounding
box, tries both polarities (dark-on-light and light-on-dark), and rejects
non-square-ish contours and implausible sizes. It does not need a fiducial
library, but it does need real contrast and a flat, glare-free print to work
as well as it does on synthetic frames — that's an open question this session
answers.

## Wizard 1 — distance calibration

Goal: a table (or fitted curve) of `apparent_size_px -> real_distance_cm`.

Steps:

1. Print/tape the marker to a fixed spot (e.g. a wall).
2. Position the AI Deck's camera facing it at a few known distances — a tape
   measure is enough, doesn't need to be precise to the millimeter. Cover a
   sensible range for "rise from the floor toward it" — e.g. 20cm to 200cm in
   some reasonable number of steps (your call on spacing; more samples near
   the expected operating distance, fewer far out, is reasonable).
3. At each distance, capture a frame (or several, and average) and run it
   through `detect_marker()` to get `size_px`.
4. Fit `size_px` vs `distance_cm`. Physically this should follow an inverse
   relationship (`size_px ≈ k / distance_cm` for some constant `k`, pinhole
   camera model) — fit that, but sanity-check the fit visually against the
   raw samples rather than trusting it blind, especially at the range's
   extremes where the marker may be too small/large for the detector's
   `min_area_frac`/`max_area_frac` guards (currently 0.001–0.5 of frame area;
   adjust if real data needs it, but say so if you do).
5. Save the calibration (table + fit constant(s)) to
   `real_flight/marker_calibration.json` — pick a schema, document it in a
   header comment or the wizard's own docstring.

A simple guided script (CLI or a small Tkinter GUI matching the existing
`testbed_gui.py`/`flight_watch_gui.py` style, your call) that walks through
"connect to deck → capture at distance X → repeat → fit → save" is the
`wizard` framing here — doesn't need to be more polished than the existing
test bed tools.

## Wizard 2 — capture

Goal: frames from real flight (or just manual repositioning near the floor,
facing the marker — doesn't need to be airborne to be useful data) that
`marker_detector.py` can label afterward.

**Check `flight_watch_gui.py` first.** If it already saves frames in a usable
format (whatever `frames.csv` + saved images currently look like) at the
right resolution, this wizard may just be "run Flight Watch, verify the
detector works on its output" rather than new code. Only build something new
if Flight Watch's format or resolution doesn't fit — say plainly which case
it turned out to be.

If something new is needed: same shape as Flight Watch — record frames with
per-frame timing, from the AI Deck WiFi stream, without needing the
Crazyradio (so it doesn't fight `fly.py` for the radio if flight is involved
later). Telemetry (Flow-deck position/height) is a nice-to-have for
cross-checking, not required — the label comes from the marker in the image
itself, not from telemetry.

## What "done" looks like

- `real_flight/marker_calibration.json` (or equivalent) exists, with enough
  real samples to trust the fit within the operating range.
- A capture session (or confirmation that `flight_watch_gui.py` already
  suffices) producing frames that `marker_detector.py` detects reliably —
  report the miss rate if it's not 100%, and why (contrast, glare, motion
  blur, marker too small/large).
- Update `real_flight/VISION_HOVER_PLAN.md`'s "Simplified first task" step
  table: mark steps 2/3 with their actual status, not just "depends on step 1".

## Failure is a valid result

If the detector doesn't hold up on real frames — glare, JPEG compression
artifacts, camera noise the synthetic test never had — that's a real, useful
finding. Report exactly how it fails (false negatives? false positives on
background clutter? size estimate noisy?) rather than tuning parameters
blind. The synthetic self-test proves the geometry math is right, not that
the CV survives a real camera.

## Out of scope

- Training the CNN (step 5) — needs steps 2–4 done first.
- Flight control changes, `fly.py` modes — separate task.
- The GAP8/DORY deployment side — Linux VM's job, already proven
  (`GAP8_DORY_RESULT.md`).

## Coordination

```bash
python3 scripts/agent_lock.py claim <AGENT_ID>
python3 scripts/agent_lock.py heartbeat <AGENT_ID> --task "marker-wizards" --status "wizard 1: distance calibration"
python3 scripts/agent_lock.py release <AGENT_ID>
```

Pull before touching shared files. Rebase before committing.
