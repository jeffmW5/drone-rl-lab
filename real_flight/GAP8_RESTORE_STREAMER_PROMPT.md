# Prompt: restore the AI Deck WiFi streamer to the GAP8

Date written: 2026-08-29
For: the Linux VM agent
Status: **not started — blocking**
Parent plan: `real_flight/VISION_HOVER_PLAN.md` (step 2, distance calibration)

## Goal

Get `aideck-stream` broadcasting again so the Windows side can capture frames.

Exit criterion: the SSID `aideck-stream` is visible from a native Windows host,
a client connects to `192.168.4.1:5000`, and JPEG frames arrive at roughly the
7.3 fps established in FACT-019.

## Why this is blocking

Distance calibration (`windows_testbed/marker_calibration_gui.py`) needs live
frames through *this* camera — focal length is exactly what it measures, so no
saved dataset and no other camera substitutes. The marker is printed, the
wizard is built, and the operator is at the bench. This is the only thing in
the way.

## What happened

The GAP8's hyperflash has been overwritten three times since the streamer last
ran:

| Order | App | Size | Session |
| --- | --- | --- | --- |
| 1 | `simple_cnn` (dorytest) | 498,556 B | Track B |
| 2 | DroNet | 993,456 B | Track B |
| 3 | `capture_timing_app` | — | perf envelope |

`capture_timing_app` is described in `GAP8_PERF_RESULT.md` as "no
WiFi/CPX/JPEG" — it is a bare camera-timing loop. The GAP8 app is what tells
the ESP32 to bring up the AP and sets the SSID, so with that flashed there is
nothing to start the network.

Confirmed from the Windows side 2026-08-29 with the drone powered on: 74 BSSIDs
scanned, **zero Espressif-OUI access points in range**. The ESP32 is not
transmitting. The ESP firmware itself was not touched by any of this — the last
ESP flash was the official `58f15fa` image, and FACT-018 was measured on it.

## Steps

1. Flash `wifi-img-streamer` back to the GAP8 over Olimex JTAG. Use the path
   proven repeatedly this week: `bitcraze/aideck` container for OpenOCD,
   `GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg`, props off,
   deck powered.
2. Power-cycle the Crazyflie.
3. Verify the SSID appears and frames flow.

**Do not flash a clean upstream clone as-is.** `AIDECK_WIFI_CHECKPOINT.md`
records that the official clone "defaults to RAW mode and a different AP name".
The working configuration is JPEG encoding with SSID `aideck-stream`. The
source believed to have produced it is
`/home/jeff/Downloads/aideck-gap8-examples-master/examples/other/wifi-img-streamer/wifi-img-streamer.c`
— that document says "likely", so confirm the SSID and encoding in the source
before flashing rather than trusting the path.

**Do not flash TXQ16.** Standing warning, unchanged — the stall it was meant to
fix was VirtualBox slirp, not the deck (FACT-018).

## Then do the thing that stops this recurring

Archive the built streamer image so restoring it never needs a rebuild. Commit
the binary under `real_flight/firmware/` beside the ESP image already kept
there, with a short note saying what it is and what SSID/encoding it carries.
Mark it `binary` in `.gitattributes` if its extension is not already covered.

Every capture session from here needs this image; every inference test
overwrites it. Rebuilding from source each time is the actual cost, and a
committed artifact removes it.

## Record the constraint

The GAP8 runs the streamer **or** a network, never both. This is not written
down anywhere and it shapes the plan:

| Plan step | Needs flashed |
| --- | --- |
| 2 — distance calibration | `wifi-img-streamer` |
| 3 — dataset capture | `wifi-img-streamer` |
| 6 — deploy the CNN | DORY app |

So the teacher→student loop crosses a JTAG round-trip in both directions, on
the Linux box, for work that is otherwise Windows-side. Add this to
`VISION_HOVER_PLAN.md` and `real_flight/STATUS.md`.

Worth flagging for later, not for this task: if the swap becomes the bottleneck
for iterating on the dataset, the answer is one app that can both stream and
infer. Do not build that now — it is scope creep on an unblock.

## Report

Short. `real_flight/GAP8_RESTORE_STREAMER_RESULT.md` is not needed if it simply
works — update `STATUS.md` and say so in the commit. If it does not work, that
is a real finding and does need writing up, because it would mean the ESP or
the deck changed state during the Track B / perf sessions and that is worth
knowing.

Add a FACT only if something new was measured.

## Out of scope

- Model training, dataset capture, flight testing
- Any change to the DORY apps or the perf results
- Flashing the ESP32 unless step 3 proves the ESP is the problem
