# AI Deck WiFi Test Bed (Windows)

Native Windows implementation of `real_flight/WINDOWS_TESTBED_PROMPT.md`.

Purpose: run the AI Deck stream diagnostics off a Windows box joined to the
`aideck-stream` AP, with no VirtualBox NAT (slirp) in the data path. Every
packet test logged before this ran through slirp, so the observed stall is
consistent with both an ESP TX-queue problem and a slirp receive stall.
**Which one it is, is `not verified`.** Settle that before flashing the TXQ16
image.

## Running it

Double-click `run_testbed.bat`. The app walks through four steps and there is
nothing to configure to get a valid run:

1. **Connect** — join the deck's WiFi, or pick *Practice run* to start a
   simulated deck on this PC and learn the tool without hardware.
2. **Find deck** — locates the deck and confirms it answers.
3. **Run tests** — packet, reconnect and throughput back to back, with a live
   frame preview and running FPS.
4. **Result** — a plain-language verdict and the folder holding the evidence.

Everything is logged automatically. One session folder per run holds each
test's own folder, a `REPORT.md` with the verdict, and a matching `.zip`.

Advanced settings (IP, port, throughput length, reconnect attempts) are behind
a collapsed panel on step 2 and only exist to override the defaults.

Console form, for headless or scripted use:

```
py -3 testbed_cli.py link
py -3 testbed_cli.py packet --duration 60 --max-frames 0
py -3 testbed_cli.py reconnect --attempts 12
py -3 testbed_cli.py throughput --throughput-duration 300
```

Runs fully offline. Standard library only; Pillow is optional and used only for
the GUI frame preview.

## The four tests

| Test | What it answers | Key artifacts |
| --- | --- | --- |
| Link check | Is the adapter on the right AP and is the deck reachable? | `summary.json` |
| Packet test | Where exactly does the stream stall? | `packets.csv`, frames, `partial_frame_payload.bin` |
| Reconnect test | Does a fresh TCP connection deliver one good frame? | `attempt_NN.jpg`, `summary.json` |
| Sustained throughput | Is it *actually* fixed over minutes, not 2 frames? | `frames.csv`, `summary.json` |

The throughput test is the new one. The ported scripts stop at `max_frames=2`
and cannot show a stall that first appears at frame 40.

Each run writes `real_flight/aideck_logs/<test>_<timestamp>/` plus a matching
`.zip` so runs move off the Windows box by whatever route is convenient.

## Flight Watch

`run_flight_watch.bat` is the companion tool for actual test flights. Four
steps: find the drone, pre-flight check, record, result.

**It never commands the drone.** Step 2 reads battery, fitted decks and the
estimator over the Crazyradio, then releases the radio. Step 3 records the
camera only. Two programs cannot share one Crazyradio, so a viewer that kept
hold of it would lock `fly.py` out of the flight.

The camera arrives over the AI Deck's WiFi and flight control goes over the
Crazyradio — separate links, so recording does not compete with flying. The
AI Deck AP was measured on WiFi channel 1 (2412 MHz) and the default
`radio://0/80/2M` sits at 2480 MHz, clear of it.

Needs `cflib` for steps 1 and 2 (`py -3 -m pip install cflib`). Without it
those steps say so and the recording still works.

Each run writes `aideck_logs/flightwatch_<timestamp>/` holding the frames,
`frames.csv` with per-frame timings, `REPORT.md`, and a zip. Times in
`frames.csv` are seconds from the start of recording, so they align with the
`fly.py` flight log by wall-clock offset.

To rehearse without hardware, put `127.0.0.1:5555` in the address box and run
`mock_deck.py --port 5555`.

## Files

- `aideck_core.py` — CPX/image protocol, Windows shell helpers, run folders
- `aideck_tests.py` — the four tests
- `testbed_gui.py` — guided 4-step Tkinter front end, live log, frame preview
- `flight_watch_gui.py` — 4-step flight recorder; reads the drone, then records the camera
- `run_flight_watch.bat` — flight watch launcher
- `testbed_cli.py` — console front end
- `mock_deck.py` — test fixture; a fake deck for validating the test bed
- `run_testbed.bat` — launcher

## Protocol constants

Preserved exactly from the Linux scripts:

- CPX packet header `<HBB` = length, routing, function. Payload is `length - 2`.
- Image header `<BHHBBI` = magic, width, height, depth, type, size.
  Magic `0xBC`, type `1` = JPEG.
- Deck TCP port `5000`. IPs `192.168.4.1` (softAP default), `192.168.7.201`.

## Platform deltas from the Linux scripts

| Linux | Windows |
| --- | --- |
| `ping -c 2 -W 2 <ip>` | `ping -n 2 -w 2000 <ip>` |
| `hostname -I`, `ip -br addr` | `ipconfig /all` |
| `ip route` | `route print -4` |
| — | `netsh wlan show interfaces` for SSID/RSSI/rate |

Console output is decoded utf-8 → cp1252 → cp850 so OEM codepage output does
not crash a run.

## Validating without hardware

`mock_deck.py` replays a real captured deck frame from `aideck_logs` over the
real CPX wire format, and can reproduce the observed failure:

```
py -3 mock_deck.py --port 5555                     # healthy stream
py -3 mock_deck.py --port 5555 --stall-after 2     # frames, then header, then silence
py -3 mock_deck.py --port 5555 --stall-mid-frame 2 # partial second frame, then silence
```

Point a test at it with `--ip 127.0.0.1 --port 5555`.

Note the mock is single-connection: once it stalls it stays wedged and will not
accept a second connection. Restart it between runs.

### Verified on 2026-08-23 (localhost, mock deck — not hardware)

The guided GUI was driven end to end through all four steps, twice:

- **Healthy deck** (mock at 20 fps): found the deck, packet test 3/3 frames,
  reconnect 2/2, throughput 119 frames at 19.70 fps, verdict *"Stream held up.
  No stall."* Session folder, `REPORT.md` and `.zip` all written.
- **Failing deck** (`--stall-after 2`): packet test got 2 frames then stalled,
  reconnect 0/2, and the mock then refused further connections. Verdict *"The
  stream breaks."* naming the wedge and pointing at TXQ16.

The second case caught a real defect: the verdict originally read only the
throughput result, so a deck that wedged before throughput could start was
reported as inconclusive. It now weighs all three tests, and treats reconnect
failures as evidence — on 2026-05-16 the real deck recovered only 1 of 8 fresh
connections, which is the same fault.

Earlier console-level verification of the engine, still current: mid-frame
stall reproduces the May 16 signature exactly — frame 1 complete, second image
header, 2044/4447 bytes, timeout, partial frame saved.

Mock runs were deleted afterwards so `aideck_logs/` holds only real captures.

**Not verified:** anything against real AI Deck hardware. No run in this folder
has touched a deck.

## First action

Run the packet test natively on Windows, on the `aideck-stream` AP, before
flashing anything. Then compare against
`aideck_logs/packet_test_20260516_112123` (the VM/slirp baseline on the same
clean official ESP image).

- If Windows streams cleanly, the stall was slirp and the ESP firmware work is
  chasing a ghost.
- If Windows stalls the same way, slirp is exonerated and TXQ16 is the next
  thing to flash.
