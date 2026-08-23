# AI Deck WiFi Test Bed (Windows)

Native Windows implementation of `real_flight/WINDOWS_TESTBED_PROMPT.md`.

Purpose: run the AI Deck stream diagnostics off a Windows box joined to the
`aideck-stream` AP, with no VirtualBox NAT (slirp) in the data path. Every
packet test logged before this ran through slirp, so the observed stall is
consistent with both an ESP TX-queue problem and a slirp receive stall.
**Which one it is, is `not verified`.** Settle that before flashing the TXQ16
image.

## Running it

```
run_testbed.bat
```

Console form:

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

## Files

- `aideck_core.py` — CPX/image protocol, Windows shell helpers, run folders
- `aideck_tests.py` — the four tests
- `testbed_gui.py` — Tkinter front end, live log, frame preview
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

Re-run end to end after the `run_testbed.bat` launcher was corrected:

- Packet test, healthy mock: 3/3 frames, 18 packets, `stall=False`. All three
  saved frames decode as 324x244 grayscale JPEG.
- Throughput, healthy mock at 20 fps: 198 frames in 10.007s = 19.787 fps,
  86.61 KiB/s, frame interval median 0.0506s, `stalled=False`.
- Reconnect: 3/3 attempts each delivered one complete 4447-byte frame.
- Stall detection, `--stall-after 2`: `stalled=True` after 2 frames.
- Mid-frame stall, `--stall-mid-frame 2`: frame 1 complete, second image header
  parsed, then 2044/4447 bytes and timeout, partial frame written. This is the
  same signature as the real May 16 logs.
- Link check against a dead port: ping replies, TCP correctly reported refused.
- Zip output and the Tkinter GUI both build and render.

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
