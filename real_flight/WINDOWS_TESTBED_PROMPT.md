# Prompt: AI Deck Windows Test Bed

Date written: 2026-08-22
Status: **built 2026-08-23** in `real_flight/windows_testbed/`. See that
folder's `README.md`. Verified against a mock deck on localhost only; `not
verified` against real AI Deck hardware.

## Goal

Build a self-contained Windows app that connects to the AI Deck WiFi stream,
runs the existing diagnostics, writes logs, and can be re-run repeatedly while
iterating on firmware.

## Why

Two problems with the current workflow.

1. **WiFi switching.** The AI Deck runs as an AP named `aideck-stream`. Testing
   means taking the dev machine off the internet, so the Claude Code session
   dies mid-debug. A separate Windows box on `aideck-stream` permanently
   decouples the test rig from the dev machine.

2. **Suspected VirtualBox NAT confound.** The Linux VM has one interface,
   `enp0s3` at `10.0.2.15/24`, default via `10.0.2.2` — VirtualBox NAT (slirp).
   Every packet test logged so far ran through the slirp user-mode TCP stack.
   The observed failure (one clean frame, then the next image header, then
   nothing) is consistent with an ESP TX-queue problem *and* with a
   receiver-side buffer stall in slirp. `not verified` which one it is.

   A native Windows run removes slirp from the data path. If the stream is
   clean on Windows, the ESP firmware work is chasing a ghost and the TXQ16
   image should not be flashed.

## What to build

A test bed that runs offline on Windows with no internet and no pip installs.

Tests, all ported from the existing scripts rather than reinvented:

- **Link check** — adapter state, IP, route, ping to the deck. Sanity gate
  before a real run.
- **Packet test** — port of `aideck_packet_test.py`. CPX packet headers,
  payload sizes, image-header metadata, inter-packet gaps, frame completion,
  and the exact stall point. Writes `packets.csv`, `summary.json`, saved
  frames, partial-frame payload.
- **Reconnect test** — port of `aideck_reconnect_test.py`. N fresh TCP
  connections, one complete frame each, pass/fail per attempt.
- **Sustained throughput** — longer continuous read with FPS and byte rate.
  This is the "is it actually fixed" test; the existing scripts stop at
  `max_frames=2` and cannot show a stall that appears at frame 40.

Each run writes a timestamped folder plus a zip, so runs move off the Windows
box by whatever route is convenient.

## Porting notes

`aideck_packet_test.py` and `aideck_reconnect_test.py` are already stdlib only
(`socket`, `struct`, `csv`, `json`, `logging`, `subprocess`, `pathlib`). The
protocol logic ports unchanged. Platform-specific deltas:

| Linux | Windows |
| --- | --- |
| `ping -c 2 -W 2 <ip>` | `ping -n 2 -w 2000 <ip>` |
| `hostname -I` | `ipconfig /all` |
| `ip -br addr` | `ipconfig` |
| `ip route` | `route print` |
| — | `netsh wlan show interfaces` for SSID/RSSI/rate |

Also: shebangs are `#!/usr/bin/env python3.11`, and the type hints use `dict |
None`, so the target needs Python 3.10+.

Protocol constants to preserve exactly:

- CPX packet header: `<HBB` = length, routing, function. Payload is
  `length - 2` bytes.
- Image header: `<BHHBBI` = magic, width, height, depth, type, size.
  Magic `0xBC`. Type `1` = JPEG.
- Deck TCP port `5000`. IP candidates `192.168.4.1` (ESP softAP default) and
  `192.168.7.201` (appears in `stream_viewer.py`, suggests a prior station-mode
  run).

## Open decisions -- resolved 2026-08-23

Checked on the Windows dev machine (`py -0p`): Python 3.14, 3.13 and 3.11 are
installed, tkinter 8.6 is present, and Pillow 12.1.1 is already installed.

- **Python installed?** Yes. Ships as a folder plus `run_testbed.bat`. No
  PyInstaller `.exe` needed, so the cross-compile problem does not arise.
- **Tkinter vs console?** Both. `testbed_gui.py` is the Tkinter front end with a
  live JPEG preview; `testbed_cli.py` is the console form for headless use. The
  preview still degrades to "frame saved" text if Pillow is absent elsewhere.
- **How runs come back?** The repo checkout lives at
  `C:\Users\JefferyWhitmire\Desktop\Shared\drone-rl-lab`, so this Windows box is
  the VM host and the shared folder is available. Runs default to
  `real_flight/aideck_logs/` and are also zipped. Output folder is selectable in
  the GUI and via `--out-dir`.

Still open: whether the test bed should live on this dev machine or a second
Windows box. Running it here still means taking this machine off the internet to
join `aideck-stream`, which is one of the two problems the spec set out to fix.

## Related

- `AIDECK_WIFI_CHECKPOINT.md` — flashed state, prior evidence, the built but
  unflashed TXQ16 image, and the GAP8 fallback plan.
- Longer term, station mode removes AP switching entirely:
  `wifi-img-streamer.c` `setupWiFi()` hardcodes `ssid = "aideck-stream"` and
  sends `WIFI_CTRL_WIFI_CONNECT` with `data[0] = 0x01` (AP). Setting `0x00`
  selects `wifi_init_sta()` in the ESP firmware's `wifi.c`. That needs a GAP8
  reflash and keeps slirp in the path, so it does not replace this test bed.

## First action

Run the packet test natively on Windows before flashing anything. Settle the
slirp question first.
