# AI Deck WiFi Stream Checkpoint

Date: 2026-05-16

## RESOLVED 2026-08-28 — it was slirp, not the deck

Superseding everything below. A native Windows client on `aideck-stream`, with
no VirtualBox NAT in the path, streamed **881 frames over 120s with zero
stalls** and reconnected **6/6** — on the same clean official `58f15fa` ESP
image that stalls after one frame in the VM.

- Evidence: `aideck_logs/session_20260828_211353/`
- VM baseline it contradicts: `aideck_logs/packet_test_20260516_112123`
  (one frame then stall) and `reconnect_test_20260516_085408` (1/8)

**Do not flash the TXQ16 image.** `WIFI_HOST_QUEUE_LENGTH (2)` was never the
problem. The image stays archived in
`firmware/esp_official_58f15fa_txq16_20260516/` but the "Next Resume Step"
below is withdrawn, and the GAP8 patches suggested under "If TXQ16 Still Fails"
are not justified by any current evidence.

Bitcraze issue #150 is not implicated on this hardware. What was actually
observed was a VirtualBox slirp receive stall.

The sections below are kept as the record of how this was diagnosed. Treat
their conclusions as superseded.


## Current Flashed State

The AI Deck ESP32 currently has clean official Bitcraze ESP firmware flashed:

- Flash log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/flash_20260516_111833/flash.log`
- Flashed app image: `/home/jeff/drone-rl-lab/real_flight/firmware/esp_official_58f15fa_20260516/aideck_esp.bin`
- SHA256: `7f76772a453cc99766d01520a11a426329b6de832522b4640d69f5b8d99ce980`
- Source commit: `58f15fa5e14ec888f9ee965b6236aaf6d5c9e269`

This official image includes Bitcraze's current larger TCP send buffer:

- `CONFIG_LWIP_TCP_SND_BUF_DEFAULT=65535`
- `CONFIG_TCP_SND_BUF_DEFAULT=65535`

## Evidence So Far

Baseline before ESP changes:

- Log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/packet_test_20260516_084711`
- Result: one complete JPEG frame arrived, then the second image metadata header arrived, then the stream stalled before the second frame JPEG payload.

Reconnect test:

- Log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/reconnect_test_20260516_085408`
- Result: 1 successful frame out of 8 reconnect attempts. Reconnect did not reliably recover the stream.

First patched ESP build:

- Log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/packet_test_20260516_092630`
- Result: only the image metadata header arrived, with zero JPEG payload bytes.

Second patched ESP build:

- Log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/packet_test_20260516_094014`
- Result: image header plus 1326/4361 JPEG bytes arrived, then timeout while reading a CPX packet payload.

Clean official ESP build, currently flashed:

- Log: `/home/jeff/drone-rl-lab/real_flight/aideck_logs/packet_test_20260516_112123`
- Result: one complete JPEG frame arrived, including footer, then the second image metadata header arrived, then no JPEG payload for the second frame.

This means Bitcraze's latest TCP send-buffer config alone is not enough on this setup, but the overly broad ESP transport patch was worse. The best current candidate is a smaller ESP patch that only increases the WiFi host queue depth.

## Built But Not Flashed Yet

A queue-depth-only ESP image has been built from clean official Bitcraze source with exactly this source change:

```diff
-#define WIFI_HOST_QUEUE_LENGTH (2)
+#define WIFI_HOST_QUEUE_LENGTH (16)
```

Candidate image:

- Source: `/home/jeff/bitcraze-src/aideck-esp-firmware-official`
- Changed file: `/home/jeff/bitcraze-src/aideck-esp-firmware-official/main/wifi.c`
- App image: `/home/jeff/drone-rl-lab/real_flight/firmware/esp_official_58f15fa_txq16_20260516/aideck_esp.bin`
- SHA256: `5ddaa2799633ec9938042d164dbe1c11ecc65cfb6157ecf64a3d85542469022d`

This image has not been flashed yet.

## Next Resume Step

When hardware is connected again, flash the queue-depth-only ESP image:

```bash
/home/jeff/Downloads/openocd-esp32/bin/openocd \
  -s /home/jeff/Downloads/openocd-esp32/share/openocd/scripts \
  -f interface/ftdi/olimex-arm-usb-tiny-h.cfg \
  -f board/esp32-solo-1.cfg \
  -c "adapter speed 1000" \
  -c "program_esp /home/jeff/drone-rl-lab/real_flight/firmware/esp_official_58f15fa_txq16_20260516/aideck_esp.bin 0x10000 verify reset exit"
```

Hardware setup:

- Propellers off.
- Crazyflie/AI Deck powered.
- Olimex ARM-USB-TINY-H connected to the AI Deck ESP debug header.
- Olimex USB connected to the VM/host.
- Internet WiFi can stay connected during flashing.

After flash:

1. Power-cycle the Crazyflie/AI Deck.
2. Switch WiFi to `aideck-stream`.
3. Run the desktop launcher `AI Deck Packet Test`.
4. Switch back to internet WiFi.
5. Inspect the newest `/home/jeff/drone-rl-lab/real_flight/aideck_logs/packet_test_*`.

## If TXQ16 Still Fails

Move to GAP8-side work. The likely GAP8 source that produced the current `aideck-stream` JPEG streamer is:

- `/home/jeff/Downloads/aideck-gap8-examples-master/examples/other/wifi-img-streamer/wifi-img-streamer.c`

Suspicious GAP8 details:

- It sends a CRTP console print after every frame even though the intended `OUTPUT_PROFILING_DATA` guard is commented out.
- The official clean clone defaults to RAW mode and a different AP name, so do not blindly flash that clone as-is.
- The likely first GAP8 patch is to disable the per-frame `cpxPrintToConsole()` call and fix the `memcpy()` in `sendBufferViaCPX()` to copy `size`, not `sizeof(packet->data)`.

Docker builder images were pruned after build to recover disk space. No OpenOCD/build/test process should be running at this checkpoint.
