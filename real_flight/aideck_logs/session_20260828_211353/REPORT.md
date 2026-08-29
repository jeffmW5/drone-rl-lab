# AI Deck WiFi test session

Run: 2026-08-28T21:16:28
Target: 192.168.4.1:5000
Mode: real hardware

## Verdict

Stream held up. No stall.

Packet test: 3 frame(s). Reconnect: 6/6 clean. Throughput: 881 frames in 120s (7.342 fps).

This PC does not use the VirtualBox NAT path. A clean run here against the same firmware that stalls in the VM means the VM's network stack was the problem, not the deck. Do not flash the TXQ16 image on the old VM logs alone.

## Numbers

- Packet test: 3 frames, 30 packets, stalled=False
- Reconnect test: 6/6 clean
- Sustained throughput: 881 frames in 120.0s, 7.342 fps, 50.5 KiB/s, stalled=False
