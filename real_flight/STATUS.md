# Real Flight Deployment — Status

Last updated: 2026-05-16 (consolidated from sessions through 2026-04-26)

## Overview

Deploying trained RL policies from MuJoCo sim to a real Crazyflie 2.1 drone. Uses cflib directly (no ROS2/Vicon). Positioning via Lighthouse V2 or Flow Deck v2. Best sim checkpoint so far: exp_069 (2x128 network, first deterministic gate passages).

## Hardware

| Component | Details |
|-----------|---------|
| Drone | Crazyflie 2.1 (cf21B_500, 43.4g) |
| Firmware | 2025.12.1 (protocol v10) — rolled back from 2026.04 (broke Flow Deck) |
| Radio | Crazyradio PA USB dongle, `radio://0/80/2M` |
| Positioning | Lighthouse V2 (primary), Flow Deck v2 (backup, bottom connector) |
| AI Deck | Top connector, ESP32 Nina W102 + GAP8 |
| JTAG | Olimex ARM-USB-TINY-H + ARM-JTAG-20-10 adapter |
| VM USB | VirtualBox passthrough, udev rules at `/etc/udev/rules.d/99-bitcraze.rules` |

## Software

| What | Version/Path |
|------|-------------|
| cflib | 0.1.32 on Python 3.11 |
| torch | 2.10.0+cpu on Python 3.11 |
| OpenOCD (ESP32) | `~/Downloads/openocd-esp32/` |

## Deployment Pipeline

### Files

| File | Purpose |
|------|---------|
| `fly.py` | Main script — `check`, `hover`, `fly` modes |
| `config.yaml` | Radio, safety limits, gate positions, drone params |
| `analyze.py` | Post-flight log visualization (4-panel plot) |
| `stream_viewer.py` | AI Deck WiFi camera stream viewer (PIL-based) |
| `PILOT_AGENT.md` | Safety guide for AI agents operating the drone |
| `logs/` | Flight logs (3 logs from 2026-04-20) |

### Architecture decisions

- **cflib-only** — avoids ROS2 + Vicon dependency. Fly with just Lighthouse or Flow Deck.
- **Auto-detect policy architecture** from checkpoint weight shapes (obs_dim, hidden_size, act_dim).
- **Safety-first sequence**: PID takeoff -> hover stabilize -> RL handoff. Ctrl+C -> safe landing. Geofence/battery/variance violations -> emergency stop.
- **State estimation** via cflib log framework (`stateEstimate.x/y/z`, `stateEstimate.qx/qy/qz/qw`, `gyro.x/y/z`) at 100Hz. Policy runs at 50Hz.
- **Log vars fix**: `stateEstimate.qx/qy/qz/qw` (not `stabilizer.qx` — doesn't exist in this firmware). Attitude+gyro split into separate LogConfig groups (26-byte cflib packet limit).

### Progressive test sequence

```
check  ->  hover  ->  fly --no-gates  ->  fly
```

Each step must succeed before advancing.

## What Has Been Tested

### Flight tests (2026-04-20)

Three flight logs recorded (`logs/flight_20260420_*.npz`).

- `check` mode: working — radio, battery, position, quaternion, variance all readable
- `hover` mode: **flew but drifted into wall** — P controller gains (kp_xy=8.0, kp_z=8000) too aggressive or drift issue. Hover rewritten to low-level `send_setpoint` commands. Kalman estimator reset on connect. Geofence z_min relaxed to -0.5m, variance threshold to 5.0 for Flow Deck.
- `fly` modes: not tested yet

### Firmware fixes applied

- Kalman estimator reset on connect
- Hover rewritten to low-level `send_setpoint`
- Geofence z_min relaxed to -0.5m
- Position variance threshold raised to 5.0 (for Flow Deck)
- Log var split for 26-byte packet limit

## Known Issue: Observation Format Mismatch

exp_069 expects **55D** input. The constructed obs (pos/quat/vel/ang_vel/target_gate/gates_pos/gates_quat/gates_visited/obstacles_pos/obstacles_visited) flattens to **62D**. The 7D difference comes from `make_race_envs` in `lsy_drone_racing/control/train_race.py` which only exists on the RunPod GPU server.

**Workaround**: auto-truncate 62D to 55D (first 55 dims = drone state + gate info).

**Proper fix**: Copy `train_race.py` from RunPod, or run there:
```python
envs = make_race_envs(config="level2_attitude.toml", num_envs=1, ...)
obs, _ = envs.reset()
print(obs.shape, envs.single_observation_space)
```

## AI Deck WiFi Camera Stream

### Status: Stalled — known Bitcraze bug

**Firmware flashed:**
- ESP32: 2025.02 via JTAG (`esp32-solo-1.cfg`)
- GAP8: 2025.02 via JTAG (Docker build + flash)
- GAP8 modified: JPEG encoding enabled, WiFi AP mode, SSID `aideck-stream`

**What works:**
- ESP32 softAP broadcasts, Windows host connects, VM gets IP on 192.168.4.x
- TCP connects to 192.168.4.1:5000
- GAP8 captures JPEG frames (~5-6KB, 73ms capture + 58ms encode)
- First CPX packet arrives (0xBC magic, 324x244 JPEG header)
- GAP8 console confirms continuous frame production

**What's broken:**
- After the first CPX packet (image header), **no more data arrives over TCP**
- JPEG data/footer packets never reach the client
- Matches **Bitcraze GitHub issue #150** — known stream freeze bug
- Root cause suspected: ESP32 wifiTxQueue depth is only 2

**Viewers built:**
| File | Notes |
|------|-------|
| `real_flight/stream_viewer.py` | PIL-based, auto-reconnect, frame save option |
| `~/aideck_viewer.py` | Pygame standalone with desktop launcher |
| `~/test_aideck_stream.py` | Diagnostic test script |

**JTAG flash commands (reference):**
```bash
# ESP32
~/Downloads/openocd-esp32/bin/openocd \
  -s ~/Downloads/openocd-esp32/share/openocd/scripts \
  -f interface/ftdi/olimex-arm-usb-tiny-h.cfg \
  -f board/esp32-solo-1.cfg \
  -c "adapter speed 1000" \
  -c "program_esp /tmp/esp-fw/aideck_esp.bin 0x10000 verify reset exit"

# GAP8
cd /tmp/gap8-2025.02
docker run --rm -v ${PWD}:/module --device /dev/ttyUSB0 --privileged -P \
  bitcraze/aideck tools/build/make-example examples/other/wifi-img-streamer \
  "clean all image flash"
```

**Next steps to unblock stream:**
1. Try official `opencv-viewer.py` to rule out viewer bug
2. If official viewer also stalls -> ESP32 firmware problem, try older version
3. Add inter-packet delays in GAP8 firmware (`pi_time_wait_us()` between `cpxSendPacketBlocking`) to avoid overwhelming 2-slot TX queue

## Open Items

- [ ] Tune hover P controller gains (drifted into wall on first flight)
- [ ] Fix obs format mismatch (get `train_race.py` from RunPod or print obs space)
- [ ] Complete progressive flight testing (hover stable -> fly --no-gates -> fly)
- [ ] Measure real gate positions and update config.yaml (currently nominal level2.toml)
- [ ] Unblock AI Deck camera stream (Bitcraze bug #150)
- [ ] Sim-to-real tuning (thrust scaling, attitude response lag, position drift)
- [ ] Wait for exp_071+ results (obs normalization, action smoothness) for better checkpoints

## File Locations (full reference)

| What | Where |
|------|-------|
| Deployment scripts | `drone-rl-lab/real_flight/` |
| Training configs | `drone-rl-lab/configs/exp_*.yaml` |
| Trained checkpoints | `drone-rl-lab/results/exp_*/model.ckpt` |
| Best checkpoint | `results/exp_069_larger_network/model.ckpt` (128 hidden, 55D) |
| Drone model params | `~/.local/lib/python3.11/site-packages/drone_models/data/params.toml` |
| cfclient config | `~/.config/cfclient/config.json` |
| Real env (needs ROS2) | `~/.local/lib/python3.11/site-packages/lsy_drone_racing/envs/real_race_env.py` |
| Level2 config | `~/.local/lib/python3.11/site-packages/config/level2_attitude.toml` |
| GAP8 source (modified) | `/tmp/gap8-2025.02/examples/other/wifi-img-streamer/` |
| ESP32 binary | `/tmp/esp-fw/aideck_esp.bin` |
| OpenOCD | `~/Downloads/openocd-esp32/` |
| Olimex udev rules | `/etc/udev/rules.d/99-olimex.rules` |
| Bitcraze udev rules | `/etc/udev/rules.d/99-bitcraze.rules` |
