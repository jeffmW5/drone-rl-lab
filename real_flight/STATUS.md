# Real Flight Deployment — Status

Last updated: 2026-05-16 (consolidated from sessions through 2026-04-26)

## Overview

Deploying trained RL policies from MuJoCo sim to a real Crazyflie 2.1 drone. Uses cflib directly (no ROS2/Vicon). Positioning via Flow Deck v2 only — no Lighthouse hardware. Flow deck gives velocity and height, so position drifts over time. Best sim checkpoint so far: exp_069 (2x128 network, first deterministic gate passages).

## Hardware

| Component | Details |
|-----------|---------|
| Drone | Crazyflie 2.1 (cf21B_500, 43.4g) |
| Firmware | 2025.12.1 (protocol v10) — rolled back from 2026.04 (broke Flow Deck) |
| Radio | Crazyradio PA USB dongle, `radio://0/80/2M` |
| Positioning | Flow Deck v2 (bottom connector) — **only** option. No Lighthouse owned (confirmed 2026-08-28). |
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

- **cflib-only** — avoids ROS2 + Vicon dependency. Flies on the Flow Deck alone.
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

### Status: WORKING natively on Windows (2026-08-28)

Resolved. The stall was **VirtualBox slirp**, not the deck and not the ESP
firmware. A native Windows client on `aideck-stream` ran 120s of sustained
throughput with **881 frames, 7.342 fps, zero stalls**, and **6/6 clean
reconnects**, on the same clean official `58f15fa` ESP image that stalls after
one frame inside the VM.

- Evidence: `aideck_logs/session_20260828_211353/`
- Tool: `real_flight/windows_testbed/run_testbed.bat`
- **Do not flash TXQ16.** See `AIDECK_WIFI_CHECKPOINT.md`.

Capture the drone's camera data from a native host, never from the VM.

### Toolchain constraint: AutoTiler unavailable (2026-08-29)

The GreenWaves website is down, so the closed-source AutoTiler can no longer be
fetched. It was never shipped inside `gap_sdk`; `make autotiler` pulled it from
that site. Bitcraze's builder image is "without autotiler", so Docker does not
route around it. Bitcraze documents **DORY** as the alternative NN deployment
path, and the vision plan has adopted it.

This blocks Bitcraze's stock NN examples (facedetection, classification) and
the `GAPflow`/`nntool` route. It does **not** affect the WiFi stream work
above — `wifi-img-streamer` builds without the AutoTiler, which is why the
existing GAP8 flashes succeeded and why they prove nothing about running a net.

See `memory/FACTS.md` FACT-020 and `VISION_HOVER_PLAN.md`.

### Historical: the VM-era diagnosis (superseded)

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

- [x] Stable hover achieved 2026-08-28 via `fly.py takeoff` (firmware high-level
      commander). 0.374m peak, 7.0s airborne, 4.4cm max drift, 0.075 m/s lateral.
- [ ] `hover` mode remains broken and is superseded by `takeoff`. Its linear
      `thrust_to_pwm` under-commands hover thrust (34861 PWM), so it reached only
      8.5cm and skidded 1.11m at 1.42 m/s. Fix or retire it.
- [ ] Fix obs format mismatch (get `train_race.py` from RunPod or print obs space)
- [ ] Complete progressive flight testing (hover stable -> fly --no-gates -> fly)
- [ ] Measure real gate positions and update config.yaml (currently nominal level2.toml)
- [x] Unblock AI Deck camera stream — was slirp, works natively on Windows (2026-08-28)
- [ ] Sim-to-real tuning (thrust scaling, attitude response lag, position drift)
- [ ] Wait for exp_071+ results (obs normalization, action smoothness) for better checkpoints
- [x] **GAP8 performance envelope.** DONE 2026-08-29. 8 cores: 63.82ms/inference
      (15.67 fps, 5.64× over 1 core), checksum OK. Clock: 150MHz hangs
      reproducibly (3/3 attempts, voltage ruled out) — 100MHz remains the
      verified ceiling, despite `gap_sdk` spec headers saying cluster should
      reach 175MHz. Capture: measured 69.4ms on-chip (confirms the old 73ms
      figure) with the current per-frame camera restart pattern — ~44ms of
      that is stop/start overhead, not pixel transfer; continuous streaming
      measures ~25ms instead. MAC-budget table now in `VISION_HOVER_PLAN.md`.
      See `GAP8_PERF_RESULT.md`.
- [ ] **Phase 1 hover — 30s held, no wall contact.** Best so far 7.0s via
      `fly.py takeoff`. Independent of the GAP8 work; runs in parallel.
- [x] **Track B — can the GAP8 run a net at all?** DONE 2026-08-29. Yes.
      DroNet (DORY's own `PULP.GAP8` stock example) runs end-to-end on the
      physical AI Deck: 359.9ms/inference (≈2.78 fps) single-core, final
      output checksum OK. Toolchain fully proven: DORY codegen → `gap_sdk`
      build (`ai_deck`/`GAP8_V2` target) → JTAG flash → JTAG boot, all clean.
      (Note for next time: `configs/ai_deck.sh` assumes the wrong Olimex
      adapter — needs `GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg`
      override; only `bitcraze/aideck` Docker's OpenOCD has the custom `gap8`
      target driver, a locally-built vanilla openocd cannot flash/run GAP8 at
      all.) An earlier custom test network (`simple_cnn` in `~/projects/gap_sdk/dorytest`)
      crashed on hardware — confirmed to be that network's own bug, not the
      toolchain, since DroNet runs clean on the identical setup. 359.9ms is
      far above the ~13.7fps ceiling `VISION_HOVER_PLAN.md`'s frame budget
      assumes — multi-core (untried, GAP8 has 8 cluster cores) or a smaller
      model will be needed for real vision-hover. See `GAP8_DORY_RESULT.md`,
      `memory/FACTS.md` FACT-023.
- [ ] Finish Track A's stated exit criterion. It called for a 5-minute sustained
      hold; only 120s was run. One long capture before trusting it for a dataset session.

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
