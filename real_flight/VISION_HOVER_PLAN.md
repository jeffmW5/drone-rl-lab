# Onboard Vision Hover — Plan

Date: 2026-08-28
Goal: Crazyflie 2.1 holds position using the AI Deck camera, with the network
running **on the GAP8**, not on a host PC.

## Locked decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Where inference runs | Onboard GAP8 | A robot cannot have WiFi in its control loop. |
| First task | Rise-to-level + distance-hold on a printed square (narrowed 2026-08-29, see below) | Smallest net that still proves the whole chain, and it also fixes the hover instability everything else depends on. |
| Training images | Real captured frames | Chosen over simulator rendering. Track A cleared this. |
| Ground truth | Printed wall marker, pose solved on the host | **No Lighthouse or mocap is owned.** Flow deck gives velocity and height only, so it cannot label absolute position. A marker of known size makes every frame self-labelling. |

## Architecture

```
HM01B0 camera -> GAP8 (int8 CNN) -> CPX/UART over deck connector -> STM32 -> motors
```

The net emits a small vector (position-hold correction), not images. WiFi is
used only for capturing training data and for debugging, never for flight.

### Frame budget

All four "not verified" items from the original 2026-08-28 version of this
section (GAP8 L2 ceiling, inference time N, the capture/encode split as an
onboard measurement, a working reference NN) are now measured. History: the
original number here was a same-day end-to-end guess (136ms/frame, split
inferred as 73ms capture + 58ms encode from an unrelated debugging session);
`GAP8_DORY_RESULT.md` and `GAP8_PERF_RESULT.md` replaced every piece of it
with direct on-chip measurement. That inferred split held up surprisingly
well (73ms vs. a measured 69.4ms, FACT-026) but the model-sizing math below
is now built from the measured numbers, not the inference.

**Measured throughput** (DroNet, 41.1M MACs, as the sizing reference — not a
design target): 114 MMAC/s at 1 core, **644 MMAC/s at 8 cores**, both at
100MHz — the verified-correct clock; 150MHz hangs reproducibly and is not
usable (FACT-025). Use the 8-core number; 1 core cannot clear a useful MAC
budget at any control rate below (see table).

**Measured capture cost** (FACT-026, on-chip `pi_perf`, 324×244 QVGA
grayscale, no JPEG): **69.4ms** with the camera stop/started every frame
(current pattern, e.g. `wifi-img-streamer.c`) — of which ~44ms is stop/start
resync, not pixel transfer. **~25ms** if the capture loop is redesigned to
stream continuously instead (measured standalone only, not yet combined with
real inference in one pipeline).

**MAC budget = (control period − capture cost) × 644 MMAC/s, 8 cores @ 100MHz:**

| Capture pattern | Rate | Period | Budget after capture | MAC budget | Fits DroNet-scale (41.1M)? |
|---|---|---|---|---|---|
| Current (69.4ms, per-frame restart) | 5Hz | 200ms | 130.6ms | 84.1M | yes |
| Current (69.4ms) | 10Hz | 100ms | 30.6ms | 19.7M | no |
| Current (69.4ms) | 15Hz | 66.7ms | — | **capture-bound, infeasible at any model size** | — |
| Continuous (25.1ms, redesign needed) | 5Hz | 200ms | 174.9ms | 112.6M | yes |
| Continuous (25.1ms) | 10Hz | 100ms | 74.9ms | 48.2M | yes |
| Continuous (25.1ms) | 15Hz | 66.7ms | 41.5ms | 26.8M | no |

**Reading this:** 5Hz works today, with the existing capture pattern, at
8 cores, with 2×-DroNet headroom (84.1M MACs). 10Hz needs the capture loop
redesigned to continuous streaming first — that alone is worth more than
switching to 8 cores was. 15Hz is off the table with the current capture
implementation regardless of model size, and even after a capture redesign
only allows a model smaller than DroNet. Size the vision-hover network to
whichever row is the actual target control rate — full detail and
methodology in `GAP8_PERF_RESULT.md`.

## What carries over from ai-grand-prix-stack

Carries: the method. Learning-by-cheating (privileged teacher -> vision
student), the DAgger hard-case loop, shadow-mode promotion gates, and the
staged `image -> features -> policy` architecture from
`docs/AI_GP_VISION_TRANSITION_PLAN.md`.

Does not carry: any trained artifact. There is no CNN in either repo.
`perception/highlighted_gate_detector.py` is classical CV tuned to the DCL
simulator's highlighted gates and will not fire on real hardware.
`policy/mlp_policy.py` takes privileged state on different dynamics.

This is a build, not a port.

## Two independent tracks

Track B does not depend on Track A. Run both.

### Track A — get frames off the deck — **PASSED 2026-08-28**

The stall was VirtualBox slirp. The deck and the official ESP firmware are
fine. Do not flash TXQ16.

Measured on real hardware, `aideck_logs/session_20260828_211353/`:

| Metric | Result |
| --- | --- |
| Sustained throughput | 881 frames / 120.0s, no stall |
| Frame rate | 7.342 fps (mean interval 136ms, median 140ms, max 406ms) |
| Payload | 50.5 KiB/s, ~7.0 KB per frame, 324x244 JPEG |
| Reconnect | 6/6 clean (VM baseline was 1/8) |

Data capture must run from a native host. Never capture through the VM.

Remaining: the exit criterion was a 5-minute hold; this was 2 minutes. Run one
longer capture before trusting it for a full dataset session.

### Track B — prove the GAP8 can run a net at all — **PASSED 2026-08-29**

DroNet (DORY stock `PULP.GAP8` example) ran end-to-end on the physical AI
Deck: 359.9ms/inference (2.78 fps) single-core, final output checksum OK.
See `GAP8_DORY_RESULT.md`. This is far too slow at DroNet's scale/core-count
for the frame budget above — multi-core and/or a smaller model are now the
open design question, not "can it run at all."

That range has since been closed by `GAP8_PERF_PROMPT.md` /
`GAP8_PERF_RESULT.md`: 8 cores gives 5.64x (63.82ms, 15.67 fps), 150MHz hangs
so 100MHz stands, and capture measures 69.4ms as-is or ~25ms if the loop stops
restarting the camera every frame. The Frame budget table above is built from
those numbers and is what the model should be sized against.

<details>
<summary>Original track definition, retained for provenance — all steps completed</summary>

No stream needed. This is the project-killer question and it is cheap.

**The original path for this track was blocked.** It said "flash Bitcraze's
stock NN example." Those examples (facedetection, classification) build through
the GreenWaves AutoTiler, which can no longer be obtained. See "Toolchain
constraint" below. The track was unchanged in purpose; only the toolchain moved.

1. Stand up DORY and its `gap_sdk` backend on the Linux VM.
2. Build and flash a known reference network to the GAP8 via Olimex JTAG,
   using the JTAG path already proven on this bench (per
   `AIDECK_WIFI_CHECKPOINT.md`).
3. Measure inference rate and free L2.
4. Exit criterion: a known network runs on our deck at a measured fps.

Prompt for this work: `real_flight/GAP8_DORY_PROMPT.md`. Result:
`real_flight/GAP8_DORY_RESULT.md`.

One caution from that run is worth carrying forward: the first attempt used a
custom `simple_cnn` rather than a stock example, it crashed on hardware, and
that conflated "our network is wrong" with "the toolchain is wrong" for a full
session (FACT-022). Test toolchains with known-good artifacts.

</details>

## Toolchain constraint — AutoTiler unavailable (recorded 2026-08-29)

The GreenWaves Technologies website is down. The AutoTiler is closed-source and
was never shipped in `gap_sdk`; it was pulled from that site at setup time via
`make autotiler`. Bitcraze's builder image is explicitly "for the GAP8 sdk
(without autotiler)", so Docker does not route around it. With no download
source there is no AutoTiler, and therefore no `GAPflow`/`nntool` deployment.

Bitcraze documents **DORY** (`pulp-platform/dory`) as the alternative NN
deployment path for GAP8. This plan adopts it. DORY is open source, so it does
not carry the same single-vendor availability risk.

Consequences already absorbed above: Track B retargeted, Phase 4 retargeted.

Two risks, both `not verified`:

- **SDK version mismatch.** DORY is tested against `gap_sdk` 3.6 at a pinned
  commit. Bitcraze's tested version for the AI-deck examples is 3.8.1. A single
  tree must satisfy both, and it is not known that one does.
- **Python pinning.** DORY pins Python 3.6.8 and old dependency versions. Use
  DORY's own Dockerfile rather than the VM's system Python.

Worth checking before any of that: if a container on the VM ever ran
`make autotiler` successfully, a copy of the library may already exist locally,
which would unblock the original path at zero cost. Not known either way.

## Where the work runs

DORY itself is portable Python that emits ANSI C. What is not portable is
everything downstream — `gap_sdk` needs the Ubuntu package set and the prebuilt
RISC-V GNU toolchain, then flashes over openocd/JTAG. That chain is Linux only.

| Work | Host | Basis |
| --- | --- | --- |
| DORY codegen, `gap_sdk` build, JTAG flash | Linux VM | Same JTAG path already used to flash ESP32 and GAP8 |
| Frame capture, dataset sessions | Native Windows | Track A; capture through the VM is broken |
| CNN training, ONNX export | Anywhere (Windows or RunPod) | No hardware dependency |

The VM is not disqualified by Track A. That failure was VirtualBox slirp
networking; JTAG is USB passthrough and is unaffected. The capture host and the
build host do not need to be the same machine, and already are not.

## Hardware layout

GAP8 is 9 cores: 1 fabric controller that orchestrates, plus an 8-core compute
cluster. The orchestrator is the 9th core, not one of the 8, so all 8 compute
and there is no core to reclaim. See `GAP8_ARCHITECTURE.md`.

## Simplified first task / integration milestone (narrowed 2026-08-29)

Full 6DOF marker pose was more than the first task needs. Narrowed to two
numbers only:

- **Vertical offset** — how far the marker's center is from vertical-center
  of frame. Drives "rise until level with the square."
- **Apparent size** — the marker's bounding-box size in pixels. A known
  real-world size (currently 4"×4" on paper; 1" did not detect) at a given
  distance maps predictably to pixel size, so this alone drives "adjust
  distance."

**Setup:** drone starts on the floor, facing the marker. Marker is a plain
high-contrast square on paper (black on white), not a fiducial tag. The
original plan was 1"×1". On the 2026-08-29 live calibration a 1" square did
not detect, so a 4"×4" square was used (`marker_calibration.json`,
HYP-MARKER-OTSU). Plain enough that classical CV (threshold → contour →
bounding box) detects it directly; no fiducial library needed.

This marker is deliberate scaffolding, not the final perception target. It
provides known geometry and cheap labels while the camera -> GAP8 -> STM32 ->
motors path is proved. Once that path works, the roadmap moves first to hands
and then to other drones; see `OBJECT_PERCEPTION_PLAN.md`.

**Method — teacher (classical CV) → student (CNN), matching the project's
existing learning-by-cheating pattern** (`docs/AI_GP_VISION_TRANSITION_PLAN.md`):
classical CV run on captured frames on the host generates the (vertical
offset, apparent size) label for free — this *is* the "marker of known size
makes every frame self-labelling" decision above, concretized. The CNN that
ships to the GAP8 learns to reproduce that same output directly from the raw
frame, since the classical detector itself is too fragile/lighting-dependent
to trust as the flight-time detector, and porting OpenCV to GAP8 isn't the
point of this project anyway — the trained net is.

Steps, with status:

| # | Step | Host | Status |
| --- | --- | --- | --- |
| 1 | Marker detector (classical CV: threshold/contour/bbox → vertical offset + size) | Linux VM (portable Python/OpenCV, no hardware needed) | synthetic tests pass and a 4 in marker detects over a narrow live range. Next: save failure frames and replace external-only/largest-box selection with hierarchy-aware or connected-component selection plus a rotation-stable size measure. |
| 2 | Distance calibration — apparent size ↔ real distance, a few known-distance reference shots | Windows (native host) | **not yet accepted.** The 2026-08-29 run had 3 samples over only 1.49x span. The 2026-08-30 run had only 2 samples over 1.47x; its 0.4826 m sample required about 30 deg marker/camera angle and is not valid for the face-on model. Combined runs suggest, but do not prove, a roughly 10-12 cm distance-datum offset. Use a face-on 4 in marker, measure from the lens plane, save raw frames, and target roughly 0.30/0.45/0.60/0.75/0.90 m after the detector fix. See `marker_calibration.json` and Git history for the overwritten first run. |
| 3 | Synchronized capture — frames (+ telemetry where useful) during flight | Windows (native host) | may largely reuse `windows_testbed/flight_watch_gui.py` already; check before building new |
| 4 | Auto-label captured frames via step 1's detector | Either host | depends on step 1 |
| 5 | Train tiny CNN (image → vertical offset + size), P-controller converts to thrust/pitch initially | Windows (1070Ti) or Linux VM | depends on step 4 |
| 6 | Deploy via the DORY pipeline already proven (`GAP8_DORY_RESULT.md`) | Linux VM | depends on step 5 |

Windows-side wizards (calibration + capture) requested via a prompt for the
Windows agent — see `WINDOWS_MARKER_WIZARDS_PROMPT.md`.

**Streamer XOR net.** The GAP8 runs `wifi-img-streamer` or a DORY network,
never both — one hyperflash image. Steps 2 and 3 need the streamer; step 6
needs the net. So the teacher->student loop crosses a JTAG round-trip in each
direction, on the Linux box, for work that is otherwise Windows-side.

This bit already: after Track B and the perf session flashed three networks in
turn, `aideck-stream` stopped broadcasting and blocked calibration. Confirmed
2026-08-29 with the drone powered on — 74 BSSIDs scanned, zero Espressif APs in
range. Unblock prompt: `GAP8_RESTORE_STREAMER_PROMPT.md`. If the swap becomes
the bottleneck for dataset iteration, the answer is one app that both streams
and infers.

## Phases after both tracks pass

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 | Stable real hover on the flow deck | Holds position 30s, no wall contact. Current state: drifted into a wall 2026-04-20 — expected, the flow deck has no absolute reference. Narrowed first task above runs independently of this — it doesn't need 30s stability, just enough manual/`fly.py` flight time to capture varied frames. |
| 2 | Synchronized capture — frames + telemetry, timestamp-aligned | A dataset of flights with per-frame pose |
| 3 | Auto-label offline from the marker's apparent size and skew | Labeled set, held-out split |
| 4 | Train small int8 CNN, deploy via DORY | Runs on GAP8 within the Track B budget |
| 5 | Shadow mode — net infers while the flow deck flies | Net output tracks the marker-derived pose |
| 6 | Bounded live — net closes the loop, flow deck as fallback | Holds position on camera alone |
| 7 | Generalize beyond the marker | Hand detection first, then nearby-drone detection/tracking under the promotion gates in `OBJECT_PERCEPTION_PLAN.md`. |

## Beyond the marker

The end target is learned object perception, not increasingly elaborate
thresholding of a printed square. Hands are the first target because they are
larger and easier to collect/label; nearby drones follow once the complete
onboard inference and safety path is proven. Object perception changes the
student output from `(vertical offset, marker size)` to class confidence and a
bounding box, with temporal tracking and a separately validated range signal.
Unknown-size hands and attitude-varying drones cannot reuse the marker's
inverse-size distance equation without qualification. The data, model,
distance, safety, and promotion plan is `OBJECT_PERCEPTION_PLAN.md`.

## Open items carried in from the existing stack

- Hover P gains untuned (`kp_xy=8.0`, `kp_z=8000` drifted into a wall)
- Obs mismatch: exp_069 wants 55D, constructed obs is 62D, currently truncated
- `fly` modes never tested on hardware

## What this plan does NOT claim

- No vision model has been trained. None exists in either repo.
- No task-specific marker, hand, or drone vision model has been trained or run
  on the GAP8. DORY itself and a stock DroNet inference have been verified on
  this bench; the open question is the task model and integrated control loop.
- The sim RL line has passed one gate, once (exp_028). It is not a usable
  teacher for racing and is not on this plan's critical path.

Superseded claim: an earlier version of this document said "Nothing in
`windows_testbed/` has touched a real AI Deck." That was true when written and
is now false — the Windows test bed produced the Track A hardware result in
`aideck_logs/session_20260828_211353/`.

## Sources for the toolchain constraint

- https://www.bitcraze.io/documentation/repository/aideck-gap8-examples/master/development/gap8/
- https://github.com/bitcraze/docker-aideck
- https://github.com/pulp-platform/dory
- https://github.com/GreenWaves-Technologies/gap_sdk
