# Onboard Vision Hover — Plan

Date: 2026-08-28
Goal: Crazyflie 2.1 holds position using the AI Deck camera, with the network
running **on the GAP8**, not on a host PC.

## Locked decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Where inference runs | Onboard GAP8 | A robot cannot have WiFi in its control loop. |
| First task | Vision-only hover / station-keep | Smallest net that still proves the whole chain, and it also fixes the hover instability everything else depends on. |
| Training images | Real captured frames | Chosen over simulator rendering. Track A cleared this. |
| Ground truth | Printed wall marker, pose solved on the host | **No Lighthouse or mocap is owned.** Flow deck gives velocity and height only, so it cannot label absolute position. A marker of known size makes every frame self-labelling. |

## Architecture

```
HM01B0 camera -> GAP8 (int8 CNN) -> CPX/UART over deck connector -> STM32 -> motors
```

The net emits a small vector (position-hold correction), not images. WiFi is
used only for capturing training data and for debugging, never for flight.

### Frame budget

Measured end-to-end on 2026-08-28: 136ms mean per frame (7.342 fps) with JPEG
streaming enabled. `STATUS.md` documents the GAP8 side as 73ms capture + 58ms
encode = 131ms, which matches within 4%.

If that attribution holds (see `memory/HYPOTHESES.md` HYP-AIDECK-RATE), onboard
inference drops the ~58ms JPEG encode, leaving:

```
73ms capture + N ms inference = control rate
```

So ~13.7 fps is the ceiling before the net runs, and N sets everything below
that. Design the model to a measured N, not to a guess.

**Not verified:** GAP8 L2 memory ceiling, actual inference time N on the
flashed firmware, the capture/encode split as an onboard measurement, and the
current Bitcraze reference NN example name. All four hard-constrain the model.

**Update 2026-08-29:** Track B measured a real N — DroNet (comparable size:
15 layers, 41M MACs) runs in 359.9ms single-core (2.78 fps) on this exact
GAP8/AI Deck. That is far above the ~73ms capture floor and blows the ~13.7fps
budget above on its own. Multi-core (untried, 8 cluster cores available) or a
substantially smaller model than DroNet-scale will be required for onboard
vision-hover to hit a usable control rate. See `GAP8_DORY_RESULT.md`.

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

No stream needed. This is the project-killer question and it is cheap.

**The original path for this track is blocked.** It said "flash Bitcraze's
stock NN example." Those examples (facedetection, classification) build through
the GreenWaves AutoTiler, which can no longer be obtained. See "Toolchain
constraint" below. The track is unchanged in purpose; only the toolchain moves.

1. Stand up DORY and its `gap_sdk` backend on the Linux VM.
2. Build and flash a known reference network to the GAP8 via Olimex JTAG,
   using the JTAG path already proven on this bench (per
   `AIDECK_WIFI_CHECKPOINT.md`).
3. Measure inference rate and free L2.
4. Exit criterion: a known network runs on our deck at a measured fps.

If that number is low, the model design and possibly the whole task change.
Measure it before designing anything.

Prompt for this work: `real_flight/GAP8_DORY_PROMPT.md`.

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

## Phases after both tracks pass

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 | Stable real hover on the flow deck | Holds position 30s, no wall contact. Current state: drifted into a wall 2026-04-20 — expected, the flow deck has no absolute reference. |
| 2 | Synchronized capture — frames + telemetry, timestamp-aligned | A dataset of flights with per-frame pose |
| 3 | Auto-label offline from the marker's apparent size and skew | Labeled set, held-out split |
| 4 | Train small int8 CNN, deploy via DORY | Runs on GAP8 within the Track B budget |
| 5 | Shadow mode — net infers while the flow deck flies | Net output tracks the marker-derived pose |
| 6 | Bounded live — net closes the loop, flow deck as fallback | Holds position on camera alone |
| 7 | Remove the marker | Hover on learned visual features alone |

## Open items carried in from the existing stack

- Hover P gains untuned (`kp_xy=8.0`, `kp_z=8000` drifted into a wall)
- Obs mismatch: exp_069 wants 55D, constructed obs is 62D, currently truncated
- `fly` modes never tested on hardware

## What this plan does NOT claim

- No vision model has been trained. None exists in either repo.
- No network of any kind has been run on the GAP8. Track B is unstarted, and
  the toolchain it depends on has itself changed. Both questions are open.
- DORY is documented by Bitcraze as the alternative and is adopted here on that
  basis. It has not been run on this bench. Nothing about it is verified.
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
