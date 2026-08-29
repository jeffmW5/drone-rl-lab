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

### Track B — prove the GAP8 can run a net at all

No stream needed. This is the project-killer question and it is cheap.

1. Flash Bitcraze's stock NN example to the GAP8 using the toolchain already
   proven on this bench (Docker `bitcraze/aideck` + Olimex JTAG, per
   `AIDECK_WIFI_CHECKPOINT.md`).
2. Measure inference rate and free L2.
3. Exit criterion: a known network runs on our deck at a measured fps.

If that number is low, the model design and possibly the whole task change.
Measure it before designing anything.

## Phases after both tracks pass

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 | Stable real hover on the flow deck | Holds position 30s, no wall contact. Current state: drifted into a wall 2026-04-20 — expected, the flow deck has no absolute reference. |
| 2 | Synchronized capture — frames + telemetry, timestamp-aligned | A dataset of flights with per-frame pose |
| 3 | Auto-label offline from the marker's apparent size and skew | Labeled set, held-out split |
| 4 | Train small int8 CNN, quantize via GAPflow/nntool | Runs on GAP8 within the Track B budget |
| 5 | Shadow mode — net infers while the flow deck flies | Net output tracks the marker-derived pose |
| 6 | Bounded live — net closes the loop, flow deck as fallback | Holds position on camera alone |
| 7 | Remove the marker | Hover on learned visual features alone |

## Open items carried in from the existing stack

- Hover P gains untuned (`kp_xy=8.0`, `kp_z=8000` drifted into a wall)
- Obs mismatch: exp_069 wants 55D, constructed obs is 62D, currently truncated
- `fly` modes never tested on hardware

## What this plan does NOT claim

- No vision model has been trained. None exists in either repo.
- Nothing in `windows_testbed/` has touched a real AI Deck.
- The sim RL line has passed one gate, once (exp_028). It is not a usable
  teacher for racing and is not on this plan's critical path.
