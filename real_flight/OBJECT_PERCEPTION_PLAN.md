# Onboard Object Perception — Hands and Other Drones

Date: 2026-08-30
Parent: `VISION_HOVER_PLAN.md`

## Goal and boundary

The printed square is an integration milestone, not the final perception
target. It exists to prove the complete onboard chain with cheap ground truth:

```
HM01B0 camera -> GAP8 int8 network -> correction vector -> STM32 -> motors
```

After that chain works, replace marker-specific perception with learned
perception of real objects. The target order is:

1. nearby hand detection;
2. hand tracking and, only if needed, gesture classification;
3. nearby drone detection;
4. drone tracking and conservative avoidance/following.

Hands come first because they are larger in the 324x244 image, easy to collect,
and easy to label. A Crazyflie-sized target is only roughly 17 px wide at 1 m
and 9 px at 2 m using the current provisional focal estimate; the raw camera
therefore imposes a hard long-range limit that no model can train away.

The marker milestone should be completed far enough to prove integration, but
marker thresholding and distance calibration are not to become an open-ended
optimization project once that proof exists.

## Output contract by behavior

| Behavior | Minimum network output |
| --- | --- |
| Presence only | class confidence |
| Locate or avoid | class, confidence, center x/y, width, height |
| Hand gesture | hand box plus gesture class (or a second classifier on the crop) |
| Follow another drone | drone box plus temporal track and a separately validated range signal |

Start with one object of interest per frame. A fixed one-object output vector is
smaller and avoids onboard non-maximum suppression. Move to a grid/multi-object
detector only after real data shows that multiple simultaneous objects are a
requirement.

## Ground truth and data

The marker detector is not the teacher for hands or drones. Capture real
324x244 grayscale frames from the HM01B0 and label them with a capable host-side
detector followed by human correction. The GAP8 model is the small student.

Each dataset must include:

- target position, distance, scale, and orientation variation;
- different people or drones, backgrounds, and lighting;
- motion blur, partial occlusion, image-edge truncation, and difficult contrast;
- abundant empty frames and confusing negatives;
- raw frames from the exact AI Deck camera path used at deployment.

Split train/validation/test by capture session, not by neighboring frames.
Random frame splits leak nearly identical video frames across sets and give a
misleading accuracy number. Preserve raw frames and labels with provenance so
teacher mistakes can be corrected rather than baked into the student.

## Distance and tracking

Do not carry the marker equation `distance = k / size_px` over unchanged.
Hands vary in physical size and pose; drones foreshorten as their attitude
changes. For first avoidance behavior, prefer conservative range bins
(`near`, `safe`, `far`) and bounding-box expansion over time. A known drone
type may later support a learned range head trained at measured distances, but
it must be validated independently before it commands approach or avoidance.

Track detections over time on the STM32 or GAP8 to reject one-frame false
positives, estimate image velocity/expansion, and bridge occasional missed
frames. Loss of detection must transition to hover, retreat, or the existing
safety controller; it must never mean continuing an approach.

## Model and hardware constraints

- Train on grayscale at the deployed resolution or an explicitly measured
  downsample, not on clean desktop RGB alone.
- Prefer a small int8 network using operations already proven through DORY.
- Size the network against `VISION_HOVER_PLAN.md`'s measured capture/inference
  budget. Prove 5 Hz with the existing capture path first; pursue 10 Hz only
  after combined continuous capture plus inference is measured on hardware.
- Evaluate detection recall, false positives, and localization error by range;
  aggregate accuracy alone hides the dangerous cases.

## Staged plan and promotion gates

| Stage | Work | Exit criterion |
| --- | --- | --- |
| 0 | Marker integration proof | A GAP8 network drives bounded rise-to-level/distance corrections with the Flow Deck fallback available. |
| 1 | Hand dataset and labels | Session-separated real dataset with corrected boxes and representative negatives. |
| 2 | Tiny hand model | Held-out recall/false-positive results reported by distance and lighting; model fits the measured GAP8 budget. |
| 3 | Hand shadow mode | Onboard output tracks reviewed host labels without commanding flight. |
| 4 | Bounded hand behavior | Conservative response in a contained test area; loss/false-positive behavior verified. |
| 5 | Drone dataset and model | Nearby-drone results reported by pixel size/range; long-range limit stated explicitly. |
| 6 | Drone tracking/range | Temporal track is stable and any range signal passes measured-distance tests. |
| 7 | Bounded avoidance/following | Flow Deck/safety controller remains authoritative until repeated shadow and tethered tests pass. |

Every promotion uses recorded benchmark sessions. A compelling live demo is
evidence, but it is not a replacement for held-out data or the fallback tests.
