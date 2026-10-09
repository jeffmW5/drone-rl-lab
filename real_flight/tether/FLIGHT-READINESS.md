# From camera inference to tether flights

The installed pipeline currently estimates whether a line is absent, left,
center, or right in a camera image. It does not estimate tension, line length,
payload swing, slack, attachment forces, or the drone's position. These require
additional sensing, geometry, or a separately validated estimator. Synthetic
classification accuracy is a software check, not evidence of flight readiness.

## Define the physical task first

A hanging payload, a ground-anchored safety line, a powered tether, and following
a visible cable have different dynamics and failure modes. Record attachment
point, line material, length, mass, payload mass, and whether a reel is present.
Weigh the complete vehicle with battery and decks. Establish thrust reserve on
the actual platform rather than borrowing a payload limit from another model.
Check line visibility through the complete intended attitude and motion range;
a forward-facing camera may lose a line underneath the vehicle.

For lift experiments, the useful future outputs are attachment direction,
payload/line keypoints and visibility, followed by a geometry-aware state
estimate. Tension cannot generally be recovered uniquely from one image.
An instrumented anchor or load cell is useful independent ground truth during
development. Keep this distinction explicit in labels and reports.

## Hardware inventory before changes

Identify the exact Crazyflie and AI-deck revisions, camera orientation, current
firmware, radio, JTAG adapter, deck stack, positioning system, and recovery
method. AI-deck 1.1 uses a grayscale camera and GAP8 revision C; older hardware
differs. Bitcraze documents a compatible JTAG debugger for recovery and
development. [Official hardware information](https://www.bitcraze.io/products/ai-deck/).

Autonomous position hold needs a demonstrated position estimate independent of
this line classifier. Lighthouse provides onboard positioning but requires
calibration and clear optical coverage. Verify the exact deck compatibility and
communication pins before stacking hardware; the prepared application routes
perception through CPX. [Lighthouse setup](https://www.bitcraze.io/documentation/tutorials/getting-started-with-lighthouse/).

## Evidence to collect in order

1. Bench camera capture: verify dimensions, grayscale format, orientation,
   exposure, preprocessing parity, and sequence continuity.
2. Real dataset: record complete sessions with different backgrounds, lighting,
   line angles, slack, motion blur, occlusion, and distracting wires. Split by
   session. Include images where no usable tether observation exists.
3. Model: report per-class confusion, missed detections, false detections, and
   out-of-distribution examples. Compare float, integer reference, and actual
   generated GAP8 outputs on identical input bytes.
4. Hardware inference: measure capture-to-receipt latency, jitter, dropped
   frames, memory high-water marks, and sustained operation. The current packet
   timestamp is a capture request timestamp, not a measured exposure timestamp.
5. Observation-only operation: log perception alongside position, attitude,
   battery, commanded motion and independent tether ground truth. Verify stale,
   reordered, missing and synthetic observations cannot enable control.
6. Baseline flight: demonstrate ordinary bounded position hold and landing with
   the intended deck stack before adding tether forces.
7. Tether identification: characterize attachment forces and swing with bounded
   experiments, starting at the least demanding physical configuration.
8. Control integration: keep the existing stabilizer responsible for attitude.
   Introduce small bounded outer-loop corrections only after defining valid
   sensing, rate limits, motion bounds, operator override and loss-of-observation
   behavior. A loss of tether visibility must not cause continued blind pulling.

The current STM32 application only logs observations. Control integration is
deliberately still an implementation milestone, and requires the task details
above. No aircraft has been flashed or flown during this setup.

## Why DORY

Bitcraze currently reports that the GreenWaves website outage prevents obtaining
AutoTiler and points to DORY as an alternative. This workspace uses DORY and
PULP kernels with a narrow integer exporter; it does not depend on acquiring or
bypassing the proprietary AutoTiler. DORY code generation is a separate step
from proving SDK compilation, numerical correctness, and camera integration.
[Official AI-deck setup notice](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/).
