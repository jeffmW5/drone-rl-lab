# V006: destination continuation and simulator consistency

Hardware confirmed by owner on 2026-10-11: Crazyflie 2.1, Flow deck, AI deck.
Deck revisions, assembled mass, payload and cable specifications remain unmeasured.

## Model changes

V006 uses the versioned `random_navigation_v2` task. V1 defaults and the frozen
V005 config remain unchanged. It adds conservative rotor/deck collision volumes,
latches obstacle contacts during each 2 ms integration substep, and measures
signed geometry clearance for the body, rotor disks, decks, every complete cable
capsule and payload. It restores initial pose/tilt randomization discarded by
V1's model rebuild. Routes use a 26 cm inflated footprint margin and slower,
acceleration-limited setpoints which hold corners until the drone catches up.

Success now requires the commanded destination: route complete, drone within
12 cm horizontally, payload within 8 cm of its final goal, swing below 10 degrees,
no termination/collision. Accurate tracking of an unfinished moving target cannot
count as success. Twenty new held-out test seeds are separate from five validation
seeds. Results under V1 and V2 are not directly comparable as an RL ablation.

Training loads V005 best policy, optimizer and its paired normalization statistics,
then collects additional transitions. Source hashes are recorded. Existing run
folders cannot be overwritten. Best checkpoint ranks collisions, destination
successes, then trajectory RMSE; the file name does not imply hardware suitability.

## Checks observed before the main run

- 25 simulator checks passed, including rotor contact outside the old body box,
  cable mid-link contact, transient substep collision and destination metrics.
- Fixed-support pendulum at 12 links: 1.531623 s period. Halving the timestep
  changed period by 0.000202 percent. Six/twelve/twenty-four link checks and the
  rigid-rod reference are recorded in `evidence/fidelity_v006.json`.
- V006 continuation smoke completed 2,048 additional transitions and final paired
  evaluation. On seed 42001, PPO had no collision, 3.11 cm drone goal distance and
  8.76 cm payload goal error: it failed the strict 8 cm payload tolerance. PD had
  3.34 cm drone distance and 8.94 cm payload error. One seed does not establish
  improvement. Smoke source run: `tmp/tether-runs/tether_v006_smoke_20261011` in WSL.
- Native .NET Command Center integration passed: private engine, monitor, V006
  preview, images, telemetry and coordinate route command. It can play a selected
  completed run; future training repeats use V006, and active trainers are checked
  before another launch.

## Hardware calibration still required

The repository records 43.4 g for a prior `cf21B_500` build in
`real_flight/STATUS.md`; that is not a new weighing of the owner's current assembly.
The vendor's bare Crazyflie 2.1 datasheet gives 29 g, AI deck 1.1 gives 4.4 g,
and Flow deck v2 gives 1.6 g. Their 35 g sum is only a nominal comparison, excluding
other attachments and any different battery. V006 deliberately spans assumed
33–46 g vehicle mass. Motor thrust limit, drag and inertia are randomized assumptions,
not identified dynamics. Cable mass/length/lag randomization carries forward V005.

The default rotor radius is 22.5 mm, matching the vendor's 45 mm stock propeller
pack. Deck volumes remain illustrative conservative primitives. They have zero
additional mass because total vehicle mass already includes the assembly.

References:
- https://www.bitcraze.io/documentation/hardware/crazyflie_2_1/crazyflie_2_1-datasheet.pdf
- https://www.bitcraze.io/products/ai-deck/
- https://www.bitcraze.io/products/flow-deck-v2/
- https://www.bitcraze.io/documentation/hardware/propellers/propellers-datasheet.pdf
- https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance

Needed next: weighing; payload/cable dimensions and masses; motor thrust vs command
and voltage; inertia/attachment geometry; free-swing decay and slack/contact recordings;
camera intrinsics and real sensing/control latency. Rigid cable links can transmit
compression. Known geometry and privileged state remain in use. The AI/Flow sensors
are not yet modeled as calibrated observations, and no real flight is approved here.
