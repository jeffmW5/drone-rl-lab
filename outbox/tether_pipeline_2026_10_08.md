# Local WSL / AI-deck pipeline integration — 2026-10-08

Pulled origin/master from cbf2ff1 to db73496 and added the new pipeline under
`real_flight/tether`. Existing square detection/calibration, flight scripts,
historical configs and results are preserved. Local flight-log ZIPs are now
ignored; none were deleted. README no longer calls completed exp_057–059
running/queued, and shell scripts have LF line endings.

Native Linux repo: `/home/n33du/ai/drone-rl-lab`. Training venv remains separate
at `/home/n33du/ai/crazyflie-tether-venv`. Ignored vendor/model/generated/build
directories link to the prior verified workspace. A Docker mount-path fix
allows SDK builds through those links without exposing USB devices.

Verification from the repo checkout:

- Camera packet fixtures: 3/3 passed.
- Portable STM32 protocol checks passed (decode, corruption, synthetic flag,
  expiry and sequence wrap).
- GPU one-epoch synthetic training completed to `runs/repo-smoke` (34.18%
  validation accuracy). This checks execution, not model quality or improvement;
  the existing five-epoch deployment smoke checkpoint was preserved.
- GVSOC: 16 cases, 112 layer executions, zero byte mismatches. Result saved in
  `real_flight/tether/evidence/repo-golden-result.json`.
- Camera application image generation passes from the repo build wrapper;
  result in `real_flight/tether/evidence/repo-camera-result.json`.

Existing real hardware evidence remains in `GAP8_DORY_RESULT.md` and
`GAP8_PERF_RESULT.md`; this integration does not repeat those measurements.
No physical flashing, camera trial or flight occurred. Omarchy guest setup
awaits unlock. Keep data capture on native Windows, following the repo's
recorded VirtualBox NAT stream limitation.

Next: unlock Omarchy, verify connected radio/JTAG and exact AI-deck revision,
collect session-separated real tether data, and measure camera/inference/CPX
latency on hardware before implementing tether control.
