# V006 continuation setup -- 2026-10-11

Implemented a versioned rotor/deck/cable collision refinement and tracking-aware
navigation task, with a true V005 policy/normalizer/optimizer transfer path.
25 simulator checks and native .NET integration passed. Pendulum timestep
consistency check passed. 2,048 additional smoke transitions completed; no broad
learning improvement established. Main V006 is configured for 200,000 additional
transitions or 1,200 training seconds plus evaluation; launch is the next step.

See `tether_sim/FIDELITY.md` and `configs/tether_v006.json` for sources, assumptions,
new held-out seeds, success criteria and required real hardware calibration.

Main V006 ran to 43,964 transitions, then saved an interrupted pair for the user-requested CUDA continuation. See tether_v007_cuda.md.
