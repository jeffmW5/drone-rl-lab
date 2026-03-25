# exp_054 — Race Start (No Random Gate Spawns)

## Hypothesis
Eliminate training-benchmark domain gap by setting random_gate_ratio=0.0. All envs
start from the actual race start position ([-1.5, 0.75, 0.01]).

## Config
- random_gate_start: false, random_gate_ratio: 0.0
- z_low: -0.05, z_high: 2.0 (unchanged)
- All other params same as exp_046

## Training
- **Mean reward:** ~8.89 (FLAT from iter 50 to 3540+)
- **Steps:** ~14.5M / 20M (time-budget limited)
- **Zero learning progress across 14.5M steps**

### Reward Pattern (exact repeating cycle)
```
8.89 → 8.89 → 8.86 → 8.78 → 7.95 → 8.03 → 8.89 → 8.89 → ...
```
This cycle repeated identically from iter ~100 to iter 3540+ (the entire training).
v_loss also cycled: 0.1 → 5 → 15 → 100 → 185 → 0.1 → ...

## Result
**FAILURE** — Ground-level hover trap. Zero learning progress in 14.5M steps.

## Analysis
1. **z_low=-0.05 gives full altitude reward at ground level:** At z=0.01, alt_error=0
   (because z>z_low=-0.05), so alt_reward=0.5. The drone gets maximum altitude reward
   without taking off.

2. **Ground equilibrium:** Per-step reward at ground: survive(0.05) + alt(0.5) + drift
   progress(~0.6) = ~1.11/step × 8 steps = 8.89 per rollout. This is a stable
   equilibrium — any action (especially thrust) risks crashing for minimal gain.

3. **Fix: z_low=0.5 penalizes ground sitting:**
   - At z=0.01: alt_error=0.49, alt_reward=0.229 (vs 0.5 before = 54% reduction)
   - At z=0.5+: alt_reward=1.0 (with alt_coef=1.0)
   - Takeoff incentive: +6.17 per rollout for climbing to gate altitude
   - exp_055 tests this with alt_coef=1.0, z_low=0.5
