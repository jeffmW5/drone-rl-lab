# AI-GP 10M Benchmark - 2026-06-09

## Scope

These results are from the Torch GPU surrogate only. They are not evidence of
passage in the Windows AI Grand Prix simulator.

## CUDA Smoke

- config: `configs/ai_gp_002_swift_teacher_gpu_ppo.yaml`
- GPU: RTX 3090
- PyTorch: `2.1.0+cu118`
- observation shape: `(256, 45)`
- forward, environment step, loss, and backward pass: finite

## Privileged Teacher

The short benchmark used
`configs/ai_gp_002_swift_teacher_gpu_ppo_10m.yaml`.

- interactions: `10,092,544`
- elapsed training plus built-in evaluation: `70.1 s`
- measured training throughput: roughly `151k-174k interactions/s`
- PPO losses and tensors: finite

Nominal deterministic evaluation, 256 race-start episodes:

- gate-0 passage: `100%`
- mean gates: `4.0`
- completion: `100%`
- collision: `0%`
- vertical runaway: `0%`

Randomized deterministic evaluation, three seeds and 768 total episodes:

- gate-0 passage: `100%`
- mean gates: `3.9935`
- completion: `99.48%`
- collision: `0%`
- vertical runaway: `0%`
- maximum altitude observed: `2.98 m`
- maximum absolute vertical speed observed: `4.23 m/s`
- minimum crossing margin observed: `0.57 mm`

The time-series trajectories include direct paths around `29 m` and detoured
paths around `37-51 m`. Nominal completion time ranged from roughly `8.2 s` to
`12.6 s`. The narrow gate margins and detours are robustness concerns even
though aggregate completion is high.

## Live-Contract Transfer

Two 10.09M-interaction transfers into the 18D feedforward live contract were
tested.

Plain behavior cloning, best checkpoint:

- completion: `0%`
- mean gates: `1.60`
- collision: `24.2%`
- out of bounds: `75.8%`
- vertical runaway: `75.8%`

DAgger-style mixed rollout, best checkpoint:

- completion: `0.39%`
- mean gates: `1.25`
- collision: `44.9%`
- out of bounds: `54.7%`
- vertical runaway: `39.8%`

Neither student is eligible for export or Windows control evaluation.

## Conclusion

The benchmark validates privileged trajectory learning in the surrogate. It
does not validate a live policy. The current 18D feedforward observation is
insufficient for reliable transfer under the tested methods.

The next training change should add temporal state or richer gate geometry to
the deployable contract, then repeat distillation and deterministic telemetry
evaluation. Only a passing 18D-compatible student should be exported for
receive-only Windows shadow evaluation.
