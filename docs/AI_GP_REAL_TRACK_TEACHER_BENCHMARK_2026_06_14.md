# AI-GP Real-Track Teacher Benchmark

Date: June 14, 2026

## Scope

This benchmark rebuilt the structured-state AI-GP PPO environment around the
measured six-gate course before any further live-policy authority testing.

Implemented:

- canonical six-gate NED centers and `2.72 m` gate dimensions
- explicit NED-to-surrogate transform: `(-north, east, offset - down)`
- explicit `27.468 m` altitude offset, placing the lowest gate `1.5 m` above
  the surrogate ground
- upright gate planes with horizontal normals inferred from the incoming course
  segment
- segment-interpolated gate-plane crossing tests
- terminal `missed_gate` events and configurable penalties
- missed-gate rates in training and deterministic telemetry reports

The environment is topology-correct for measured centers and dimensions. Gate
orientation was not present in the logged track payload and remains inferred.
Surrogate thrust, drag, latency, mass, and rate response remain dynamics-unfitted.

## Validation

Local:

- full unit suite: `30/30` passed
- CPU forward/backward smoke: passed with finite `(64, 45)` observations

RunPod:

- GPU: NVIDIA GeForce RTX 3090
- CUDA smoke: passed with finite `(256, 45)` observations

## Run 018: Initial Real-Track Teacher

Config:

```text
configs/ai_gp_018_real_track_teacher_10m.yaml
```

Training:

- interactions: `10,092,544`
- elapsed: `88.21 s`
- throughput: `114,409 interactions/s`

Best-checkpoint nominal telemetry, 256 episodes:

| Metric | Result |
|---|---:|
| Gate-0 passage | `0.0%` |
| Mean gates | `0.000` |
| Finish rate | `0.0%` |
| Missed-gate rate | `100.0%` |
| Collision rate | `0.0%` |
| Out-of-bounds rate | `0.0%` |
| Vertical-runaway rate | `0.0%` |

All sampled trajectories crossed gate 0 high. Typical trajectories gained about
`3.4 m` before reaching the plane. The `12 x 23 m` approach-progress return
outweighed the original `50` missed-gate penalty.

## Run 019: One Reward Correction

Config:

```text
configs/ai_gp_019_real_track_teacher_reward_fix_10m.yaml
```

The one allowed correction added gate-relative altitude error cost and made
valid crossing or terminal miss dominate approach progress.

Training:

- interactions: `10,092,544`
- elapsed: `79.45 s`
- throughput: `127,038 interactions/s`
- selected checkpoint: step `2,621,440`

Best-checkpoint nominal telemetry, 256 episodes:

| Metric | Result |
|---|---:|
| Gate-0 passage | `11.3%` |
| Mean gates | `0.113` |
| Finish rate | `0.0%` |
| Missed-gate rate | `99.6%` |
| Collision rate | `0.0%` |
| Out-of-bounds rate | `0.4%` |
| Vertical-runaway rate | `0.0%` |
| Minimum valid crossing margin | `0.026 m` |

Twenty-nine episodes passed gate 0, but none passed gate 1. The 227 gate-0
misses had mean vertical error `+1.17 m`. After a gate-0 pass, gate-1 misses had
mean vertical error `+3.81 m`.

Randomized deterministic evaluation used 256 episodes per seed:

| Seed | Gate 0 | Mean gates | Missed | Collision | OOB | Vertical runaway |
|---:|---:|---:|---:|---:|---:|---:|
| `1001` | `2.0%` | `0.020` | `61.7%` | `1.2%` | `37.5%` | `36.3%` |
| `1002` | `1.6%` | `0.016` | `63.7%` | `1.2%` | `35.2%` | `36.7%` |
| `1003` | `0.8%` | `0.008` | `58.6%` | `2.3%` | `39.5%` | `44.1%` |

Three-seed means:

- gate-0 passage: `1.4%`
- mean gates: `0.014`
- finish rate: `0.0%`
- missed-gate rate: `61.3%`
- collision rate: `1.6%`
- out-of-bounds rate: `37.4%`
- vertical-runaway rate: `39.1%`

## Decision

The benchmark failed promotion:

- required nominal gate-0 passage: at least `95%`; observed `11.3%`
- required mean gates: at least `2`; observed `0.113`
- collision target was met nominally, but randomized bounds and vertical
  behavior failed

No checkpoint is approved for student distillation, export, or Windows command
evaluation. The reward correction improved gate-0 passage but did not establish
multi-gate control or randomized robustness.

## Artifacts

Generated artifacts remain outside Git:

```text
results/ai_gp_018_real_track_teacher_10m/
results/ai_gp_019_real_track_teacher_reward_fix_10m/
```

Each directory contains config, metrics, best/final checkpoints, and nominal
telemetry. Run 019 also contains randomized telemetry for seeds `1001-1003`.

## Next Engineering Work

Fit surrogate thrust, body-rate response, drag, and latency from synchronized
Windows command/telemetry trajectories before another long PPO run. Obtain
actual gate orientations if available; current plane yaw is inferred from course
topology. Then revisit race-start and vertical-control curriculum using the
measured dynamics rather than launching another reward or authority sweep.
