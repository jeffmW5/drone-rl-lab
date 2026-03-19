# exp_021 — Smooth Yaw-Aware Trajectory Benchmark

**Backend:** racing (lsy_drone_racing)
**Model:** exp_020 checkpoint (7.79 reward, 10M GPU steps) — NO RETRAINING
**Change:** Replaced midpoint-based trajectory with yaw-aware approach/departure waypoints (3 per gate)

---

## Trajectory Changes

The exp_020 controller used simple midpoints between gates:
```
waypoints = [takeoff, climbout, mid(prev, gate1), gate1, mid(gate1, gate2), gate2, ...]
```

exp_021 uses gate quaternion yaw to create smooth fly-through paths:
```
For each gate:
  pre_gate  = gate_pos - 0.5 * [cos(yaw), sin(yaw), 0]
  gate_center = gate_pos
  post_gate = gate_pos + 0.5 * [cos(yaw), sin(yaw), 0]
```

This creates approach/departure vectors aligned with the gate orientation.

---

## Level 0 Benchmark (5 runs)

| Controller | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg Gates |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|
| exp_020 (midpoint) | 1 gate, 5.80s | 1 gate, 5.78s | 1 gate, 5.76s | 1 gate, 5.80s | 1 gate, 5.78s | **1.0** |
| **exp_021 (yaw-aware)** | 3 gates, 11.4s | 2 gates, 11.26s | 2 gates, 11.3s | 2 gates, 11.3s | 3 gates, 12.4s | **2.4** |

**exp_021 passes 2.4x more gates on Level 0.** The yaw-aware trajectory eliminates the gate 1→2 crash.

## Level 2 Benchmark (10 runs)

| Run | Gates | Time (s) | Notes |
|:---:|:-----:|:--------:|-------|
| 1 | 3 | 11.10 | Best run — reached gate 4 approach |
| 2 | 1 | 6.84 | Crashed after gate 1 |
| 3 | 0 | 3.64 | Early crash |
| 4 | 0 | 2.04 | Early crash |
| 5 | 3 | 12.42 | Reached gate 4 approach |
| 6 | 1 | 6.80 | Crashed after gate 1 |
| 7 | 1 | 3.66 | Crashed after gate 1 |
| 8 | 2 | 11.56 | Reached gate 3 |
| 9 | 0 | 2.12 | Early crash |
| 10 | 0 | 2.18 | Early crash |

**Summary:** 0/10 finishes, avg 1.1 gates, best 3 gates. 4/10 runs crashed before gate 1.

### Comparison: exp_020 vs exp_021 on Level 2

| Metric | exp_020 (midpoint) | exp_021 (yaw-aware) |
|--------|:------------------:|:-------------------:|
| Max gates | 1 | **3** |
| Avg gates | 0.4 | **1.1** |
| Finishes | 0/10 | 0/10 |
| Gate 1 pass rate | 40% | **60%** |
| Gate 2+ pass rate | 0% | **30%** |
| Gate 3 pass rate | 0% | **20%** |

---

## Fallback Variants (Level 2, 5 runs each)

| Variant | Avg Gates | Best Gates | Notes |
|---------|:---------:|:----------:|-------|
| **d=0.5 (baseline)** | **1.1** | **3** | **Best overall** |
| d=0.75 | 0.2 | 1 | Too wide, more crashes |
| d=1.0 | 0.4 | 1 | Too wide, early crashes dominate |
| d=0.5 + alt interp | 1.2 | 2 | Similar to baseline, no improvement |

Wider approach distances (0.75m, 1.0m) make things worse — the spline overshoots when waypoints are too spread out. The original d=0.5 is optimal.

---

## Analysis

1. **Yaw-aware trajectories are a clear win**: 2.4x more gates on L0, 2.75x more on L2 vs midpoint approach
2. **The gate 1→2 crash is fixed**: exp_020 always crashed at ~5.8s between gates 1 and 2. exp_021 passes through gate 2 in 30% of L2 runs
3. **New bottleneck: gate 3→4 transition**: Best runs reach 3 gates but fail approaching gate 4. Similar altitude transition issue (z changes)
4. **Early crashes on L2 (40%)**: 4/10 runs crash before even reaching gate 1. These are likely due to extreme gate randomization making the initial trajectory unrecoverable
5. **Still no finishes**: 0/10 on L2. The gap to competition (sub-5s, ~100% finish) remains large
6. **Lap times are too slow**: Even 3-gate runs take 11-12s. At this pace, a full lap would be ~15s vs target 5s

## Bottleneck Assessment

The remaining gap is NOT trajectory shape alone. Key issues:
- **Speed**: The policy is too conservative. Training reward penalizes deviation but doesn't incentivize speed
- **Gate 3→4**: Same type of altitude transition crash as the old gate 1→2 problem
- **Robustness**: 40% of L2 runs crash before gate 1 — extreme randomization overwhelms the policy
- **Fundamental approach**: The policy follows pre-computed trajectories. Competition winners likely use reactive gate-seeking policies

## Recommendation

The trajectory-following approach has been pushed about as far as it can go. Next steps should be:
1. **Train directly on RaceCoreEnv** — skip the trajectory abstraction, train end-to-end gate racing
2. **Add gate proximity reward** — incentivize approaching and passing through gates
3. **Speed reward component** — penalize slow laps, not just trajectory deviation
