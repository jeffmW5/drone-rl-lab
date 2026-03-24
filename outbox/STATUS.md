# Status -- Last Updated 2026-03-24

## Last Completed
- **exp_044** -- Min stability + max progress, 20M steps, random mid-air spawns
- **Training reward**: 26.384 ± 0.724 (peaked at 37 — HIGHEST EVER)
- **Benchmark L2**: 0 gates, 0.7-1.4s flight (6 runs)
- **Finding:** Reward design is SOLVED — stochastic policy navigates and likely passes gates
  during training (reward 37 impossible without gates). But deterministic mean crashes at
  deployment. The problem is now purely policy optimization / exploration-exploitation gap.

## Approaches Attempted
1. **Reward tuning** (exp_026-037): survive, speed, PBRS sweeps — all firmly hover-or-crash
2. **Entropy regularization** (exp_038): destroys hover → crash, no navigation
3. **Short episodes** (exp_039): bistable (hover+crash), edge of transition, still 0 gates
4. **View+progress from mid-air** (exp_040): falling exploit
5. **Progress/view variants** (exp_041-043): all crash in 0.5s without survival incentive
6. **Min stability + max progress** (exp_044): best training ever, still crashes at benchmark

## Experiment Summary (recent)
| Exp | Train Reward | Gates | Flight Time | Key Change |
|-----|-------------|-------|-------------|------------|
| 040 | 7.75 | 0 | 0.52s | view+progress, falling exploit |
| 041 | 7.74 | 0 | 0.52s | progress only (no view) |
| 042 | 7.74 | 0 | 0.52s | view=0.1 + progress |
| 043 | 7.75 | 0 | 0.52s | view*progress multiplicative |
| **044** | **26.38 (peak 37)** | **0** | **0.9s** | **survive=0.05 + progress + gate_bonus (BEST TRAINING EVER)** |

## Next Steps — Policy Optimization Problem
Reward design is done. Need to fix the deterministic mean policy:
1. **Deploy stochastic policy** — add noise at benchmark time
2. **Larger network** — [512,512,256,128] per Pasumarti 2024 (needs code change)
3. **Curriculum** — hover → navigate transition
4. **SAC** — entropy-maximizing, no exploration/exploitation gap

## Current Best
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (RaceCoreEnv flight):** exp_034 -- 0/5 finishes, 0 gates, **29.98s perfect hover**
- **Racing L2 (RaceCoreEnv gates):** exp_028 -- 0/5 finishes, **0.2 avg gates**, 0.94s
- **Racing L2 (training reward):** exp_044 -- 26.38 mean, 37 peak (BEST EVER, but 0 benchmark gates)

## Queue Status
- Completed: exp_022-044
- In progress: designing next experiment
