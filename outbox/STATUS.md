# Status -- Last Updated 2026-03-19

## Last Completed
- **exp_022** -- RaceCoreEnv baseline (direct gate-racing RL) + L2 benchmark
- **Training (re-run):** Mean reward 10.46 ± 0.98, peak 11.83 at 2.03M steps (budget: 1800s)
- **Benchmark L2:** 0/10 finishes, avg 0.5 gates, all crash at 2.02s
- **Root cause:** Model flies upward, passes gate 0 (~step 40), but hits ceiling (z=3.06 at step 101).
  The 100-step grace period (bug: uses sim_freq/5=100 env steps instead of env_freq/5=10) masks
  the out-of-bounds termination during training. Model never learns altitude control.
- **Key finding:** No physics gap — same MuJoCo sim in training and benchmark. Issue is purely
  reward shaping: model maximizes gate proximity by flying straight up, never penalized for OOB.

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv):** exp_022 -- 0/10 finishes, 0.5 gates avg (altitude control failure)

## Queue Status
- Completed: exp_022 (training + benchmark)
- Next: exp_023 (RaceCoreEnv with altitude penalty + longer training)
- **Critical for exp_023:** Add OOB penalty to training reward, possibly reduce grace period
