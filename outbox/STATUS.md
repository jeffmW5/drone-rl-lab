# Status -- Last Updated 2026-03-19

## Last Completed
- **exp_022** -- RaceCoreEnv baseline (direct gate-racing RL)
- **Result:** Mean reward 6.34 ± 2.60, peak 10.04 at 1.23M steps
- **Note:** First model trained on MuJoCo gate-racing physics. 1.93M/3M steps (hit budget).
  Agent consistently passes 1-2 gates in training. Needs L2 benchmark.

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max (trajectory-following)
- **Racing L2 (RaceCoreEnv):** exp_022 -- reward 6.34, benchmark pending

## Queue Status
- Completed: exp_022 (training)
- Next: exp_022 benchmark on L2 sim
- Queued: exp_023 (extended RaceCoreEnv training)

## Pending Orchestrator Review
- outbox/exp_022_racecore.md (new)
