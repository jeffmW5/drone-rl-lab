# Status -- Last Updated 2026-03-18

## Last Completed
- **exp_021** -- Smooth yaw-aware trajectory benchmark
- **Result:** 0/10 L2 finishes, but 2.4x more gates vs exp_020
- **Recommendation:** Train directly on RaceCoreEnv (see outbox/exp_021_smooth_traj.md)

## Current Best
- **Hover:** exp_002 -- reward 474 (ceiling for ONE_D_RPM)
- **Racing L0:** exp_010 -- 13.36s, 5/5 finishes
- **Racing L2 (lap time):** exp_016 -- 13.49s, 2/10 finishes
- **Racing L2 (gate count):** exp_021 -- 0/10 finishes, 3 gates max

## Queue Status
- Completed: exp_021
- Remaining: 0 items in queue

## Pending Orchestrator Review
- outbox/exp_021_smooth_traj.md (new)
