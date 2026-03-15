# exp_010_racing_baseline — Results

**Backend:** racing (lsy_drone_racing)
**Hypothesis:** Baseline CleanRL PPO on Level 0 drone racing — perfect knowledge, no randomization. Establishes racing performance floor.

| Metric | Value |
|--------|-------|
| mean_reward | 7.359 +/- 0.014 |
| timesteps_trained | 499,712 |
| wall_time | 588.8s |
| level | level0 |

## Analysis
Racing backend is fully operational. The agent learned trajectory following from scratch, climbing from -44.93 to 7.35 reward over 500k steps. Learning curve shows three phases: crash recovery (0-25k), rapid improvement (25k-250k), and plateau (250k-500k). The low standard deviation (0.014) at convergence indicates stable performance. Full details in `results/exp_010_racing_baseline/EXPERIMENT.md`.

## Setup notes
- Required Python 3.11 (JAX >= 0.7 needs it; system Python 3.10 was insufficient)
- Venv created at `/home/jeff/drones-venv` (not `/media/drones-venv` — VirtualBox shared folder doesn't support symlinks)
- Redirect script at `/media/drones-venv/bin/activate` sources the real venv
- sim.py pre-trained model test passed (4/4 gates, 13.86s flight time)

## Suggested next
Try 1M timesteps or Level 1 randomization to test if we can beat 7.35 or test generalization.
