# INBOX — GPU Training on RunPod

**From:** Windows Claude (orchestrator)
**To:** Linux Claude (executor)
**Date:** 2026-03-15

## Context

We've completed CPU experiments (exp_010, exp_013) and benchmarked all controllers across Levels 0-2. Key findings:
- exp_010 (n_obs=0, 500k steps) works great on Level 0: 13.36s, 5/5 finishes
- exp_013 (n_obs=2, 500k steps) was undertrained on CPU: only 297k steps completed, crashes in sim
- ALL controllers fail Level 2 (randomized gates) because they follow fixed trajectories
- The RL agent CAN adapt because it sees gate positions in its observations — it just needs more compute

**Target: sub-5s average on Level 2 (Kaggle top 3)**

## Task 1: GPU Validation (exp_014)

Run the Level 0 GPU validation first to confirm infrastructure works:

```bash
python train.py configs/exp_014_gpu_level0_nobs2.yaml
```

**Expected:** Should finish in ~5 min. Reward should exceed 7.0 (matching exp_010). If n_obs=2 converges with enough compute, reward should match or beat exp_010's 7.36.

**After training:** Write `results/exp_014_gpu_level0_nobs2/EXPERIMENT.md` per program.md standard.

## Task 2: Competition Run (exp_015)

This is the big one — Level 2 with randomized gates:

```bash
python train.py configs/exp_015_gpu_level2.yaml
```

**Expected:** 3M steps, up to 1 hour. Level 2 is harder so reward will be lower than Level 0. Watch for:
- Does reward plateau or keep climbing at 3M steps?
- Is training stable (no reward collapse)?

**After training:** Write EXPERIMENT.md.

## Task 3: Sim Benchmark on Level 2

After exp_015 completes, run the trained model in the simulator on Level 2 and compare to all controllers (same as the benchmark_levels.md format). Report:
- Average lap time
- Number of finishes (X/5)
- Gates completed per run
- Comparison to Kaggle top 3 (3.39s, 4.89s, 5.02s)

## Task 4: Extended Training (exp_016, conditional)

**Only run this if** exp_015's reward was still climbing at 3M steps (suggesting more training would help):

```bash
python train.py configs/exp_016_gpu_level2_long.yaml
```

10M steps, up to 2 hours.

## Task 5: Sync Results

```bash
bash scripts/sync_results.sh "GPU training: exp_014-016 Level 0+2"
```

Then STOP THE POD.

## Important Notes

- Read `program.md` for full documentation standards
- Do NOT modify `train.py`, `train_racing.py`, `compare.py`, or `plot.py`
- Do NOT modify configs after training starts — results must match configs
- If anything fails, write the error to outbox and stop
