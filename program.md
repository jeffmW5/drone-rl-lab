# Drone RL Lab — Program

## Mission
Train autonomous drones using Reinforcement Learning across multiple backends:
- **hover** — Crazyflie CF2X hovering at [0,0,1] using gym-pybullet-drones + SB3 PPO
- **racing** — Crazyflie gate racing using lsy_drone_racing + CleanRL PPO

We iterate experiment-by-experiment, guided by Windows Claude (orchestrator),
executed by Linux Claude (executor). Each experiment is a frozen YAML config.

---

## Step 0 — Read Memory (MANDATORY)

Before starting ANY experiment or task, read the memory files:
1. **`memory/HARD_RULES.md`** — never violate these
2. **`memory/EXPERIMENT_LOG.md`** — don't repeat failed experiments without a clear reason
3. **`memory/INSIGHTS.md`** — Kaggle targets, benchmarks, architecture notes

- If your new experiment contradicts a Hard Rule, **STOP** and note the conflict in `outbox/`
- Refer to `inbox/INBOX.md` for the current task queue, memory files for historical context

---

## How to run an experiment

```bash
source /media/drones-venv/bin/activate
cd /media/drone-rl-lab
python train.py configs/exp_NNN.yaml
```

The dispatcher reads the `backend:` field from the config and routes to the
correct trainer (`train_hover.py` or `train_racing.py`). Direct calls also work:
```bash
python train_hover.py configs/exp_001_baseline.yaml
python train_racing.py configs/exp_010_racing_baseline.yaml
```

### After training completes:
1. Read the printed results
2. Write `results/exp_NNN/EXPERIMENT.md` (see documentation standard below)
3. Run `python compare.py` to see the leaderboard
4. Run `python plot.py` to generate training curves
5. Update `outbox/exp_NNN.md` with your analysis
6. `git add -A && git commit -m "exp_NNN: <short description>"`
7. `git push`
8. Run `python compare.py --generate-log` to update `memory/EXPERIMENT_LOG.md`
9. If you discover a new hard rule, add it to `memory/HARD_RULES.md`
10. Update `memory/NEXT.md` — strikethrough completed items
11. Update `outbox/STATUS.md` with the latest summary

---

## How to create a new experiment

Linux Claude: create a new YAML file in `configs/`.

### Hover config (gym-pybullet-drones):
```yaml
name: exp_006_description
backend: hover          # optional, defaults to "hover"
hypothesis: "What you're testing and why"
budget_seconds: 180
reward_code: |
  dist = np.linalg.norm(self.TARGET_POS_CUSTOM - state[0:3])
  return max(0, 2 - dist**4)
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  clip_range: 0.2
```

### Racing config (lsy_drone_racing):
```yaml
name: exp_010_racing_baseline
backend: racing
hypothesis: "Baseline CleanRL PPO on Level 0 racing"
budget_seconds: 600
racing:
  level: level0
  total_timesteps: 500000
  num_envs: 64
  learning_rate: 0.0015
  gamma: 0.94
  gae_lambda: 0.97
  clip_coef: 0.26
  ent_coef: 0.007
  vf_coef: 0.7
  cuda: false
  seed: 42
```

**Do NOT edit train_hover.py or train_racing.py** unless explicitly approved by the project owner — all experiment parameters should live in the config where possible.

---

## Paper Research

When experiments plateau (3+ consecutive with no improvement), the orchestrator
auto-queues a research task. The executor uses `/research <topic>` to:

1. Search Hugging Face Papers via MCP semantic search
2. Read top 3-5 papers as markdown (`huggingface.co/papers/ARXIV_ID.md`)
3. Extract techniques, architectures, reward designs, hyperparameters
4. Write summary to `research/<topic_slug>.md`
5. Propose concrete experiment configs with paper citations
6. Update `memory/INSIGHTS.md` with paper references

### Research summary documentation standard

```markdown
# Research: <Topic>

**Date:** YYYY-MM-DD
**Plateau context:** <current bottleneck summary>
**Search queries used:** <list>

## Papers Reviewed
### Paper N: <Title>
- **ArXiv:** <ID> | **HF:** https://huggingface.co/papers/<ID>
- **Core technique:** ...
- **Key insight for us:** ...
- **Proposed experiment:** ...

## Synthesis
<What do the papers collectively suggest?>

## Proposed Experiments
### exp_NNN_<name>
- **Hypothesis:** <from papers>
- **What to change:** <config/code changes>
- **Paper basis:** <citation>
```

---

## Available tools

| Tool | What it does |
|------|-------------|
| `/research <topic>` | Search HF Papers, extract insights, propose experiments |
| `python train.py configs/exp_NNN.yaml` | Run an experiment (auto-detects backend) |
| `python compare.py` | Print leaderboard of all experiments |
| `python compare.py --backend hover` | Leaderboard filtered to hover only |
| `python compare.py --backend racing` | Leaderboard filtered to racing only |
| `python compare.py --csv` | CSV output for machine-readable results |
| `python compare.py --generate-log` | Auto-generate `memory/EXPERIMENT_LOG.md` |
| `python compare.py --filter level=level2` | Filter experiments by key=value |
| `python scripts/benchmark.py -e exp_NNN` | Structured benchmark with JSON output |
| `python plot.py` | Generate training curve plots |
| `python plot.py --steps exp_NNN` | Plot per-step distance/velocity detail |

## Racing backend setup

To use the racing backend, lsy_drone_racing must be installed:
```bash
cd /media/lsy_drone_racing    # or wherever the fork is cloned
pip install -e ".[sim,rl]"
```

---

## GPU Training (RunPod)

For serious training runs, use a cloud GPU via RunPod.

### First-time setup
1. Create account at **runpod.io**, add payment method
2. **Set spending cap**: Settings > Billing > e.g. $10/month
3. Launch a pod: GPU Pods > Deploy > **PyTorch** template > **RTX 3090** (~$0.30/hr)
4. SSH in and run:
```bash
bash scripts/setup_deploy_key.sh    # first time only — follow instructions to add key on GitHub
bash scripts/setup_runpod.sh        # installs everything, starts 4h auto-shutdown timer
```

### Training on GPU
```bash
cd /root/drone-rl-lab
python train.py configs/exp_011_racing_gpu.yaml
```

GPU configs use `cuda: true` and `num_envs: 1024` (16x more parallel envs than CPU).

### After training
```bash
bash scripts/sync_results.sh "exp_011 racing GPU baseline"
# STOP YOUR POD via dashboard or: runpodctl stop pod $RUNPOD_POD_ID
```

Then on your local machine: `cd /media/drone-rl-lab && git pull`

### Safety features
- **Auto-shutdown**: Pod stops itself after 4 hours max
- **Spending cap**: Set in RunPod billing settings
- **Deploy keys**: SSH key scoped only to drone-rl-lab repo (not your full GitHub)

---

## What Linux Claude is allowed to change

### ✅ YES
- Create new config YAML files in `configs/`
- Write `results/exp_NNN/EXPERIMENT.md`
- Write `outbox/exp_NNN.md` and `outbox/STATUS.md`
- Update `memory/HARD_RULES.md`, `memory/NEXT.md`
- Run `compare.py --generate-log` (auto-updates `memory/EXPERIMENT_LOG.md`)
- Git commit and push
- Modify `train_racing.py`, `compare.py` for owner-approved infrastructure improvements

### ❌ NO — do not modify (without explicit approval)
- `train.py`, `train_hover.py` (infrastructure)
- `plot.py` (tools)
- `memory/EXPERIMENT_LOG.md` (auto-generated — use `compare.py --generate-log`)
- Physics engine, episode length, drone model
- Observation type (keep `KIN` for hover)
- Action type — unless explicitly instructed by Windows Claude

---

## Competition Target

**Goal: sub-5s average lap time on Level 2** — top 3 on the TUM Kaggle leaderboard.

Reference: WS25 Kaggle private leaderboard (Level 2, randomized physics + gates):

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |
| 4 | Liangyu Chen, Tuo Yang | 5.612 |
| 5 | Jai Seth | 9.558 |

Our Level 0 baseline: 13.36s (exp_010, CPU, 64 envs, 500k steps).
Default controllers on Level 0: ~13.3–13.9s.

---

## Evaluation Metric
**Primary (racing):** Average lap time (seconds) on the target level — lower is better.
**Secondary (racing):** `mean_reward` from training — higher is better.
**Primary (hover):** `mean_reward` from 10 evaluation episodes after training.
**Secondary:** `timesteps_trained` within the budget (sample efficiency proxy).

An experiment is a success if it improves the primary metric over previous best.

---

## EXPERIMENT.md Documentation Standard

Every experiment produces `results/exp_NNN/EXPERIMENT.md`:

```markdown
# Experiment NNN — <Short Title>

## What we changed
<Specific change — reference the config file>

## Why (the RL concept)
<What RL principle does this test? Explain simply and accurately.>

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | X.XX | Y.YY ✅/❌ |
| timesteps_trained | N | M |

## What this tells us
<Conclusions. Be honest about uncertainty.>

## Questions this opens up
- <Question 1>
- <Question 2>

## Suggested next experiment
<One specific hypothesis to test next>
```
