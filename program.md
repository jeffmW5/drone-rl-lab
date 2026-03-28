# Drone RL Lab -- Program

## Mission

Train autonomous drones using Reinforcement Learning across multiple backends:
- **hover** -- Crazyflie CF2X hovering at `[0, 0, 1]` using
  `gym-pybullet-drones` + SB3 PPO
- **racing** -- Crazyflie gate racing using `lsy_drone_racing` + CleanRL PPO

We iterate experiment-by-experiment, guided by Windows Claude (orchestrator)
and executed by Linux Claude (executor). Each experiment is a frozen YAML config.

---

## Step 0 -- Read Memory (MANDATORY)

Before starting ANY experiment or task, read:
1. **`memory/HARD_RULES.md`** -- repo integrity and evaluation constraints
2. **`memory/EPISTEMIC_SCHEMA.md`** -- how to classify claims
3. **`memory/BELIEF_AUDIT.md`** -- current claims under review
4. **`memory/FACTS.md`** -- direct observations with sources
5. **`memory/HYPOTHESES.md`** -- active explanations being tested
6. **`memory/TENTATIVE_LESSONS.md`** -- scoped heuristics, not absolutes
7. **`memory/EXPERIMENT_LOG.md`** -- do not repeat failed experiments without a clear reason
8. **`memory/INSIGHTS.md`** -- benchmarks, papers, and background context

- If your new experiment contradicts a hard rule, **STOP** and note the conflict in `outbox/`.
- Refer to `inbox/INBOX.md` for the current task queue.
- Read `outbox/STATUS.md` for the latest narrative summary.
- Read `state/current.json` when present for machine-readable lab state.

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

### After training completes
1. Read the printed results.
2. Write `results/exp_NNN/EXPERIMENT.md` (see documentation standard below).
3. Run `python3 scripts/capture_provenance.py --experiment exp_NNN`.
4. Run `python compare.py` to see the leaderboard.
5. Run `python plot.py` to generate training curves.
6. Update `outbox/exp_NNN.md` with your analysis.
7. Run `python compare.py --generate-log` to update `memory/EXPERIMENT_LOG.md`.
8. Add direct observations to `memory/FACTS.md` when they matter beyond one report.
9. Add explanations to `memory/HYPOTHESES.md` or `memory/TENTATIVE_LESSONS.md` using the schema.
10. Only add something to `memory/HARD_RULES.md` if it is a true process invariant, not an empirical interpretation.
11. Update `memory/NEXT.md` by striking through completed items.
12. Update `outbox/STATUS.md` with the latest summary.
13. Run `python3 scripts/lab_state.py`.
14. `git add -A && git commit -m "exp_NNN: <short description>"`
15. `git push`

---

## How to create a new experiment

Linux Claude creates a new YAML file in `configs/`.

### Hover config (`gym-pybullet-drones`)

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

### Racing config (`lsy_drone_racing`)

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

**Do NOT edit** `train_hover.py` or `train_racing.py` unless explicitly approved
by the project owner. All experiment parameters should live in config where possible.

---

## Paper Research

When experiments plateau (3+ consecutive with no improvement), the orchestrator
can queue a research task. The executor uses `/research <topic>` to:

1. Search Hugging Face Papers via MCP semantic search.
2. Read top 3-5 papers as markdown (`huggingface.co/papers/ARXIV_ID.md`).
3. Extract techniques, architectures, reward designs, and hyperparameters.
4. Write a summary to `research/<topic_slug>.md`.
5. Propose concrete experiment configs with citations.
6. Update `memory/INSIGHTS.md` with paper references.
7. Add or revise repo hypotheses only after separating paper-supported ideas from local evidence.

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
|------|--------------|
| `/research <topic>` | Search HF Papers, extract insights, propose experiments |
| `python train.py configs/exp_NNN.yaml` | Run an experiment (auto-detects backend) |
| `python compare.py` | Print leaderboard of all experiments |
| `python compare.py --backend hover` | Leaderboard filtered to hover only |
| `python compare.py --backend racing` | Leaderboard filtered to racing only |
| `python compare.py --csv` | CSV output for machine-readable results |
| `python compare.py --generate-log` | Auto-generate `memory/EXPERIMENT_LOG.md` |
| `python compare.py --filter level=level2` | Filter experiments by key=value |
| `python scripts/benchmark.py -e exp_NNN` | Structured benchmark with JSON output |
| `python3 scripts/capture_provenance.py --experiment exp_NNN` | Save repo and environment provenance next to results |
| `python3 scripts/lab_state.py` | Refresh `state/current.json` |
| `python plot.py` | Generate training curve plots |
| `python plot.py --steps exp_NNN` | Plot per-step distance / velocity detail |

## Racing backend setup

To use the racing backend locally, `lsy_drone_racing` must be installed. For
ad-hoc local installs:

```bash
cd /media/lsy_drone_racing    # or wherever the fork is cloned
pip install -e ".[sim,rl]"
```

On fresh RunPod pods, prefer `bash scripts/setup_runpod.sh`, which bootstraps
the fork through Pixi and adds the RL extras inside the Pixi GPU environment.

---

## GPU Training (RunPod)

For serious training runs, use a cloud GPU via RunPod.

### First-time setup
1. Create an account at **runpod.io** and add a payment method.
2. Set a spending cap in billing.
3. Launch a pod: GPU Pods -> Deploy -> **PyTorch** template -> **RTX 3090**.
4. SSH in and run:

```bash
bash scripts/setup_deploy_key.sh    # first time only
bash scripts/setup_runpod.sh        # installs Pixi GPU env + RL extras, starts 4h auto-shutdown timer
```

### Training on GPU

```bash
cd /root/drone-rl-lab
drone-rl-gpu-python train.py configs/exp_NNN.yaml
```

GPU configs use `cuda: true` and `num_envs: 1024` for heavy parallelism.

### After training

```bash
bash scripts/sync_results.sh "exp_NNN description"
# STOP YOUR POD via dashboard or: runpodctl stop pod $RUNPOD_POD_ID
```

Then on your local machine: `cd /media/drone-rl-lab && git pull`

### Safety features
- **Auto-shutdown**: Pod stops itself after 4 hours max.
- **Spending cap**: Set in RunPod billing settings.
- **Deploy keys**: SSH key scoped only to this repo.

---

## What Linux Claude is allowed to change

### YES
- Create new config YAML files in `configs/`
- Write `results/exp_NNN/EXPERIMENT.md`
- Write `outbox/exp_NNN.md` and `outbox/STATUS.md`
- Update `memory/HARD_RULES.md` and `memory/NEXT.md`
- Run `compare.py --generate-log` (auto-updates `memory/EXPERIMENT_LOG.md`)
- Run workflow automation such as provenance and state refresh
- Git commit and push
- Modify `train_racing.py` or `compare.py` for owner-approved infrastructure improvements

### NO -- do not modify without explicit approval
- `train.py`
- `train_hover.py`
- `plot.py`
- `memory/EXPERIMENT_LOG.md` (auto-generated -- use `compare.py --generate-log`)
- Physics engine, episode length, drone model
- Hover observation type (`KIN`)
- Action type unless explicitly instructed by Windows Claude

---

## Competition Target

**Goal: sub-5s average lap time on Level 2** -- top 3 on the TUM Kaggle leaderboard.

Reference: WS25 Kaggle private leaderboard (Level 2, randomized physics + gates):

| Rank | Team | Avg Lap (s) |
|:----:|------|:-----------:|
| 1 | Team Y | 3.394 |
| 2 | Group6 | 4.886 |
| 3 | Limo | 5.022 |
| 4 | Liangyu Chen, Tuo Yang | 5.612 |
| 5 | Jai Seth | 9.558 |

Our Level 0 baseline: 13.36s (`exp_010`, CPU, 64 envs, 500k steps).
Default controllers on Level 0: about 13.3-13.9s.

---

## Evaluation Metric

**Primary (racing, final objective):** Average lap time on the target level -- lower is better.

**Primary (racing, pre-finish operational ranking):** finish rate, then average
gates passed, then standardized benchmark survival / average time.

**Secondary (racing):** `mean_reward` from training -- useful within a similar
reward definition, but not a global cross-experiment leaderboard when reward
terms change.

**Primary (hover):** `mean_reward` from 10 evaluation episodes after training.

**Secondary:** `timesteps_trained` within the budget (sample efficiency proxy).

An experiment is a success if it improves the primary metric over previous best.

---

## EXPERIMENT.md Documentation Standard

Every experiment produces `results/exp_NNN/EXPERIMENT.md`:

```markdown
# Experiment NNN -- <Short Title>

## What we changed
<Specific change -- reference the config file>

## What was held constant
<Name the important things that did NOT change>

## Why (the RL concept)
<What RL principle does this test? Explain simply and accurately.>

## Results
| Metric | Previous best | This experiment |
|--------|---------------|-----------------|
| mean_reward | X.XX | Y.YY |
| timesteps_trained | N | M |

## Observations
<Direct measurements only>

## Inference
<Interpretation. Be explicit about uncertainty and scope.>

## Confidence
low | medium | high

## What this does NOT prove
<Prevent over-promotion of the claim>

## Next falsification test
<What result would overturn your current interpretation?>

## Suggested next experiment
<One specific hypothesis to test next>
```

Avoid universal claims like "the bottleneck is" or "X is exhausted" unless the
repo has replicated that conclusion cleanly.
