# Local 1070 Ti research loop

The original lab's pipeline.sh executes the inbox queue, trains, benchmarks,
documents and publishes. Its run_job.py adds saved job steps, while the agent
reviews results and queues hypotheses. This tether loop retains that division:

1. A recurring Codex heartbeat in this chat reviews evidence and hypotheses.
2. System Python runs `python3 tether_sim/research_loop.py tick` in the native
   WSL checkout. It waits for any active trainer, then starts one PufferLib job.
3. The trainer uses the 1070 Ti for the policy, optimization and CUDA advantages;
   four CPU workers run eight MuJoCo environments. No concurrent compiler job.
4. Fixed validation seeds select a provisional checkpoint: crashes, 1 mm
   successes, then full 3D destination error. Promotion requires no extra crash
   or lost route completions, and either more successes or at least 5% lower
   mean destination error. Three seeds provide a search signal, not proof.
5. Record both positive and negative trials in the ledger and outbox table.
6. After the three-trial batch, review failure cases before another campaign.
   Run the selected policy once on reserved 71001-71020 seeds for an audit;
   do not use those audit results to select within this campaign.

The initial campaign waits for tether_puffer_precision_v001 to finish. It tests
learning_rate=1e-5, more transitions, then gamma=.999. Each trial resumes the
current provisional incumbent, preserving its learning rate/gamma unless the
trial changes them. This is adaptive sequential search, not a matched ablation
or replicated statistical result. Trial test_seeds deliberately equal validation
seeds so repeated final evaluations are never called unseen held-out tests.

The benchmark remains 90 seconds, six randomized obstacle boxes and the existing
physics ranges: payload <=1 mm for 3 continuous seconds, speed <=5 mm/s,
completed route, no crash. These are uncalibrated privileged-state simulation
metrics; they do not validate real hardware, vision or millimeter tracking.

Commands in /home/n33du/ai/drone-rl-lab (Ubuntu-Laya):

```
/usr/bin/python3 tether_sim/research_loop.py status
/usr/bin/python3 tether_sim/research_loop.py tick
/usr/bin/python3 tether_sim/research_loop.py pause
/usr/bin/python3 tether_sim/research_loop.py resume
/usr/bin/python3 tether_sim/research_loop.py dry-run
```

Pause prevents new launches; it lets the current experiment finish. Artifacts:
state/tether_research/v001.json, outbox/tether_research_1070ti.md,
configs/research_tether_research_*.json, tmp/tether-runs/ and tmp/tether-research/.
Never overwrite frozen configs or failed runs. The research lock excludes
concurrent ticks, and Puffer trainers share an OS-held GPU lease. Stops or failed
launches need review and are not silently retried. The existing trainer was
launched before that lease was added; the runner waits for its processes too.

Initial campaign limit: three trials, <=400000 transitions and <=1800 training
seconds each, plus bounded finite-seed evaluations. A future heartbeat may
prepare a new bounded campaign after evidence review. Local recurring checks
depend on the desktop scheduler and this PC/WSL being available.
