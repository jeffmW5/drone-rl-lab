# Local 1070 Ti research loop

Current user objective: continuous C/CUDA pure learned takeoff/hover research.
Methodology improvement now uses the official Hugging Face papers skill through
the heartbeat research agent. Follow tether_sim/HF_METHODOLOGY_LOOP.md after each
three completed trials or a novel failure; save sources, review and versioned
hypotheses in state/cuda_research/methodology/. The GPU daemon executes trials;
the agent reads methods and assesses evidence before changing experiments.
scripts/cuda_autoresearch.py run is the active serial scheduler; it launches the
next bounded trial immediately with no daily cap and no timed playback delay.
state/cuda_research/loop.json records all trials; outbox/tether_cuda_loop.md reports
phase and provisional incumbent. Old MuJoCo plans below are archived references.
Commands: python3 scripts/cuda_autoresearch.py status / pause / resume.
Pause lets healthy active training finish and blocks subsequent launches.
One daemon lock and the shared GPU lease prevent duplicate trainers/compilers.
Default trials <=65536 transitions /90s elapsed learner budget plus finite evaluations;
versioned methodology queuev003 tests65536 versus262144 transitions with a600s
elapsed learner cap for both arms (including periodic evaluation).
three reused validation seeds. Initial and periodic checkpoints are preserved.
Promotion uses crashes, prototype sustained hover, takeoffs and destination error;
prototype hover is <=10cm and <=0.2m/s for1s, not the original strict1mm criterion.
Automatic phases takeoff->hover->destination require zero crashes,>=18/24 hover
successes and mean error<100mm. These are provisional curriculum gates. No
hardware accuracy or original obstacle-route completion claim. Three consecutive
process failures pause for review; an agent reviews batches via the heartbeat.

2026-10-12 user update: daily training-launch cap removed from the runner and
heartbeat. One GPU trainer at a time, bounded individual trials, frozen artifacts
and review of failed runs remain required. Historical allowance dates below
describe earlier decisions and no longer delay new trials.

Active default plan is configs/tether_research_1070ti_v002.json, with separate
state/tether_research/v002.json and outbox/tether_research_1070ti_v002.md.
This bounded one-trial campaign tests matched residual training with opt-in
attitude_gain=.012 and 1ms integration. Its parent validation is the corrected
frozen-policy diagnosticv002 at the same settings, not the old2ms benchmark.
Original v001 failed/queued trials remain preserved. Legacy gain defaults=.015.

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

2026-10-11 recovery: campaignv001 longer-training trial failed numerically;
default tick now halts at failed trials. See outbox/tether_research_recovery.md
and state/tether_research/recovery_v001.json for the separate1ms integration
diagnostic. It is the third launch in the24h allowance. Do not start another
trainer or the original third trial until that allowance returns. After this
diagnostic completes, review it and create a new versioned plan if justified;
never compare its1ms results as if they shared the original2ms benchmark.

2026-10-12 user instruction: remove angular-velocity penalty. Reward7 equals baseline2 with omega squared cost removed, upright term retained. Legacy reward versions frozen. Queuev023 tests three seeds; low-priority queuev999 maintains reward7 for the already-running scheduler while newer methodology queues take precedence. New daemon default source also7; running daemon is not restarted. Before low-priority continuation exhaustion ensure no return to reward2; future campaigns must honor user removal except explicit frozen control arms. Geometry/observation/control/evaluation unchanged; CPU cost tests passed. At this check332 completed trials,total31784960 training transitions; recent65536-step runs~63s/~1040 transitions/s including in-budget evaluation. No achieved hover/hardware precision claim.

2026-10-12 user steering: prioritize reward iterations aimed at actual target approach/braking until repeatable measured improvement. High-priority frozen queuev000_user_approach_stop compares reward8 versus7 at39001-39003; leave healthy runs undisturbed. After completion review crashes/takeoffs/airborne acquisition/hover and paired updated-checkpoint destination error. If it fails, diagnose reward/trajectory evidence and create a new versioned reward batch rather than silently resume the blind LR/seed continuation sweep. Do not change angular-rate removal, controls or benchmark during reward comparison. Reserve audit seeds. Reward improvements are not hardware1mm accuracy. Preserve failed reward versions; do not claim initialbest as learned improvement.
