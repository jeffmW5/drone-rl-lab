# Methodology review 0001

Last reviewed trial index: 16. Scope: reduced C/CUDA controls2 takeoff prototype,
reward2, three repeatedly reused seeds (24 domains). Rankings are provisional.

Observation: controls2 trials 5-16 retained best_step=0; no learned checkpoint
surpassed the initial reference. Trial16 eval_32768.json records 0 crashes,
24 takeoffs, 0 sustained hover, mean final error 7941.763mm; eval_65536.json
records 0 crashes, 24 takeoffs, 0 hover, 6488.628mm. Selected initial reference
is 1080.487mm. These are not learned improvements. Source: native WSL
tmp/cuda-research/cuda_auto_0016_takeoff and state/cuda_research/loop.json.
Confidence high for these recorded evaluations; low for explanations of failure.

## Sources read

- Learning to Fly in Seconds, arXiv:2311.13081v2, sections IV and VI:
  https://arxiv.org/html/2311.13081v2 . HF full markdown and metadata retrieved
  successfully; author implementation https://github.com/arplaboratory/learning-to-fly .
  It combines off-policy TD3, a penalty curriculum, action history and privileged
  critic information. This suggests several separate experiments, not a guarantee
  that substituting TD3 alone will solve the tether task.
- Curriculum-based Sample Efficient Reinforcement Learning for Robust Stabilization
  of a Quadrotor, arXiv:2501.18490, sections III and IV-B:
  https://arxiv.org/html/2501.18490 . HF returned404, original arXiv HTML used.
  Its staged initialization and bounded exploration/truncation motivate testing
  episode design before repeatedly changing optimizer settings.

These papers concern quadrotor control; their reported successes do not establish
tether/contact accuracy or motors-off ground takeoff in our simulator. Our actor
already receives simulated motor states; simply adding them is not a new method.
Our PPO stack differs from the first paper's TD3 implementation. No author code
was executed, and no hardware/audit seeds were used.

## Next falsifiable hypothesis (proposed, not yet implemented)

Run a versioned training-only bounded-flight termination ablation: end a training
episode when drone position leaves a declared region around the initial takeoff
goal. Keep deterministic evaluation unchanged, including ground starts and all
flight failures; distinguish reset/truncation from physical crashes. The mechanism
is reducing rollout time spent far from recoverable takeoff states. Freeze the
region, reward2, controls2, observation98, budget and baseline before launch.
Replicate baseline and candidate with matched three training seeds. Reject any
increase in evaluation crashes or loss of takeoffs; require improved sustained
hover and paired destination behavior. Preserve every initial/periodic/final result.
Do not infer better physics from a training curriculum improvement.

Before implementation, inspect representative worst trajectories and quantify
time spent outside the proposed region. If failure occurs near the goal instead,
this hypothesis is not supported and should be replaced with documented evidence.
Separate future hypotheses: penalty scheduling, action-history observation,
off-policy sample reuse. Never bundle these into the same ablation.
