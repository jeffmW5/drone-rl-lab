# Hugging Face methodology review

User requested Hugging Face skills to improve the continuous research loop.
Use the official huggingface-papers skill installed at
C:/Users/n33du/.codex/skills/huggingface-papers/SKILL.md. Read it before use.
This is the research agent's paper-review stage; the serial GPU daemon remains
the trial executor. Fetching papers does not change the controller automatically.

After each batch of three completed trials since the last review, or a novel
failure, inspect initial, periodic and final evaluations and actual trajectories.
Check state/cuda_research/methodology/ for the last reviewed trial index before
repeating a review. Record a review even if no defensible new method is found.
Do not label an unchanged initial checkpoint as learned progress.

1. Write the observed failure and evidence paths first. Separate observation from
   explanation; distinguish optimizer failure, task/reward mismatch and physics defects.
2. Search HF papers for that failure. Read 2-4 relevant original methods, not only
   abstracts, AI summaries or trending rankings. Fetch HF markdown and metadata;
   use original arXiv HTML/PDF and author code when HF has no full text.
   Public reads need no token. Never invent a token, publish or claim authorship.
3. Save retrieval URLs, date, paper versions and available author code links in a
   new immutable methodology directory. Use scripts/hf_methodology.py, e.g.
   python3 scripts/hf_methodology.py --query "quadrotor reinforcement learning stabilization curriculum" --paper 2311.13081 --output state/cuda_research/methodology/review_0001
4. Write review.md alongside retrieval.json: last reviewed trial index, concrete
   findings, exact paper sections supporting the method, applicability to a
   tethered Crazyflie, simulator assumptions, counterevidence and confidence.
   Paper results are not results from this simulator. Missing evidence stays explicit.
5. Propose one falsifiable change, expected mechanism, control baseline and rejection
   rule. Create a new frozen version with source/review reference in its plan.
   Keep reward, observation, action and physics versions separate; incompatible
   checkpoint weights must not be resumed. Do not silently retune multiple factors.
6. Test under the existing one-GPU lease and bounded trials. No daily launch cap.
   Preserve healthy training and paused state; source changes take effect only
   after a healthy trial drains. Replicate promising changes on training seeds,
   paired reused validation and unchanged criteria before promoting. Audit seeds
   remain reserved. Reject increased crashes, lost takeoff/hover or worse paired
   destination behavior, regardless of reward. Report uncertainty and regressions.

Priorities to investigate: survival/hover curriculum, reward clipping and scale,
action parameterization/exploration, termination/reset design, motor observation
and actuator dynamics. Check tether attachment/contact fidelity before escalating
to obstacle-navigation claims. Never add a hidden PID helper to pure learned mode.
No automatic downloads/execution of author code. Inspect adaptations first.
If retrieval is unavailable, preserve the failure and continue the existing
bounded campaign; do not substitute fabricated literature.
