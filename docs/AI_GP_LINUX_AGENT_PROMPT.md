# AI-GP Linux Agent Prompt

Paste this prompt into a Linux agent working in the canonical repo:

```text
Work autonomously in /home/jeff/drone-rl-lab on the AI Grand Prix PPO stack.
Pull the latest origin/master first. Read CLAUDE.md, docs/AI_GP_AGENT_HANDOFF.md,
docs/AI_GP_CONTROL_CALIBRATION.md, docs/AI_GP_RUNPOD.md, ai_gp_rl/env.py,
train_ai_gp.py, and tests/test_ai_gp_swift_env.py before editing.

Objective: implement and train one topology-correct structured-state PPO
teacher for the measured AI-GP six-gate course. This is not a hyperparameter
study and not a student-distillation task.

Measured source data:
- NED gate poses and 2.72 m dimensions are defined in
  scripts/run_ai_gp_bounded_windows.py as KNOWN_TRACK_GATES_NED.
- Positive policy roll maps directly to positive simulator roll command.
- The current surrogate track has the wrong gate-0-to-gate-1 turn direction.
- Releasing more authority and applying up to 200% thrust gain produced zero
  repeatable gate-0 passes.
- A missed gate must not leave the last non-neutral command active.

Required implementation:
1. Put the six-gate AI-GP track in one reusable training-side definition.
2. Add an explicit, documented, unit-tested NED-to-surrogate transform. Preserve
   the measured forward, vertical, and roll/lateral conventions. If the existing
   ground plane requires an altitude offset, make that offset explicit.
3. Support the measured 2.72 m gate width and height.
4. Detect active-gate plane crossing outside the aperture. Expose `missed_gate`
   in info, apply a configurable large penalty, and terminate that episode.
5. Add focused tests for:
   - transformed gate order and geometry
   - valid inside-aperture crossing
   - outside-aperture missed-gate termination
   - finite observations/rewards on the six-gate track
6. Add one config named like
   configs/ai_gp_018_real_track_teacher_10m.yaml. Use the existing Swift teacher
   observation and PPO implementation. Keep raw pixels and student distillation
   out of scope.
7. Run the relevant CPU tests and full test suite.
8. Run CUDA smoke, then train 10 million interactions on the available GPU.
   On the Linux VM, use existing RunPod access without printing secrets. Check
   scripts/manage_pod.sh and the persisted pod/SSH conventions in CLAUDE.md.
9. Run deterministic nominal evaluation and at least three randomized seeds.
   Report gate-0 pass rate, mean gates, finish rate, missed-gate rate, collision
   rate, out-of-bounds rate, vertical-runaway rate, elapsed training time, GPU,
   and interactions/second.
10. Write a short benchmark report under docs/. Clearly label the environment
    as topology-correct but dynamics-unfitted unless command-aligned Windows
    telemetry was actually used to fit it.

Acceptance target for this first benchmark:
- at least 95% deterministic nominal gate-0 passage
- mean gates passed >= 2
- collision rate < 20%
- no persistent vertical runaway

If the first 10M run fails, inspect trajectory time-series and make at most one
evidence-based environment/reward correction, then rerun. Do not launch a broad
matrix.

Do not modify unrelated drone-racing code. Do not claim Windows simulator
success from surrogate results. Do not export into jeffmW5/ai-grand-prix-stack
until the teacher meets the acceptance target. Keep generated checkpoints and
large telemetry out of git, but state their exact paths.

Finish by updating the handoff/docs with measured results, committing all scoped
code/config/test/doc changes, pushing them, and reporting the commit hash,
commands run, metrics, artifact paths, and remaining blockers.
```
