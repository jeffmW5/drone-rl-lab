# AI-GP Linux Agent Prompt

Paste this prompt into a Linux agent working in the canonical repo:

```text
Work autonomously in /home/jeff/drone-rl-lab on the AI Grand Prix structured
controller training stack.

Pull latest origin/master first. Read CLAUDE.md, docs/AI_GP_AGENT_HANDOFF.md,
docs/AI_GP_TRANSFER_TRAINING_HANDOFF_2026_06_28.md,
docs/AI_GP_VISION_TRANSITION_PLAN.md,
exports/ai_gp/ai_gp_windows_transfer_handoff_2026_06_28.json,
configs/ai_gp_040_near_gate_teacher_bc_30m.yaml, ai_gp_rl/env.py,
ai_gp_rl/track.py, train_ai_gp.py, distill_ai_gp_teacher.py, and the existing
AI-GP export/evaluation scripts before editing.

Objective: train `ai_gp_041_windows_transfer_gate2_hardcase_30m`, a better
structured-state teacher/controller for Windows AI-GP transfer. Do not start a
vision-only pilot yet. Vision needs a stronger structured teacher first.

Starting point:
- policy checkpoint: results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt
- current export: exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json
- Windows hard-case handoff:
  exports/ai_gp/ai_gp_windows_transfer_handoff_2026_06_28.json

Windows evidence:
- `040` solves the surrogate but not the Windows simulator.
- Manual tuning and a 12-config sweep topped out at active gate index 2.
- Top sweep runtime was thrust 1.12, roll 2.00, pitch 1.00, yaw 2.00.
- Pitch boosting hurt. Roll 2.00 helped. Useful thrust was about 1.10-1.12.
- Simulator gate normals did not improve transfer in the tested y-axis mode.
- Best failures are gate-1/gate-2 collision hard cases at about 3.8-5.6 m/s.

Required work:
1. Create a scoped `041` config and result name.
2. Start from the `040` best policy, not from scratch.
3. Use the handoff JSON hard cases to bias training/evaluation toward gate 1
   and gate 2 recovery/crossing states while preserving full-course surrogate
   behavior.
4. Prefer an anchored PPO or BC-plus-PPO update over broad reward sweeps.
5. Keep pitch runtime assumptions at 1.00; do not train around pitch boost.
6. Keep generated checkpoints and bulk telemetry out of git.
7. Export a new structured JSON only after surrogate checks pass.
8. Update docs with exact commands, metrics, artifact paths, and blockers.
9. Commit and push scoped config/code/test/doc changes.

Minimum acceptance before asking Windows to retest:
- surrogate nominal success does not materially regress from `040`
- Windows retest target: gate0 pass rate >= 0.90
- Windows retest target: mean max active gate index > 2.0
- Windows retest target: best active gate index >= 3

Windows retest baseline after export:

python -B scripts/run_ai_gp_structured_windows.py --continuous --duration 30 \
  --target-gates 0 --allow-gate-plane-miss \
  --thrust-multiplier 1.12 --roll-rate-multiplier 2.00 \
  --pitch-rate-multiplier 1.00 --yaw-rate-multiplier 2.00 \
  --run-id watch_structured_041_baseline

Do not claim Windows success from surrogate metrics. Report surrogate metrics,
export path/SHA, training runtime/GPU, and the exact Windows retest command.
```
