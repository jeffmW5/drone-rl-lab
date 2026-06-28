# AI-GP Transfer Training Handoff - 2026-06-28

## Current State

`040` is a strong structured-state surrogate policy, but Windows transfer is
not solved. Runtime multiplier tuning is not enough. `041` has now been trained
from the Windows gate-1/gate-2 hard-case handoff and exported as a Windows A/B
candidate, but it is not a clean surrogate replacement for `040`.

Tracked Windows handoff:

```text
exports/ai_gp/ai_gp_windows_transfer_handoff_2026_06_28.json
```

Handoff SHA-256:

```text
ec359ac473a851d5058ad9fea3b58f68db33e0bc63a17dc518635c61ba52592e
```

Policy export:

```text
exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json
1581cc4cb0a0753eb7ba87ae1e34a09dd8d7badbd048b1fce823a28775d9da60
```

Windows A/B candidate export:

```text
exports/ai_gp/ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json
583762c48fa7e24a7a5ea69dfa1269104a54481b61f3282143e9eaabb4f42ca7
```

## Windows Findings

- Manual tuning and a 12-config sweep topped out around active gate index 2.
- Best sweep config: thrust `1.12`, roll `2.00`, pitch `1.00`, yaw `2.00`.
- Prior stable watch baseline: thrust `1.10`, roll `2.00`, pitch `1.00`, yaw
  `2.00`.
- Pitch boosts made behavior worse.
- Roll `2.00` was better than `1.75`.
- Useful thrust was about `1.10-1.12`; thrust `1.50` caused altitude aborts.
- Simulator gate normals in tested y-axis mode were worse than exported normals.
- Most useful failures are gate-1/gate-2 collisions at about `3.8-5.6 m/s`.

## Training Objective

Train:

```text
ai_gp_041_windows_transfer_gate2_hardcase_30m
```

Tracked config:

```text
configs/ai_gp_041_windows_transfer_gate2_hardcase_30m.yaml
```

Start from:

```text
results/ai_gp_040_near_gate_teacher_bc_30m/best_policy.pt
```

Goal: improve Windows transfer past gate 2 without sacrificing `040` surrogate
behavior.

Recommended approach:

- Use the handoff JSON hard cases for focused gate-1/gate-2 training states.
- Prefer anchored PPO or BC-plus-PPO from `040`.
- Preserve nominal/full-course surrogate performance.
- Keep pitch multiplier assumptions at `1.00`.
- Do not start vision training in this run.

## 041 Training Result

`ai_gp_041_windows_transfer_gate2_hardcase_30m` trained on RunPod RTX 3090 for
`30,146,560` steps in `984 s`. It started from `040`, used anchored PPO, and
focused near-gate starts on active gates 1 and 2.

Training artifacts:

```text
results/ai_gp_041_windows_transfer_gate2_hardcase_30m/best_policy.pt
results/ai_gp_041_windows_transfer_gate2_hardcase_30m/ai_gp_structured_policy.json
```

Dense surrogate validation:

```text
nominal:   100.00% success, 6.00 / 6 gates, 0.00% collision
seed 1001: 98.24% success, 5.97 / 6 gates, 0.00% collision
seed 1002: 99.02% success, 5.99 / 6 gates, 0.00% collision
seed 1003: 99.22% success, 5.99 / 6 gates, 0.00% collision
average:   98.83% success, 5.99 / 6 gates, 0.00% collision
```

Compared with `040`, this is a small randomized surrogate regression
(`99.22%` -> `98.83%` average success), but it preserves nominal completion and
zero collisions. Use it for Windows simulator A/B testing against `040`; do not
call it promoted until the Windows sim shows it clears farther than gate 2.

## Acceptance

Before Windows retest:

- Surrogate nominal success does not materially regress from `040`.
- Export a new structured JSON with test vectors and SHA.
- Document training runtime, GPU, commands, and artifact paths.

Windows retest target:

- gate0 pass rate `>= 0.90`
- mean max active gate index `> 2.0`
- best active gate index `>= 3`

Preferred A/B retest command:

```powershell
& 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe' -B scripts\run_ai_gp_policy_ab_windows.py `
  --attempts-per-policy 5 `
  --duration 30 `
  --thrust-multiplier 1.12 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id structured_ab_040_041_YYYYMMDD_HHMMSS
```

Retest command baseline:

```powershell
& 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv\Scripts\python.exe' -B scripts\run_ai_gp_structured_windows.py `
  --continuous `
  --duration 30 `
  --target-gates 0 `
  --allow-gate-plane-miss `
  --thrust-multiplier 1.12 `
  --roll-rate-multiplier 2.00 `
  --pitch-rate-multiplier 1.00 `
  --yaw-rate-multiplier 2.00 `
  --run-id watch_structured_041_baseline
```
