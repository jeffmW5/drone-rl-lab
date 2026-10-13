# Deep learner audit (CPU-only), 2026-10-13

Production code/config/queue/package was read only. Existing trial839 is complete. No GPU operations, builds, simulator runs, daemon actions or reserved seeds were used. Scripts and evidence in this directory are uncommitted independent diagnostics; parent decides integration.

## Confirmed observations

1. Installed source SHA256 efc5ea7537670bf89ac18347bdc97126dd6f282b9ede5598424eaeace6f8f5f4, `.../site-packages/pufferlib/pufferl.py`, stores arriving rewards/dones/values with outgoing observations/actions. C/CUDA recurrence uses reward[t+1]/done[t+1]/value[t+1]. `test_boundary.py` called the installed CPU C operator and matched an independent terminal-ending sequence within2e-7. `test_signed_credit.py` verifies positive credited action increases likelihood and negative credited action decreases likelihood under exact Gaussian PPO equations. These refute a global basic reward/action sign reversal in this tested scope.

2. Kernel deliberately loops horizon-2..0; final column advantage remains0. `pufferl.py`333-394 nevertheless normalizes and trains all columns. Actual839 final update has384 last-column sampled cells among12288 occurrences (3.125%); all12 minibatches assign them positive normalized credit, ranging0.0678..0.3849, mean0.2325. A synthetic independent-policy loss produces nonzero final-column gradient despite no reward for that outgoing action. This is a concrete invalid-cell loss issue. It does not prove the cause of flight instability: in expectation a constant state-dependent credit may contribute zero score gradient, it affects a small fraction of cells, and clipping/representation/physics/credit horizon remain possible causes.

3. `policy_update_before.pt` matches rollout behavior logprob within4.29e-6 and value within1.04e-7 on exact observations/actions, supporting correct capture alignment. Across final update, weighted normalized-credit times likelihood-change is positive overall (+0.000710). Stable states have19 unique visited cells/84 occurrences, credit1.881 but weighted likelihood-change dot -0.00156; only42.1% of stable cells increase their sampled action likelihood. This is counterevidence to the strong statement that sensible aggregate advantages guarantee every stable action is reinforced. A shared policy and12 interacting minibatches need not improve every action. It is not evidence of a loss sign bug.

4. Actual839 mean absolute final-update action changes are approximately0.000085 collective,0.000100 roll,0.000124 pitch,0.000166 yaw; max over all cells/channels0.000537. Actual recorded optimizer counter192 is consistent with fresh Adam192 steps.0% of captured raw Normal actions exceed[-1,1], so action-clipped/environment-vs-logprob mismatch is absent in this buffer.

5. CudaPolicy's collective observation version2 computes a second learned body pass with carrier-relative uniform XY position/velocity removed for collective only; the other outputs and critic retain full state. Autograd is still active. Optimizer restoration exact-policy/hash/hyperparameter guard is reasonable for audited current parameter ordering; it restores moments only, not RNG/environment/global rollout continuation. Default fresh optimizer is openly recorded. Trial839 uses that default.

## Prepared correction and tests

`learner_valid_mask.py` extracts only audited PuffeRL.train by AST and changes10 exactly-once fragments in a separate function scope, then preserves the existing record exception wrapper. It can bind one trainer instance; installed package is unchanged. Version1 leaves baseline; version2:
- normalizes raw advantages over columns0..horizon-2;
- averages policy/value/entropy loss over those valid cells;
- restricts corresponding KL/clip/importance/explained-variance diagnostics;
- retains final values as bootstrap inputs, same sampler/priority/returns/value refresh and optimizer behavior.

`test_valid_mask.py` runs a single analytic CPU table-policy update using this copied exact source and compares to independent valid-cell equations. Baseline final actor parameters change+0.009999; candidate remains0. Candidate actor and critic match independent reference exactly. `train_valid_mask_candidate.py` is the extracted candidate function for review. No flight result is implied.

Interactions: final importance is unused by recurrence; updating it is harmless. Final-column zero raw advantage contributes nothing to priority sum, so raw priority and sampler remain untouched. Excluding invalid cells changes loss denominator and normalization intentionally. It does NOT recover final outgoing transition's reward: future data collection with an extra bootstrap observation could do that, but is a separate factor requiring a separate frozen version. Shared parameters can still change final-state outputs through valid samples; the invariant is zero direct invalid-cell loss gradient.

## Additional hypotheses (unproved)

Trace gamma*lambda=.94525 gives0.355-second exponential time constant; buffer has32 observations, only31 credited actions (~0.62seconds), versus failures after5-8seconds. The remaining value bootstrap must carry delayed consequences. Recorded final cached explained variance~0.014, low LR3e-6, tiny measured mean updates and fresh Adam each bounded run make short credit horizon/insufficient uninterrupted optimization plausible. This is not evidence that gamma/lambda is wrong; longer rollout, older optimizer/gain changes already have counterevidence and should remain preserved. The parent's matched short/long x baseline/masked test isolates two factors better than another unrelated reward sweep.

## Evidence

- `839_update_alignment.json`, `839_normalized_credit_by_mb.json`
- `boundary_invariant.json`, `test_boundary.py`
- `signed_credit_invariant.json`, `test_signed_credit.py`
- `valid_mask_cpu_test.json`, `test_valid_mask.py`
- Trial839 native `tmp/cuda-research/cuda_auto_0839_takeoff/` immutable captures and result

No claim of sustained hover, strict1mm obstacle navigation, hardware accuracy or calibrated tether physics. Confidence high for source/CPU invariants and measured839buffer facts; low for causal hover explanations.
