"""Reuse immutable R16 controls and freeze one reward-only R17 adaptation."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path

root=Path('/home/n33du/ai/drone-rl-lab')
directory=root/'state/cuda_research/methodology/review_0112/velocity_reference'
directory.mkdir(parents=True,exist_ok=True)
base_rows=json.loads((root/'configs/cuda_research/methodology_queue_v000_user_abca_0_tenfold_reward.json').read_text())
rows=[]
for seed,baseline_index in ((70601,886),(70602,888),(70603,890)):
    baseline_name=f'cuda_auto_{baseline_index:04d}_takeoff'
    baseline_plan=root/f'configs/cuda_research/{baseline_name}.json'
    baseline_result=root/f'tmp/cuda-research/{baseline_name}/result.json'
    control=json.loads(baseline_plan.read_text())
    assert control['reward_version']==16 and control['seed']==seed and control['steps']==655360
    row=copy.deepcopy(next(r for r in base_rows if r['seed']==seed and r['reward_version']==16))
    for key in ('duration_version','duration_multiplier','short_reference_methodology_trial','short_prefix_reference','short_prefix_reference_sha256'):
        row.pop(key,None)
    row.update(
        methodology_trial=f'velocity_reference_reward17_seed{seed}',
        methodology_condition='velocity_reference_reward17',
        methodology_review='state/cuda_research/methodology/review_0112/review.md',
        reward_version=17, velocity_reference_version=1,
        hypothesis='Reward-only horizontal velocity reference distinguishes inward motion from equal-speed outward drift, then requests zero velocity at the hover center. Test braking and sustained hover, keeping physics and all motor control unchanged.',
        matched_control_trial=baseline_name,
        matched_control_plan=str(baseline_plan),
        matched_control_plan_sha256=hashlib.sha256(baseline_plan.read_bytes()).hexdigest(),
        matched_control_result=str(baseline_result),
        matched_control_result_sha256=hashlib.sha256(baseline_result.read_bytes()).hexdigest(),
        rejection_rule='Reject without replicated improvement in actual ground sustained hover or braking/hold without increased true failures or lost takeoffs. Existing R16 controls are reused immutable controls, not new independent validation. Verify identical initial physical metrics and unchanged simulator/learner/action settings; no reward-magnitude or post-reset endpoint promotion.',
    )
    frozen_settings=['steps','budget_seconds','eval_every_steps','evaluation_steps','validation_seeds','batch_size','bptt_horizon','learning_rate','gae_lambda','seed','resume','physics_version','training_reset_version','evaluation_reset_version','control_version','collective_observation_version','learner_valid_mask_version','value_gradient_version','integration_version','diagnostic_zone_version','selection_version','initial_throttle','optimizer_resume_version','evaluation_trace_version','pufferl_source_sha256']
    assert all(row[key]==control[key] for key in frozen_settings), 'Matched control setting mismatch'
    rows.append(row)
queue=root/'configs/cuda_research/methodology_queue_v000_user_abca_00_velocity_reference.json'
temporary=queue.with_suffix('.json.pending')
with temporary.open('x') as handle:
    handle.write(json.dumps(rows,indent=2)+'\n')
assert not queue.exists()
os.link(temporary,queue)
temporary.unlink()
plan=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=str(queue.relative_to(root)),queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),
    trials=3,requested_transitions=655360,wall_clock_cap_seconds=1800,parallel_environments=128,
    controls='Immutable completed reward16 trials 886,888,890 at the same seeds, source weights, learner and simulator settings; no redundant baseline rerun.',
    only_behavior_change='Horizontal speed cost becomes 2*||vXY-v_ref||^2, with v_ref=-displacementXY*(1/s), vector norm capped at0.5m/s. Vertical speed cost and all reward16 distance/ground/band/failure/upright terms stay fixed. Reference affects reward only; the learned policy selects all motor actions.',
    required_validation='CPU old14/15/16 parity, direction/cap/boundary/sign/dispatch and reset checks; GPU initial physical metric equivalence to matched controls; source snapshots preserved.',
    rejection_rule=rows[0]['rejection_rule'],scope='A falsifiable next hypothesis from partial survival gains and persistent drift. No demonstrated hover, calibrated physics or hardware precision claim.')
with (directory/'plan.json').open('x') as handle:handle.write(json.dumps(plan,indent=2)+'\n')
with (directory/'queue_source.py').open('x') as handle:handle.write(Path(__file__).read_text())
print(json.dumps(plan,indent=2))
