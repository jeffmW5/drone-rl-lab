"""Freeze user-requested reward16 pairs without mutating existing trials."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

root = Path('/home/n33du/ai/drone-rl-lab')
review = root / 'state/cuda_research/methodology/review_0109/fine_reward'
review.mkdir(parents=True, exist_ok=False)
base = json.loads((root / 'configs/cuda_research/methodology_queue_v000_user_abd_attitude_recovery.json').read_text())[0]
rows = []
for seed in (70601, 70602, 70603):
    for reward in (15, 16):
        row = copy.deepcopy(base)
        row.update(
            methodology_trial=f'fine_distance_reward{reward}_seed{seed}',
            methodology_condition=f'fine_distance_reward{reward}',
            hypothesis='Reward16 removes the near-hover upper-clip plateau and resolves millimetre/centimetre distance changes; compare measured ground hover and failures against reward15 with identical physics, resets, observations and learning.',
            seed=seed, reward_version=reward, training_reset_version=3,
            steps=65536, budget_seconds=600, recovery_probe_version=0,
            rejection_rule='Reject without replicated improvement in genuine ground-start sustained hover or motion/hold with no added true failures and no lost takeoffs. Initial or auto-reset endpoint altitude and larger reward are not progress. Reused identical validation is provisional.',
        )
        row.pop('recovery_models', None)
        row.pop('recovery_rollout', None)
        rows.append(row)
for offset in range(0, len(rows), 2):
    control, treatment = rows[offset:offset+2]
    excluded = {'methodology_trial', 'methodology_condition', 'reward_version'}
    assert {k:v for k,v in control.items() if k not in excluded} == {k:v for k,v in treatment.items() if k not in excluded}
queue = root / 'configs/cuda_research/methodology_queue_v000_user_abca_fine_reward.json'
with queue.open('x') as handle:
    handle.write(json.dumps(rows, indent=2) + '\n')
source = root / 'tmp/cuda-research/cuda_auto_0759_takeoff/best.pt'
plan = dict(
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    queue=str(queue.relative_to(root)), queued_trials=6,
    source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),
    pair_difference=['reward_version', 'methodology_trial', 'methodology_condition'],
    reward_definition='s(d)=sqrt(d*d+0.01**2)-0.01; C=4*s(z-0.8)+s(dx)+s(dy)+2*speed**2; core=0.95*A/(1+C)+0.05*upright. Inside square: (1-D)*core-ground_cost. In 1<D<2 band: -(D-1)-ground_cost. At true failure: -1.',
    units='positions/distances in metres; speeds in metres per second; each reward is paid at 50 Hz',
    scope='No controller, action, physics, benchmark or angular-rate-penalty change. New queue has priority after the current healthy reset6 arm; remaining frozen recovery arms are retained.',
    hypothesis='Clipping and squared metre errors hid fine position differences. Removing that plateau and using a 1 cm smooth absolute scale may provide a useful learning signal; this does not prove PPO can learn stabilizing feedback.',
    rejection_rule=rows[0]['rejection_rule'],
    missing_evidence='CPU reward invariants do not establish learned hover or hardware accuracy. Three validation seeds repeat one identical physical condition; audit seeds stay reserved.',
)
with (review / 'plan.json').open('x') as handle:
    handle.write(json.dumps(plan, indent=2) + '\n')
with (review / 'queue_source.py').open('x') as handle:
    handle.write(Path(__file__).read_text())
print(json.dumps(plan, indent=2))
