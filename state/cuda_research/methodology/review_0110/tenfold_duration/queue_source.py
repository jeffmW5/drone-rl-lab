"""Freeze the user's tenfold duration request; do not modify active trials."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

root = Path('/home/n33du/ai/drone-rl-lab')
review = root / 'state/cuda_research/methodology/review_0110/tenfold_duration'
review.mkdir(parents=True, exist_ok=False)
short_queue = root / 'configs/cuda_research/methodology_queue_v000_user_abca_fine_reward.json'
short_rows = json.loads(short_queue.read_text())
rows = []
for source in short_rows:
    row = copy.deepcopy(source)
    row.update(
        methodology_trial=f"fine_distance_10x_reward{row['reward_version']}_seed{row['seed']}",
        methodology_condition=f"fine_distance_10x_reward{row['reward_version']}",
        methodology_review='state/cuda_research/methodology/review_0110/review.md',
        hypothesis='User-requested tenfold duration: recreate the same short-run prefix then continue within one uninterrupted optimizer/environment run; compare reward15/16 with unchanged physical and hover criteria.',
        steps=10*source['steps'], budget_seconds=1800,
        rejection_rule='The first short reward16 seed added escapes and no hover. Requested longer exposure is a falsifiable test, not a promoted method. Require replicated ground sustained-hover improvement with no additional true failures or lost takeoffs; inspect initial/periodic/final and verify exact short prefixes. Do not rank changed reward magnitudes or post-reset altitude.',
        duration_version=2,
        duration_multiplier=10,
        short_reference_methodology_trial=source['methodology_trial'],
    )
    if row['seed']==70601:
        trial = 881 if row['reward_version']==15 else 882
        row['short_prefix_reference'] = str(root / f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff/policy.pt')
        row['short_prefix_reference_sha256'] = hashlib.sha256(Path(row['short_prefix_reference']).read_bytes()).hexdigest()
    rows.append(row)
for i in range(0, len(rows), 2):
    excluded = {'methodology_trial','methodology_condition','reward_version','short_reference_methodology_trial','short_prefix_reference','short_prefix_reference_sha256'}
    assert {k:v for k,v in rows[i].items() if k not in excluded} == {k:v for k,v in rows[i+1].items() if k not in excluded}
    assert rows[i]['steps']==rows[i+1]['steps']==655360
queue = root / 'configs/cuda_research/methodology_queue_v000_user_abca_0_tenfold_reward.json'
with queue.open('x') as handle:
    handle.write(json.dumps(rows, indent=2) + '\n')
plan = dict(
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    user_request='Train for 10x longer',
    queue=str(queue.relative_to(root)),
    queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),
    short_transitions=65536, requested_transitions=655360, budget_seconds_per_trial=1800,
    parallel_environments=128, control_hz=50,
    nominal_seconds_per_environment=655360/128/50,
    evaluation_every_transitions=16384, evaluation_episode_steps=250,
    duration_comparison='New runs recreate identical starting weights/seed/fresh Adam and then continue within the same optimizer/environment state. Prefix equality is expected and must be checked; it is not assumed from configuration alone.',
    pair_difference='reward_version and identifying/reference labels only; fixed physics5/reset3/observation2/control2/mask2/learningrate3e-6',
    dispatch='Queue takes priority only after the current healthy trainer drains; pending short and recovery plans remain frozen and retained.',
    rejection_rule=rows[0]['rejection_rule'],
    scope='Explicit user duration request despite short-run counterevidence. No hover, reward16 promotion, held-out validation or hardware accuracy claim.',
)
with (review/'plan.json').open('x') as handle:
    handle.write(json.dumps(plan, indent=2) + '\n')
with (review/'queue_source.py').open('x') as handle:
    handle.write(Path(__file__).read_text())
print(json.dumps(plan, indent=2))
