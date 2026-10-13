"""Read complete periodic artifacts without interacting with active GPU training."""
import datetime
import hashlib
import json
import math
from pathlib import Path

root=Path('/home/n33du/ai/drone-rl-lab')
run=root/'tmp/cuda-research/cuda_auto_0885_takeoff'
status=json.loads((run/'status.json').read_text())
rows=[]
for path in sorted(run.glob('eval_*.json'),key=lambda p:p.name):
    if not path.stem[5:].isdigit() or int(path.stem[5:])>=status['steps']:
        continue
    before=path.read_bytes()
    data=json.loads(before)
    assert before==path.read_bytes(), 'Periodic artifact changed during read'
    finite=True
    def scan(value):
        global finite
        if isinstance(value,float) and not math.isfinite(value): finite=False
        elif isinstance(value,dict):
            for item in value.values():scan(item)
        elif isinstance(value,list):
            for item in value:scan(item)
    scan(data)
    terminals=[event for item in data['runs'] for event in item.get('evaluation_terminal_trace',{}).get('first_terminal_by_lane',[]) if event is not None]
    rows.append(dict(step=int(path.stem[5:]),file=str(path),sha256=hashlib.sha256(before).hexdigest(),all_serialized_numbers_finite=finite,
        crashes=data['crashes'],takeoffs=data['takeoffs'],hover=data.get('zone_hover_successes',0),altitude_error_mm=data.get('zone_hover_error_mm'),
        peak_hold_s=max(max(item.get('zone_hover_hold_s',[0])) for item in data['runs']),first_terminal=terminals[0] if terminals else None))
rows.sort(key=lambda row:row['step'])
evidence=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),trial='cuda_auto_0885_takeoff',status_at_capture=status,
    requested_transitions=655360,budget_seconds=1800,periodic=rows,
    scope='Partial long reward15 control, not a final result or reward16 comparison. Complete stable immutable periodic artifacts only; active evaluation was excluded. No GPU call or trainer/config/source modification. Reused validation repeats one physical condition.',
    observation='First 65k reproduces short881 exactly. Evaluations subsequently contain true zone failures; there is no sustained hover in the reviewed prefix. The healthy user-requested long run continues. Known failure mode, not proof of wrong physics.',
    last_completed_formal_review_index=882,new_completed_since_review=[883,884],methodology_review_due_after='The next completed trial or a novel failure mode; do not count partial checkpoints as completed trials.')
dest=root/'state/cuda_research/methodology/review_0110/tenfold_duration/heartbeat_progress_885_v001.json'
with dest.open('x') as handle:handle.write(json.dumps(evidence,indent=2)+'\n')
with dest.with_suffix('.py').open('x') as handle:handle.write(Path(__file__).read_text())
print(json.dumps(dict(active=status,reviewed_through=rows[-1]['step'],first_failure_checkpoint=next((row['step'] for row in rows if row['crashes']),None),peak_hover=max(row['peak_hold_s'] for row in rows),all_hover=sum(row['hover'] for row in rows)),indent=2))
