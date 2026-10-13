"""Catch up immutable evaluations after the review gap, without GPU access."""
import datetime
import hashlib
import json
import math
from pathlib import Path

root=Path('/home/n33du/ai/drone-rl-lab')
out=root/'state/cuda_research/methodology/review_0112/catchup'
out.mkdir(parents=True,exist_ok=True)
state=json.loads((root/'state/cuda_research/loop.json').read_text())
rows=[]
def metric(value):
    runs=value.get('runs',[])
    frames=[frame for run in runs for frame in run.get('trajectory',[])]
    terminals=[event for run in runs for event in run.get('evaluation_terminal_trace',{}).get('first_terminal_by_lane',[]) if event is not None]
    return dict(crashes=value.get('crashes'),takeoffs=value.get('takeoffs'),hover=value.get('zone_hover_successes'),altitude_error_mm=value.get('zone_hover_error_mm'),
        peak_hold=max((max(run.get('zone_hover_hold_s',[0])) for run in runs),default=0),
        first_failure_time=terminals[0]['absolute_time_s'] if terminals else None,
        last_sample_speed=math.sqrt(sum(x*x for x in frames[-1]['velocities'][0])) if frames else None,
        scope='Endpoint and sparse frames can follow autoreset; absence of a recorded terminal is not survival beyond the five-second benchmark.')
for trial in state['trials']:
    index=int(trial['name'].split('_')[2])
    if not 886<=index<=930:continue
    directory=root/trial['run'];result=directory/'result.json'
    if trial['status']!='complete' or not result.exists():
        rows.append(dict(index=index,status=trial['status'],missing_result=not result.exists()));continue
    data=json.loads(result.read_text());plan_path=root/trial['plan'];plan=json.loads(plan_path.read_text()); selection=json.loads((directory/'selection.json').read_text())
    evaluations=[]
    for path in sorted(directory.glob('eval_*.json')):
        if not path.stem[5:].isdigit():continue
        raw=path.read_bytes();value=json.loads(raw)
        assert raw==path.read_bytes(),'Completed evaluation changed while read'
        evaluations.append(dict(step=int(path.stem[5:]),sha256=hashlib.sha256(raw).hexdigest(),**metric(value)))
    evaluations.sort(key=lambda row:row['step'])
    rows.append(dict(index=index,name=trial['name'],methodology=trial.get('methodology_trial'),status=trial['status'],
        versions={key:plan.get(key) for key in ['reward_version','physics_version','training_reset_version','collective_observation_version','learner_valid_mask_version','recovery_probe_version']},
        steps=data['steps'],requested_steps=plan['steps'],seconds=data['training_seconds'],budget=plan['budget_seconds'],censored=data['steps']<plan['steps'],
        initial=metric(data['initial']),final=metric(data['final']),selected=metric(selection['best']),selected_step=selection['best_step'],periodic=evaluations,
        source_paths=dict(plan=str(plan_path),result=str(result),selection=str(directory/'selection.json')),
        source_sha256=dict(plan=hashlib.sha256(plan_path.read_bytes()).hexdigest(),result=hashlib.sha256(result.read_bytes()).hexdigest(),selection=hashlib.sha256((directory/'selection.json').read_bytes()).hexdigest())))
artifact=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_reviewed_index=930,previous_formal_review_index=885,
    gap='The research-agent usage-limit failure interrupted the review cadence while the independent daemon continued. This catch-up does not claim timely intermediate reviews.',
    trials=rows,scope='Read-only completed artifact audit, not a new training/physics experiment. Reserved audit/hardware seeds are not used; reused validation is provisional.')
with (out/'evaluations_886_930.json').open('x') as handle:handle.write(json.dumps(artifact,indent=2)+'\n')
with (out/'collector_source.py').open('x') as handle:handle.write(Path(__file__).read_text())
print(json.dumps(dict(count=len(rows),completed=sum(row['status']=='complete' for row in rows),final_hover_trials=[row['index'] for row in rows if row.get('final',{}).get('hover')],
    periodic_hover_trials=[row['index'] for row in rows if any(ev.get('hover') for ev in row.get('periodic',[]))],max_hold=max((max([row['final']['peak_hold']]+[ev['peak_hold'] for ev in row.get('periodic',[])]) for row in rows if 'final' in row),default=0),
    methods=[(row['index'],row.get('methodology'),row.get('final',{}).get('crashes'),row.get('final',{}).get('hover')) for row in rows if row.get('methodology')]),indent=2))
