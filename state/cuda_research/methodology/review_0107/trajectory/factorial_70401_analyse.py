import hashlib,json,math,datetime
from pathlib import Path
import torch
ROOT=Path('/home/n33du/ai/drone-rl-lab');OUT=ROOT/'tmp/deep_audit_physics';stem=OUT/'factorial_70401_trajectory'
assert not stem.with_suffix('.json').exists(), 'Refuse overwrite'
read_time=datetime.datetime.now(datetime.timezone.utc).isoformat()

def summarise(d,trial,label,step,source):
 rows=[];frame_hashes=[]
 for run in d.get('runs',[]):
  fr=[]
  for f in run['trajectory']:
   p=f['positions'][0];v=f['velocities'][0];q=f['orientation'];w,x,y,z=q[:4]
   speed=math.sqrt(sum(i*i for i in v));extent=max(abs(p[0]),abs(p[1]));Rzz=1-2*(x*x+y*y)
   fr.append(dict(time_s=(f['step']+1)*.02,carrier=p,velocity=v,speed_m_s=speed,extent_m=extent,height_error_m=p[2]-.8,tilt_deg=math.degrees(math.acos(max(-1,min(1,Rzz)))),roll_deg=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),pitch_deg=math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x))))),rates_rad_s=q[4:],motors_N=f['motors'],terminal=f['terminal'],sample_meets_zone_state=(extent<=1 and abs(p[2]-.8)<.1 and speed<.2 and not f['terminal'])))
  frame_hashes.append(hashlib.sha256(json.dumps(fr,sort_keys=True).encode()).hexdigest())
  rows.append(dict(seed=run['seed'],frames=fr,full_eval_zone_hold_s=run['zone_hover_hold_s'],zone_counts=run['zone_diagnostics']))
 first=rows[0]['frames']
 return dict(trial=trial,label=label,training_step=step,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),crashes=d.get('crashes'),takeoffs=d.get('takeoffs'),zone_hover_successes=d.get('zone_hover_successes'),zone_escapes=d.get('zone_escapes'),mean_carrier_altitude_error_mm=d.get('zone_hover_error_mm'),payload_final_error_mm=d.get('mean_final_error_mm'),max_full_eval_zone_hold_s=max(h for r in rows for h in r['full_eval_zone_hold_s']),unique_saved_trajectory_domains=len(set(frame_hashes)),saved_sample_count=len(first),sample_max_extent_m=max(f['extent_m'] for f in first),sample_max_speed_m_s=max(f['speed_m_s'] for f in first),sample_max_tilt_deg=max(f['tilt_deg'] for f in first),sample_max_height_m=max(f['carrier'][2] for f in first),sample_stable_count=sum(f['sample_meets_zone_state'] for f in first),sample_terminal_count=sum(f['terminal'] for f in first),last_saved_sample=first[-1],trajectory_runs=rows)

result=dict(retrieved_utc=read_time,scope='Saved deterministic5s evaluation artifacts, full14node physics5 stillair zone1, reusedseeds51011..13. Sparse saved frames do not measure continuous holds; full-evaluation hold counters used. No new simulator/GPU run, production edit or trial interruption.',trials={},rows=[])
for trial in (850,851):
 folder=ROOT/f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff'
 config=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text())
 state=dict(plan=config['research_plan'],status_at_read=status,final_available=(folder/'result.json').exists(),selection_available=(folder/'selection.json').exists())
 if state['final_available']:
  d=json.loads((folder/'result.json').read_text());state['result_steps']=d['steps'];result['rows'].append(summarise(d['initial'],trial,'initial',0,folder/'result.json'))
  result['rows'].append(summarise(d['final'],trial,'final',d['steps'],folder/'result.json'))
 if state['selection_available']:state['selection_metadata']={k:v for k,v in json.loads((folder/'selection.json').read_text()).items() if k!='best'}
 for source in sorted(folder.glob('eval_*.json'),key=lambda p:int(p.stem[5:])):
  d=json.loads(source.read_text());result['rows'].append(summarise(d,trial,'periodic',int(source.stem[5:]),source))
 result['trials'][str(trial)]=state
# Confirm initial function equality if incomplete851 has no initial evaluation artifact.
a=torch.load(ROOT/'tmp/cuda-research/cuda_auto_0850_takeoff/initial.pt',map_location='cpu',weights_only=True)
b=torch.load(ROOT/'tmp/cuda-research/cuda_auto_0851_takeoff/initial.pt',map_location='cpu',weights_only=True)
result['initial_policy_weights_exactly_equal']=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
result['initial_policy_comparison_scope']='Weights and architecture plan match;850 initial trajectory is shared reference if851 initial evaluation not written until completion.'
result['rows'].sort(key=lambda r:(r['trial'],r['training_step'],r['label']))
# Compare available equal training steps including0,65k,262k and all saved later checkpoints.
comparisons=[]
for n in sorted(set(r['training_step'] for r in result['rows'])):
 pairs={t:next((r for r in result['rows'] if r['trial']==t and r['training_step']==n and (r['label']=='periodic' or n==0)),None) for t in (850,851)}
 if all(pairs.values()):comparisons.append(dict(step=n,mask1={k:pairs[850][k] for k in ['crashes','zone_escapes','zone_hover_successes','max_full_eval_zone_hold_s','mean_carrier_altitude_error_mm','sample_max_extent_m','sample_max_speed_m_s','sample_max_tilt_deg','last_saved_sample']},mask2={k:pairs[851][k] for k in ['crashes','zone_escapes','zone_hover_successes','max_full_eval_zone_hold_s','mean_carrier_altitude_error_mm','sample_max_extent_m','sample_max_speed_m_s','sample_max_tilt_deg','last_saved_sample']}))
result['paired_saved_steps']=comparisons
stem.with_suffix('.json').write_text(json.dumps(result,indent=2))
lines=['# Factorial70401 independent savedtrajectory review',f'Retrieved {read_time}. No GPU evaluation or production writes.','',result['scope'],'',f'Initial checkpoint weights exactly equal: {result["initial_policy_weights_exactly_equal"]}.','', '|Trial|Step|Kind|Escapes|Hover|Full-eval max hold(s)|Carrier altitude error(mm)|Sample max XY extent(m)|Last sample height(m)|Last sample speed(m/s)|Lastsample tilt(deg)|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in result['rows']:
 f=r['last_saved_sample'];lines.append(f'|{r["trial"]}|{r["training_step"]}|{r["label"]}|{r["zone_escapes"]}|{r["zone_hover_successes"]}|{r["max_full_eval_zone_hold_s"]:.3f}|{r["mean_carrier_altitude_error_mm"]:.3f}|{r["sample_max_extent_m"]:.3f}|{f["carrier"][2]:.3f}|{f["speed_m_s"]:.3f}|{f["tilt_deg"]:.3f}|')
lines += ['', 'Both masks evaluated on deterministic stillair trajectories;24episode counts do not mean24 independentdomains. Unique saved trajectories percheckpoint are listed inJSON. Full-evaluation counters determine sustainedhover; sparseframes alone cannot. Lastsaved periodicframe normally4.52s, not final5s endpoint. Altitudeerror is5s scalar and can differ sharply fromlastsampleheight. Escaped then reset endpoints are never evidence ofsuccess.','',f'851 stateatread: {json.dumps(result["trials"]["851"]["status_at_read"])}; finalavailable={result["trials"]["851"]["final_available"]}. If incomplete, do notattribute a finaloutcome.']
stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
