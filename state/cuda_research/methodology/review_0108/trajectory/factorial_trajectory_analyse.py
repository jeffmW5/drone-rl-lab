import argparse,datetime,hashlib,json,math
from pathlib import Path
import torch
ROOT=Path('/home/n33du/ai/drone-rl-lab')
OUT=ROOT/'tmp/deep_audit_physics'

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

parser=argparse.ArgumentParser(description='Read-only saved evaluation trajectory analysis; no GPU use')
parser.add_argument('--seed',type=int,required=True);parser.add_argument('--trials',type=int,nargs='+',required=True)
args=parser.parse_args();stem=OUT/f'factorial_{args.seed}_trajectory'
assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists(),'Refuse overwrite'
result=dict(retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Saved deterministic5s evaluations, full14node physics5 stillair zone1. Sparseframes cannotmeasurecontinuoushold or exactfailuretimes; full-eval counters preserved. Reusedseeds51011..13 nonindependent. No GPU, newevaluation, trainer or production changes.',trials={},rows=[],source_manifest={})
weights=[]
for trial in args.trials:
 folder=ROOT/f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff'
 config_path=folder/'config.json';status_path=folder/'status.json'
 config=json.loads(config_path.read_text());plan=config['research_plan'];status=json.loads(status_path.read_text())
 assert plan['seed']==args.seed
 state=dict(plan=plan,status_at_read=status,final_available=(folder/'result.json').exists(),selection_available=(folder/'selection.json').exists())
 for path in [config_path,status_path]:result['source_manifest'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 if state['final_available']:
  path=folder/'result.json';d=json.loads(path.read_text());state['result_steps']=d['steps'];state['training_seconds']=d['training_seconds']
  result['rows'].append(summarise(d['initial'],trial,'initial',0,path));result['rows'].append(summarise(d['final'],trial,'final',d['steps'],path))
 if state['selection_available']:
  path=folder/'selection.json';d=json.loads(path.read_text());state['selection_metadata']={k:v for k,v in d.items() if k!='best'}
  result['rows'].append(summarise(d['best'],trial,'selected',d['best_step'],path))
 for path in sorted(folder.glob('eval_*.json'),key=lambda p:int(p.stem[5:])):
  d=json.loads(path.read_text());result['rows'].append(summarise(d,trial,'periodic',int(path.stem[5:]),path))
 initial_path=folder/'initial.pt';weights.append((trial,torch.load(initial_path,map_location='cpu',weights_only=True)))
 state['initial_checkpoint_sha256']=hashlib.sha256(initial_path.read_bytes()).hexdigest()
 result['trials'][str(trial)]=state
result['initial_policy_weights_exactly_equal']=all(set(weights[0][1])==set(w) and all(torch.equal(weights[0][1][k],w[k]) for k in w) for _,w in weights[1:])
result['rows'].sort(key=lambda r:(r['trial'],r['training_step'],r['label']))
# Matchedmask comparisons at commoncompletedsteps. Context preservesduration/budgets.
comparisons=[]
for duration in sorted(set(x['plan']['steps'] for x in result['trials'].values())):
 arm={x['plan']['learner_valid_mask_version']:int(t) for t,x in result['trials'].items() if x['plan']['steps']==duration}
 if set(arm)!={1,2}:continue
 for step in sorted(set(r['training_step'] for r in result['rows'] if r['trial']==arm[1])):
  pair={m:next((r for r in result['rows'] if r['trial']==arm[m] and r['training_step']==step and (r['label']=='periodic' or (step==0 and r['label']=='initial'))),None) for m in (1,2)}
  if all(pair.values()):comparisons.append(dict(duration=duration,step=step,mask1={k:pair[1][k] for k in ['trial','crashes','takeoffs','zone_hover_successes','zone_escapes','max_full_eval_zone_hold_s','mean_carrier_altitude_error_mm','sample_max_extent_m','sample_max_speed_m_s','last_saved_sample']},mask2={k:pair[2][k] for k in ['trial','crashes','takeoffs','zone_hover_successes','zone_escapes','max_full_eval_zone_hold_s','mean_carrier_altitude_error_mm','sample_max_extent_m','sample_max_speed_m_s','last_saved_sample']}))
result['paired_saved_steps']=comparisons
# Exactshort/long65kprefix comparisons of checkpointtensor values and evaljson.
prefix=[]
for mask in (1,2):
 by_duration={x['plan']['steps']:int(t) for t,x in result['trials'].items() if x['plan']['learner_valid_mask_version']==mask}
 if set(by_duration)!={65536,262144}:continue
 short,long=by_duration[65536],by_duration[262144]
 a=ROOT/f'tmp/cuda-research/cuda_auto_{short:04d}_takeoff';b=ROOT/f'tmp/cuda-research/cuda_auto_{long:04d}_takeoff'
 wa=torch.load(a/'checkpoint_65536.pt',map_location='cpu',weights_only=True);wb=torch.load(b/'checkpoint_65536.pt',map_location='cpu',weights_only=True)
 da=json.loads((a/'eval_65536.json').read_text());db=json.loads((b/'eval_65536.json').read_text())
 prefix.append(dict(mask=mask,short_trial=short,long_trial=long,checkpoint_tensors_exactly_equal=set(wa)==set(wb) and all(torch.equal(wa[k],wb[k]) for k in wa),saved_eval_dicts_exactly_equal=da==db))
result['matched65k_prefix']=prefix
for row in result['rows']:
 result['source_manifest'][row['source']]=row['source_sha256']
 runs=row.pop('trajectory_runs')
 row['saved_frames_first_domain']=runs[0]['frames']
 row['validation_run_metadata']=[{k:v for k,v in r.items() if k!='frames'} for r in runs]
 if row['unique_saved_trajectory_domains']>1:row['additional_frames_nonidentical_domains']=[r for r in runs[1:] if r['frames']!=runs[0]['frames']]

stem.with_suffix('.json').write_text(json.dumps(result,indent=2))
lines=[f'# Factorial{args.seed} independent saved trajectory review',f'Retrieved {result["retrieved_utc"]}.',result['scope'],'',f'Initialweights exactlyequal: {result["initial_policy_weights_exactly_equal"]}. Matched65k prefixes: {json.dumps(prefix)}.','', '|Trial|Mask|Targetsteps|Checkpoint|Kind|Failures/escapes|Takeoffs|Hover|Peakhold(s)|Altitudeerr(mm)|Savedmaxextent(m)|Lastsampleheight(m)|Lastspeed(m/s)|Lasttilt(deg)|','|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in result['rows']:
 f=r['last_saved_sample'];plan=result['trials'][str(r['trial'])]['plan']
 lines.append(f'|{r["trial"]}|{plan["learner_valid_mask_version"]}|{plan["steps"]}|{r["training_step"]}|{r["label"]}|{r["crashes"]}/{r["zone_escapes"]}|{r["takeoffs"]}|{r["zone_hover_successes"]}|{r["max_full_eval_zone_hold_s"]:.3f}|{r["mean_carrier_altitude_error_mm"]:.3f}|{r["sample_max_extent_m"]:.3f}|{f["carrier"][2]:.3f}|{f["speed_m_s"]:.3f}|{f["tilt_deg"]:.3f}|')
lines+=['', 'Eachcheckpoint recordsone uniquesavedtrajectory acrossvalidationseeds. Counts24are repeateddeterministicdomain outcomes. Lastsavedframe usually4.52s, not5s finalendpoint. Sampleswithterminalfalse do notruleout unsampled earlierboundaryfailure andreset. Fullholdcounters takeprecedence oversparse stableframes. No promotion orcausal benefit established beforeallthree trainingseedscomplete.','']
stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
