import hashlib,json,math,datetime
from pathlib import Path
ROOT=Path('/home/n33du/ai/drone-rl-lab');OUT=ROOT/'tmp/deep_audit_physics'
stem=OUT/'trace_gpu_analysis_v001';assert not stem.with_suffix('.json').exists(),'Refuse overwrite'
result=dict(retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read-only savedGPU terminaltrace artifacts860/861; initialfixedsource replay reproduces850/851finalpolicy. New pre-reset statedecodedfromFP32obs, exactabsolute/episode steps; no newGPUexecution, production orbenchmark change by auditor.',trials=[])
for trial in [860,861]:
 f=ROOT/f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff'
 if not f.exists():result['trials'].append(dict(trial=trial,available=False));continue
 comparison=json.loads((f/'eval_terminal_trace_comparison.json').read_text())
 item=dict(trial=trial,status=json.loads((f/'status.json').read_text()),comparison=comparison,result_available=(f/'result.json').exists(),stages=[])
 if item['result_available']:
  source=f/'result.json';item['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();d=json.loads(source.read_text())
  for stage in ['initial','final']:
   data=d[stage];states=[]
   for run in data['runs']:
    trace=run['evaluation_terminal_trace'];events=[]
    for e in trace['terminals']:
     p=e['positions'][0];v=e['velocities'][0];q=e['orientation'];w,x,y,z=q[:4];m=e['motors'];torque=[.032*(m[0]+m[1]-m[2]-m[3]),.032*(-m[0]+m[1]+m[2]-m[3]),.006*(-m[0]+m[1]-m[2]+m[3])]
     events.append(dict(lane=e['lane'],absolute_step=e['absolute_step'],absolute_time_s=e['absolute_time_s'],episode_step=e['episode_step'],episode_time_s=e['episode_time_s'],failed=e['failed'],truncated=e['truncated'],reason=e['reason'],reward=e['reward'],carrier=p,velocity=v,speed_m_s=math.sqrt(sum(a*a for a in v)),horizontal_speed_m_s=math.hypot(v[0],v[1]),extent_m=max(abs(p[0]),abs(p[1])),tilt_deg=math.degrees(math.acos(max(-1,min(1,1-2*(x*x+y*y))))),roll_deg=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),pitch_deg=math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x))))),rates_rad_s=q[4:],motors_N=m,torques_Nm=torque,finite=not e['has_nonfinite']))
    states.append(dict(seed=run['seed'],trace_terminal_count=trace['terminal_count'],events=events,first_lane0=next((e for e in events if e['lane']==0),None)))
   item['stages'].append(dict(stage=stage,crashes=data['crashes'],zone_escapes=data['zone_escapes'],zone_hover_successes=data['zone_hover_successes'],terminal_count=sum(s['trace_terminal_count'] for s in states),max_peak_zone_hold_s=max(h for r in data['runs'] for h in r['zone_hover_hold_s']),validation_runs=states))
 result['trials'].append(item)
stem.with_suffix('.json').write_text(json.dumps(result,indent=2))
lines=['# GPU saved terminaltrace independent analysis',result['scope'],f'Retrieved {result["retrieved_utc"]}.','', '|Trial|Stage|Sourcelegacyexact|Terminals|Failedcode|Firsttime(s)|Extent(m)|Height(m)|Horizspeed(m/s)|TotalSpeed(m/s)|Tilt(deg)|Pitch(deg)|PitchRate(rad/s)|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for item in result['trials']:
 if not item.get('available',True):continue
 for stage in item['stages']:
  e=stage['validation_runs'][0]['first_lane0']
  if e is None:continue
  lines.append(f'|{item["trial"]}|{stage["stage"]}|{item["comparison"]["legacy_fields_exactly_equal"] if stage["stage"]=="initial" else "notclaimed"}|{stage["terminal_count"]}|{e["failed"]}|{e["absolute_time_s"]:.3f}|{e["extent_m"]:.6f}|{e["carrier"][2]:.6f}|{e["horizontal_speed_m_s"]:.6f}|{e["speed_m_s"]:.6f}|{e["tilt_deg"]:.6f}|{e["pitch_deg"]:.6f}|{e["rates_rad_s"][1]:.6f}|')
 if not item['result_available']:lines.append(f'\nTrial{item["trial"]} fullresult notavailableatread; comparisonmanifest present with legacyexact={item["comparison"]["legacy_fields_exactly_equal"]},24expected/observedfailures, terminalevents unavailablehere. Noeventstate/timeinference frommanifestalone.')
lines+=['','Flag3 meansX/Yzoneboundary failure, not numericfailure orfloorcontact. Initial stages are fixed850/851finalpolicy reproductions; final stages follow4096additionaltrainingtransitions and cannotbeclaimed identicaltolegacyreference. Traceexactlegacycomparison preservesoldmetricsandtrajectory, showing instrumentation doesnot improveflight. Repeatedlanes/seedsareidenticalstillairdomains. Timebeyondfirstterminal thenauto-reset isnotrecovery.', '','The initialsource850terminalisnearhoverheight butmovesoutward acrosspositiveX at1.64m/s withpositivepitchabout10degreesandincreasingpitchrate. Terminal motorpitchtorqueisalsopositive, so thisobservedstate isnotbrakingX. Thisfactsupportsfailed attitude/velocity recovery butdoesnotproveauniversalcommandsignerror: neuralresponseandconstraintmodesdependonstate. No fulltrajectorycredit orcausaloptimizerdefect inferredfromoneendstate.']
stem.with_suffix('.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
