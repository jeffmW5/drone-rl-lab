"""CPU-only885 vs88165k prefix proof; exclusive immutable evidence outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,torch,zipfile,time
from pathlib import Path
ROOT=Path('/home/n33du/ai/drone-rl-lab');SHORT=ROOT/'tmp/cuda-research/cuda_auto_0881_takeoff';LONG=ROOT/'tmp/cuda-research/cuda_auto_0885_takeoff'
OUTS=[ROOT/'state/cuda_research/methodology/review_0110/tenfold_duration/prefix_885_881.json',ROOT/'tmp/deep_audit_learner/prefix_885_881.json']
if any(p.exists() for p in OUTS):raise FileExistsError('Refuse existing evidence')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def info(path):
 before=path.stat();data=path.read_bytes();after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Changed during read: '+str(path))
 return {'path':str(path),'bytes':len(data),'mtime_ns':after.st_mtime_ns,'sha256':hashlib.sha256(data).hexdigest(),'stable_during_read':True},data

def read_json(path):
 meta,data=info(path);return meta,json.loads(data)
def checkpoint(path):
 meta,data=info(path)
 with zipfile.ZipFile(path) as z:
  bad=z.testzip();members=len(z.namelist())
 state=torch.load(path,map_location='cpu',weights_only=True)
 after=path.stat()
 if (meta['bytes'],meta['mtime_ns'])!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Checkpoint changed during load')
 if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in state.values()):raise RuntimeError('Unexpected state dict')
 meta.update(zip_crc_complete=bad is None,zip_members=members,weights_only_load_complete=True,tensor_keys=len(state),total_elements=sum(v.numel() for v in state.values()))
 return meta,state
cm_a,cfg_a=read_json(SHORT/'config.json');cm_b,cfg_b=read_json(LONG/'config.json');pa,pb=cfg_a['research_plan'],cfg_b['research_plan']
meta_a,a=checkpoint(SHORT/'policy.pt');meta_b,b=checkpoint(LONG/'checkpoint_65536.pt');keys_equal=a.keys()==b.keys();rows=[]
if keys_equal:
 for key in a:
  x,y=a[key],b[key];shape=x.shape==y.shape;dtype=x.dtype==y.dtype
  rows.append({'key':key,'shape':list(x.shape),'dtype_a':str(x.dtype),'dtype_b':str(y.dtype),'shape_equal':shape,'dtype_equal':dtype,'equal':shape and dtype and torch.equal(x,y),'max_abs_delta':float((x.double()-y.double()).abs().max()) if shape and x.numel() else None,'finite_a':bool(torch.isfinite(x).all()),'finite_b':bool(torch.isfinite(y).all())})
reference_match=pb.get('short_prefix_reference')==str(SHORT/'policy.pt') and pb.get('short_prefix_reference_sha256')==meta_a['sha256']
core_ignore={'total_timesteps','data_dir','research_plan'}
plan_ignore={'name','steps','budget_seconds','hypothesis','limitations','methodology_trial','methodology_condition','methodology_review','rejection_rule','duration_version','duration_multiplier','short_reference_methodology_trial','short_prefix_reference','short_prefix_reference_sha256'}
core_differences={k:{'short':cfg_a.get(k),'long':cfg_b.get(k)} for k in cfg_a.keys()|cfg_b.keys() if k not in core_ignore and cfg_a.get(k)!=cfg_b.get(k)}
plan_differences={k:{'short':pa.get(k),'long':pb.get(k)} for k in pa.keys()|pb.keys() if k not in plan_ignore and pa.get(k)!=pb.get(k)}
source_groups=[]
for suffix,files in [('.sources',None),('',None)]:
 da,db=Path(str(SHORT)+suffix),Path(str(LONG)+suffix)
 if suffix:
  hm_a,ha=read_json(da/'hashes.json');hm_b,hb=read_json(db/'hashes.json')
  records=[]
  for relative in sorted(ha.keys()|hb.keys()):
   path_a,path_b=da/Path(relative).name,db/Path(relative).name;ma,_=info(path_a);mb,_=info(path_b)
   records.append({'relative_source':relative,'short':ma,'long':mb,'actual_hash_equal':ma['sha256']==mb['sha256'],'short_declared_hash_verified':ma['sha256']==ha.get(relative),'long_declared_hash_verified':mb['sha256']==hb.get(relative)})
  source_groups.append({'kind':'daemon_core_source_snapshot','manifest_short':hm_a,'manifest_long':hm_b,'declared_manifests_equal':ha==hb,'sources':records})
 else:
  names=sorted({p.name for p in da.iterdir() if p.suffix in('.py','.cu','.c')}|{p.name for p in db.iterdir() if p.suffix in('.py','.cu','.c')})
  records=[]
  for name in names:
   if not (da/name).is_file() or not (db/name).is_file():
    records.append({'source_name':name,'both_present':False,'actual_hash_equal':False});continue
   ma,_=info(da/name);mb,_=info(db/name);records.append({'source_name':name,'both_present':True,'short':ma,'long':mb,'actual_hash_equal':ma['sha256']==mb['sha256']})
  source_groups.append({'kind':'run_auxiliary_source_snapshots','sources':records})
all_sources_match=all(r['actual_hash_equal'] and r.get('short_declared_hash_verified',True) and r.get('long_declared_hash_verified',True) for g in source_groups for r in g['sources'])
puffer_path=Path('/home/n33du/ai/puffer-tether-venv/lib/python3.12/site-packages/pufferlib/pufferl.py');puffer_meta,_=info(puffer_path);puffer_matches=puffer_meta['sha256']==pa.get('pufferl_source_sha256')==pb.get('pufferl_source_sha256')
result_meta,short_result=read_json(SHORT/'result.json');eval_meta,long_eval=read_json(LONG/'eval_65536.json');short_final=short_result['final']
def without_trace(x):
 if isinstance(x,dict):return {k:without_trace(v) for k,v in x.items() if k!='evaluation_terminal_trace'}
 if isinstance(x,list):return [without_trace(v) for v in x]
 return x
all_eval_equal=short_final==long_eval;old_eval_equal=without_trace(short_final)==without_trace(long_eval)
pass_checks={'reference_path_sha_matches_long_plan':reference_match,'matched_seed70601_reward15_mask2':cfg_a['seed']==cfg_b['seed']==70601 and pa['reward_version']==pb['reward_version']==15 and pa['learner_valid_mask_version']==pb['learner_valid_mask_version']==2,'short65536_long655360':cfg_a['total_timesteps']==65536 and cfg_b['total_timesteps']==655360,'tenfold_budget1800':pb['budget_seconds']==1800 and pb.get('duration_multiplier')==10,'behavioral_configs_identical_except_duration_metadata':not core_differences and not plan_differences,'all_snapshot_sources_actual_equal_and_declared_verified':all_sources_match,'installed_puffer_source_matches_guard':puffer_matches,'all10_tensor_keys29931_elements':keys_equal and len(a)==10 and sum(x.numel() for x in a.values())==29931,'all_tensors_finite_exact_equal':keys_equal and all(r['equal'] and r['finite_a'] and r['finite_b'] for r in rows),'checkpoint_crc_complete':meta_a['zip_crc_complete'] and meta_b['zip_crc_complete'],'all_evaluation_fields_and_traces_equal':all_eval_equal,'all_legacy_evaluation_fields_equal':old_eval_equal}
result={'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'passed' if all(pass_checks.values()) else 'mismatch','scope':'CPU-only existing immutable short881 final vs active tenfold885prefix at65536. CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS1, CPU weights_only loads; no GPU/build/training/config/source/queue edits. Exact65kprefix proves matched start/early optimization only; future655360 outcome remains pending. No sustained-hover/hardware result inferred.','checks':pass_checks,'configuration_sources':[cm_a,cm_b],'short_reference_sha_in_plan':pb.get('short_prefix_reference_sha256'),'checkpoint_sources':[meta_a,meta_b],'keys_equal':keys_equal,'all_tensor_equal':keys_equal and all(r['equal'] for r in rows),'max_abs_delta':max(r['max_abs_delta'] for r in rows if r['max_abs_delta'] is not None) if rows else None,'tensor_comparison':rows,'core_config_differences_excluding_duration':core_differences,'plan_differences_excluding_duration_metadata':plan_differences,'source_comparison':source_groups,'puffer_source':puffer_meta,'evaluation_sources':[result_meta,eval_meta],'evaluation_short_final_field':'result.json.final','evaluation_full_equal':all_eval_equal,'evaluation_legacy_equal':old_eval_equal,'evaluation_prefix_outcome':{k:long_eval.get(k) for k in ('crashes','takeoffs','zone_hover_successes','zone_hover_error_mm','zone_escapes','hover_successes','mean_final_error_mm')},'evaluation_max_zone_hold_s':max(h for run in long_eval['runs'] for h in run['zone_hover_hold_s'])}
for path in OUTS:
 with path.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'status':result['status'],'checks':pass_checks,'all_tensor_equal':result['all_tensor_equal'],'max_abs_delta':result['max_abs_delta'],'evaluation_full_equal':all_eval_equal,'source_files_compared':sum(len(g['sources']) for g in source_groups),'outputs':[str(p) for p in OUTS]},indent=2))
