"""CPU-only explicit matched prefix audit for seed70402; exclusive outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,torch,zipfile,time
from pathlib import Path
ROOT=Path('/home/n33du/ai/drone-rl-lab');OUT=ROOT/'tmp/deep_audit_root/prefix_70402.json'
PAIRS=((1,852,854),(2,853,855))
if OUT.exists():raise FileExistsError(OUT)
torch.set_num_threads(1)
def read_checkpoint(path):
 before=path.stat();payload=path.read_bytes()
 with zipfile.ZipFile(path) as stream:
  bad=stream.testzip();members=stream.namelist();assert bad is None
 state=torch.load(path,map_location='cpu',weights_only=True)
 assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in state.values())
 after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
 return state,{'path':str(path),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':hashlib.sha256(payload).hexdigest(),'zip_crc_complete':True,'zip_members':len(members),'weights_only_load_complete':True,'stable_during_read':True,'tensor_keys':len(state),'total_elements':sum(v.numel() for v in state.values())}
def compare_pair(mask,short_index,long_index):
 short_dir=ROOT/f'tmp/cuda-research/cuda_auto_{short_index:04d}_takeoff';long_dir=ROOT/f'tmp/cuda-research/cuda_auto_{long_index:04d}_takeoff'
 # Verify explicitly matched seed/mask from the immutable recorded configs.
 configs=[json.loads((p/'config.json').read_text()) for p in (short_dir,long_dir)]
 assert all(c['seed']==70402 for c in configs)
 plans=[c['research_plan'] for c in configs]
 assert all(p.get('learner_mask_version',p.get('learner_valid_mask_version',1))==mask for p in plans),[(p.get('learner_mask_version'),p.get('learner_valid_mask_version')) for p in plans]
 a,fa=read_checkpoint(short_dir/'policy.pt');b,fb=read_checkpoint(long_dir/'checkpoint_65536.pt')
 keys_equal=a.keys()==b.keys();rows=[]
 if keys_equal:
  for key in a:
   x,y=a[key],b[key];same_shape=x.shape==y.shape;same_dtype=x.dtype==y.dtype
   delta=(x.to(torch.float64)-y.to(torch.float64)).abs() if same_shape else None
   rows.append({'key':key,'shape':list(x.shape),'dtype_a':str(x.dtype),'dtype_b':str(y.dtype),'shape_equal':same_shape,'dtype_equal':same_dtype,'equal':same_shape and same_dtype and torch.equal(x,y),'max_abs_delta':float(delta.max()) if delta is not None and delta.numel() else None,'finite_a':bool(torch.isfinite(x).all()),'finite_b':bool(torch.isfinite(y).all())})
 return {'seed':70402,'mask_version':mask,'short_index':short_index,'long_index':long_index,'short_total_timesteps':configs[0]['total_timesteps'],'long_total_timesteps':configs[1]['total_timesteps'],'files':[fa,fb],'keys_equal':keys_equal,'all_tensor_equal':keys_equal and all(r['equal'] for r in rows),'max_abs_delta':max(r['max_abs_delta'] for r in rows if r['max_abs_delta'] is not None) if rows else None,'all_finite':all(r['finite_a'] and r['finite_b'] for r in rows),'tensor_comparison':rows}
result={'scope':'CPU-only immutable checkpoint reads with CUDA_VISIBLE_DEVICES empty, CPU map_location. Explicit matchedseed70402short852/853 final versuslong854/855prefix65536, masks1/2 respectively. No GPU/build/training/production or immutable review mutations. Serialization bytes can differ while tensors match exactly.','created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'comparisons':[compare_pair(*pair) for pair in PAIRS]}
with OUT.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({**{k:v for k,v in result.items() if k!='comparisons'},'comparisons':[{k:v for k,v in p.items() if k!='tensor_comparison'} for p in result['comparisons']]},indent=2))
