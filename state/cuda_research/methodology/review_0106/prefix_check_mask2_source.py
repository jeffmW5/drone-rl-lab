"""CPU-only exact checkpoint prefix comparison; create evidence exclusively."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,torch,zipfile,time
from pathlib import Path
root=Path('/home/n33du/ai/drone-rl-lab');out=root/'tmp/deep_audit_root/prefix_70401_mask2.json'
if out.exists():raise FileExistsError(out)
paths=[root/'tmp/cuda-research/cuda_auto_0849_takeoff/policy.pt',root/'tmp/cuda-research/cuda_auto_0851_takeoff/checkpoint_65536.pt']
torch.set_num_threads(1)
files=[];states=[]
for path in paths:
 before=path.stat();payload=path.read_bytes()
 with zipfile.ZipFile(path) as stream:
  bad=stream.testzip();members=stream.namelist();assert bad is None
 state=torch.load(path,map_location='cpu',weights_only=True)
 assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
 after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
 assert all(v.device.type=='cpu' for v in state.values())
 files.append({'path':str(path),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':hashlib.sha256(payload).hexdigest(),'zip_crc_complete':bad is None,'zip_members':len(members),'weights_only_load_complete':True,'stable_during_read':True,'tensor_keys':len(state),'total_elements':sum(v.numel() for v in state.values())});states.append(state)
a,b=states;keys_equal=a.keys()==b.keys();rows=[]
if keys_equal:
 for key in a:
  x,y=a[key],b[key];same_shape=x.shape==y.shape;same_dtype=x.dtype==y.dtype
  delta=(x.to(torch.float64)-y.to(torch.float64)).abs() if same_shape else None
  rows.append({'key':key,'shape':list(x.shape),'dtype_a':str(x.dtype),'dtype_b':str(y.dtype),'shape_equal':same_shape,'dtype_equal':same_dtype,'equal':same_shape and same_dtype and torch.equal(x,y),'max_abs_delta':float(delta.max()) if delta is not None and delta.numel() else None,'finite_a':bool(torch.isfinite(x).all()),'finite_b':bool(torch.isfinite(y).all())})
result={'scope':'CPU-only existing immutable policy/checkpoint reads, CUDA_VISIBLE_DEVICES empty and map_location cpu. Compare shortmask2 seed70401 final849 against longmask2 seed70401 prefix851 at65536steps. File serialization names may differ despite tensor equality. No GPU, build, training or production edits.','created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':files,'keys_equal':keys_equal,'all_tensor_equal':keys_equal and all(r['equal'] for r in rows),'max_abs_delta':max(r['max_abs_delta'] for r in rows if r['max_abs_delta'] is not None) if rows else None,'all_finite':all(r['finite_a'] and r['finite_b'] for r in rows),'tensor_comparison':rows}
with out.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
