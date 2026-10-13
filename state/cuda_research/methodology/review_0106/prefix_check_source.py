from pathlib import Path
import torch,json
root=Path.cwd();a=root/'tmp/cuda-research/cuda_auto_0848_takeoff/policy.pt';b=root/'tmp/cuda-research/cuda_auto_0850_takeoff/checkpoint_65536.pt'
if not b.exists():print('Long65kprefix pending');raise SystemExit(0)
x=torch.load(a,map_location='cpu',weights_only=True);y=torch.load(b,map_location='cpu',weights_only=True)
rows={k:{'equal':torch.equal(x[k],y[k]),'max_abs_delta':float((x[k]-y[k]).abs().max())} for k in x};result={'short':str(a),'long_prefix':str(b),'all_tensors_equal':all(v['equal'] for v in rows.values()),'tensors':rows};(root/'tmp/deep_audit_root/prefix_70401_mask1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
