from pathlib import Path
import torch,json
r=Path.cwd();p=r/'tmp/cuda-research/cuda_auto_0805_takeoff';roll=torch.load(p/'rollout_65536.pt',map_location='cpu',weights_only=True);o=roll['observations'];h=o[...,2]+.2;speed=o[...,42:45].norm(dim=-1);extent=o[...,:2].abs().max(dim=-1).values
masks={'stable_instant':(h-.8).abs().lt(.1)&speed.lt(.2)&extent.le(1),'height_vicinity':(h-.8).abs().lt(.1),'fast':speed.gt(1),'outside1m':extent.gt(1),'ground':h.lt(.15)};rows=[]
for f in sorted(p.glob('advantages_*.pt')):
 x=torch.load(f,map_location='cpu',weights_only=True);a=x['advantages']
 if a.shape!=h.shape:continue # Minibatch calls lack sampled indices; never align to full observations.
 row={'file':f.name,'finite':bool(torch.isfinite(a).all()),'groups':{}}
 for name,mask in masks.items():
  y=a[mask];row['groups'][name]={'n':y.numel(),'mean':float(y.mean()) if y.numel() else None,'positive_fraction':float((y>0).float().mean()) if y.numel() else None}
 rows.append(row)
print('calls',len(rows));print(rows[0]);print(rows[-1]);(r/'tmp/advantage805_summary.json').write_text(json.dumps(rows,indent=2))
