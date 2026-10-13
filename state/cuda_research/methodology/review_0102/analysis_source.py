from pathlib import Path
import torch,json
p=Path('tmp/cuda-research/cuda_auto_0819_takeoff');roll=torch.load(p/'rollout_65536.pt',map_location='cpu',weights_only=True);rows=[]
for f in sorted(p.glob('minibatch_indices_*.pt')):
 x=torch.load(f,map_location='cpu',weights_only=True);idx=x['indices'];call=x['preceding_advantage_call'];a=torch.load(p/f'advantages_65536_{call:02d}.pt',map_location='cpu',weights_only=True)['advantages'][idx];norm=(a-a.mean())/(a.std()+1e-8)
 o=roll['observations'][idx];h=o[...,2]+.2;speed=o[...,42:45].norm(dim=-1);extent=o[...,:2].abs().max(dim=-1).values
 masks={'stable_instant':(h-.8).abs().lt(.1)&speed.lt(.2)&extent.le(1),'fast':speed.gt(1),'outside1m':extent.gt(1),'ground':h.lt(.15)}
 row={'file':f.name,'preceding_full_call':call,'groups':{}}
 for key,mask in masks.items():
  v=norm[mask];row['groups'][key]={'n':v.numel(),'mean':float(v.mean()) if v.numel() else None,'positive_fraction':float((v>0).float().mean()) if v.numel() else None}
 rows.append(row)
print('minibatches',len(rows));print(rows[0]);print(rows[-1]);Path('tmp/normalized819_summary.json').write_text(json.dumps(rows,indent=2))
