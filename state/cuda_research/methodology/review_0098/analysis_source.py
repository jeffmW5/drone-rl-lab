from pathlib import Path
import torch,json
r=Path.cwd();p=r/'tmp/cuda-research/cuda_auto_0791_takeoff';rows=[]
for f in sorted(p.glob('rollout_*.pt')):
 x=torch.load(f,map_location='cpu',weights_only=True);o=x['observations'].reshape(-1,98);height=o[:,2]+.2;speed=o[:,42:45].norm(dim=-1);extent=o[:,:2].abs().max(dim=-1).values
 row={'file':str(f),'shapes':{k:list(v.shape) for k,v in x.items()},'height_range':[float(height.min()),float(height.max())],'altitude_vicinity_fraction':float(((height-.8).abs()<.1).float().mean()),'stable_state_fraction':float(((height-.8).abs().lt(.1)&speed.lt(.2)&extent.le(1)).float().mean()),'ground_fraction':float((height<.15).float().mean()),'speed_quantiles':speed.quantile(torch.tensor([0.,.5,.9,1.])).tolist(),'terminal_count':int(x['terminals'].sum()),'truncation_count':int(x['truncations'].sum())};rows.append(row);print(row)
(r/'tmp/rollout791_summary.json').write_text(json.dumps(rows,indent=2))
