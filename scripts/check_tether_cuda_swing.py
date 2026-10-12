"""Fixed-support small-angle cable swing and timestep counterevidence."""
import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_backend import CudaTether

rows=[]
output=Path(sys.argv[1] if len(sys.argv)>1 else 'tmp/cuda-tether-v001/swing_checks_v002.json')
assert not output.exists(), 'Preserve existing diagnostic; choose a versioned output'
for dt,substeps in [(.001,20),(.0005,40)]:
    e=CudaTether(1);e.mass[:,0]=1e6;e.params[:,2]=1e7;e.params[:,3]=0
    distance=torch.arange(14,device='cuda',dtype=torch.float64)*.6/13
    e.position[0,:,0]=distance*math.sin(.05);e.position[0,:,2]=3-distance*math.cos(.05)
    a=(e.mass.sum(1,keepdim=True)*9.81/(4*e.params[:,2:3])).expand(-1,4).contiguous()
    e.motors[:]=e.mass.sum(1,keepdim=True)*9.81/4
    trajectory=[];max_stretch=0
    for i in range(200):
        e.step(a,dt=dt,substeps=substeps,iterations=128)
        trajectory.append(float(e.position[0,-1,0]-e.position[0,0,0]))
        max_stretch=max(max_stretch,float((torch.linalg.vector_norm(e.position[:,1:]-e.position[:,:-1],dim=2)-.6/13).abs().max()))
    crossings=[]
    for i in range(1,len(trajectory)):
        if trajectory[i-1]>0>=trajectory[i]:
            crossings.append((i-1+trajectory[i-1]/(trajectory[i-1]-trajectory[i]))*.02)
    period=crossings[1]-crossings[0] if len(crossings)>1 else None
    rows.append(dict(dt=dt,period_s=period,max_link_stretch_m=max_stretch,anchor_drift_m=float((e.position[0,0]-torch.tensor([0,0,3],device='cuda')).abs().max()),crashes=int(e.failed.sum()),trajectory_x_m=trajectory))
analytic=2*math.pi*math.sqrt(.6/9.81)
result=dict(scope='Near-fixed support (drone1e6kg), cable1g + payload5g, .05rad initial tilt. Analytical reference is massless string; distributed cable differs. Concurrent live playback, not throughput benchmark.',massless_reference_period_s=analytic,runs=rows,
    timestep_period_difference_s=abs(rows[0]['period_s']-rows[1]['period_s']) if all(r['period_s'] for r in rows) else None,
    limitation='No angular coupling from cable attachment or segment-contact response. No MuJoCo equivalence established.')
output.write_text(json.dumps(result,indent=2))
print(json.dumps({**result,'runs':[{k:v for k,v in r.items() if k!='trajectory_x_m'} for r in rows]},indent=2))
