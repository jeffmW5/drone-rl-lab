"""Check finite-radius segment detection; no collision response or hardware claim."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_backend import CudaTether
from tether_sim.training_lock import acquire_training_lease

output=Path(sys.argv[1]);assert not output.exists()
lease=acquire_training_lease()
try:
    libraries={1:'tmp/cuda-tether-v001/libtether.so',2:'tmp/cuda-tether-v002/libtether.so'}
    results={}
    cases={'midpoint':(0,0,False),'near_grazing':(.00249,0,False),
           'miss':(.003,0,False),'corner_miss':(.0024,.0024,False),'endpoint':(0,0,True)}
    for version,library in libraries.items():
        sim=CudaTether(1,library=library);action=torch.zeros(1,4,device='cuda');flags={}
        for name,(x,y,endpoint) in cases.items():
            sim.reset();sim.motors.zero_();centre=sim.position[0,6].clone() if endpoint else (sim.position[0,6]+sim.position[0,7])/2
            centre[0]+=x;centre[1]+=y
            sim.boxes=torch.cat([centre.float(),torch.full((3,),.001,device='cuda')]).reshape(1,1,6)
            sim.step(action,dt=.000001,substeps=1);flags[name]=int(sim.failed.item())
        expected=dict(midpoint=version==2,near_grazing=version==2,miss=False,corner_miss=False,endpoint=True)
        assert flags=={k:int(v) for k,v in expected.items()},flags
        results[str(version)]=flags
    positions=[]
    for version,library in libraries.items():
        sim=CudaTether(256,library=library)
        sim.boxes=torch.tensor([2,2,.35,.1,.1,.35],device='cuda').repeat(256,6,1)
        action=(sim.mass.sum(1,keepdim=True)*9.81/(4*sim.params[:,2:3])).expand(-1,4).contiguous()
        torch.cuda.synchronize();start=time.monotonic()
        for _ in range(10):sim.step(action)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        assert not bool(sim.failed.any());positions.append(sim.position.clone())
        results[str(version)]['physics_transitions_per_second']=2560/elapsed
    difference=float((positions[0]-positions[1]).abs().max());assert difference<1e-8
    results['collision_free_position_difference_m']=difference
    results['scope']='Axis-aligned boxes; detection only. No contact response, offset attachment, hardware calibration or original1mm claim.'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(results,indent=2));print(json.dumps(results))
finally:lease.close()
