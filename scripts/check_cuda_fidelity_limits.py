"""Reproduce reduced-model limits; these are counterexamples, not parity tests."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tether_sim.cuda_backend import CudaTether
from tether_sim.training_lock import acquire_training_lease

output=Path(sys.argv[1]);assert not output.exists()
lease=acquire_training_lease()
try:
    s=CudaTether(1);a=torch.zeros(1,4,device='cuda')
    s.motors.zero_();s.params[:,3]=0
    # A tiny obstacle at a cable link midpoint; endpoints are outside its volume.
    centre=(s.position[0,6]+s.position[0,7])/2
    s.boxes=torch.cat([centre.float(),torch.full((3,),.001,device='cuda')]).reshape(1,1,6)
    s.step(a,dt=.000001,substeps=1)
    midpoint_failed=int(s.failed.item());assert midpoint_failed==0
    s.reset();s.motors.zero_();s.boxes[0,0,:3]=s.position[0,6].float()
    s.step(a,dt=.000001,substeps=1)
    node_failed=int(s.failed.item());assert node_failed==1
    s.reset();s.boxes=torch.empty(1,0,6,device='cuda');s.motors.zero_()
    # An angled hanging cable translates the centre attachment but cannot torque it.
    z=s.position[0,0,2].clone()
    s.position[0,:,0]=torch.arange(14,device='cuda')*.6/13*.6
    s.position[0,:,2]=z-torch.arange(14,device='cuda')*.6/13*.8
    start=s.position[0,0].clone()
    for _ in range(10):s.step(a)
    omega=s.orientation[0,4:].cpu().tolist()
    movement=float(torch.linalg.vector_norm(s.position[0,0]-start))
    assert movement>0 and max(abs(x) for x in omega)==0
    result=dict(segment_midpoint_intersects_box=True,midpoint_crash_flag=midpoint_failed,
                node_crash_flag=node_failed,angled_cable_centre_movement_m=movement,
                angled_cable_angular_velocity_rad_s=omega,
                scope='Thin-obstacle counterexample and zero-offset centre attachment; no hardware calibration')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:lease.close()
