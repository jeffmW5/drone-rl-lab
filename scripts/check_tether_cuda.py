"""Frozen basic checks and synchronized throughput; not MuJoCo equivalence."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_backend import CudaTether
from tether_sim.training_lock import acquire_training_lease

lease=acquire_training_lease()
try:
    output=Path(sys.argv[1] if len(sys.argv)>1 else 'tmp/cuda-tether-v001/checks_v002.json')
    assert not output.exists(), 'Preserve existing results; choose a versioned output'
    e=CudaTether(8)
    a=(e.mass.sum(1,keepdim=True)*9.81/(4*e.params[:,2:3])).expand(-1,4).contiguous()
    start=e.position.clone()
    for _ in range(50): e.step(a)
    torch.cuda.synchronize()
    hover_drift=float((e.position-start).abs().max())
    stretch=float((torch.linalg.vector_norm(e.position[:,1:]-e.position[:,:-1],dim=2)-.6/13).abs().max())
    assert e.failed.sum()==0
    assert hover_drift<.02, hover_drift
    assert stretch<.001, stretch
    e.reset(); e.params[:,3]=0; e.motors.zero_(); a.zero_()
    initial=e.position.clone()
    e.step(a,substeps=100)
    fall=float((initial[:,:,2]-e.position[:,:,2]).mean())
    assert abs(fall-9.81*.001**2*100*101/2)<1e-5,fall
    e.reset(); e.boxes=e.position[:,13:].float().repeat(1,1,2).contiguous()
    e.boxes[:,:,3:]=.03
    e.step(a,substeps=1)
    assert bool((e.failed==1).all())
    e=CudaTether(1024)
    a=(e.mass.sum(1,keepdim=True)*9.81/(4*e.params[:,2:3])).expand(-1,4).contiguous()
    for _ in range(3): e.step(a)
    torch.cuda.synchronize(); started=time.perf_counter()
    for _ in range(20): e.step(a)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    result=dict(status='basic checks passed',gpu=torch.cuda.get_device_name(0),hover_1s_max_drift_m=hover_drift,
      max_link_stretch_m=stretch,freefall_100ms_m=fall,collision_detection=True,batch=1024,
      synchronized_control_transitions_per_second=1024*20/elapsed,seconds=elapsed,
      scope='FP64 particle-tether prototype; throughput includes stepping only, no reset, rendering, inference or PPO. No MuJoCo trajectory parity or hardware calibration established.')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
finally: lease.close()
