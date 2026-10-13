import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import CudaPolicy
p=CudaPolicy();p.load_state_dict(torch.load('tmp/cuda-research/cuda_auto_0713_takeoff/best.pt',map_location='cpu',weights_only=True))
torch.manual_seed(92);x=torch.randn(8,98)*.1
base=copy.deepcopy(p);p.configure_collective_observation(2)
shift=x.clone();shift[:,:42].reshape(-1,14,3)[:,:,:2]+=.5;shift[:,42:84].reshape(-1,14,3)[:,:,:2]-=.5
before=base(x);out=p(x);changed=p(shift)
assert torch.equal(out[0].mean[:,1:],before[0].mean[:,1:])
assert torch.equal(out[1],before[1]) and torch.equal(out[0].scale,before[0].scale)
assert torch.allclose(out[0].mean[:,0],changed[0].mean[:,0],atol=1e-6)
assert not torch.allclose(out[0].mean[:,1:],changed[0].mean[:,1:])
loss=out[0].mean.sum()+out[1].sum();loss.backward()
assert all(torch.isfinite(t.grad).all() for t in p.parameters() if t.grad is not None)
base.configure_collective_observation(1);assert torch.equal(base(x)[0].mean,before[0].mean)
print('Actual713 weights: collective uniform XY/velocity invariance; full-state other means/value/noise unchanged; finite gradients; version1 unchanged')

assert torch.allclose(p(x.reshape(2,4,98))[0].mean.reshape(8,4),p(x)[0].mean,atol=1e-6)
print('Batched temporal shapes verified')
