import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import CudaPolicy
from tether_sim.policy_update_capture import query
p=CudaPolicy();p.load_state_dict(torch.load('tmp/cuda-research/cuda_auto_0759_takeoff/best.pt',map_location='cpu',weights_only=True));p.configure_collective_observation(2)
x=torch.randn(2,32,98);a=torch.randn(2,32,4);rng=torch.get_rng_state().clone();weights={k:v.clone() for k,v in p.state_dict().items()}
y=query(p,x,a);assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(v,weights[k]) for k,v in p.state_dict().items())
assert y['mean_action'].shape==a.shape and y['logprob'].shape==a.shape[:-1]
assert all(torch.isfinite(v).all() for v in y.values()) and all(t.grad is None for t in p.parameters())
print('Actual759 version2 queries preserve weights/RNG/no-grad; matched shapes and finite output')
