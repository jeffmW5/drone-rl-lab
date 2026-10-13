import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import CudaPolicy,configure_collective_response
torch.manual_seed(91);p=CudaPolicy()
with torch.no_grad():p.actor.weight.normal_(0,.0001)
old=copy.deepcopy(p)
x=torch.zeros(4,98);x[:,:42]=torch.zeros(14,3).flatten();x[:,84]=1;x[:,91:95]=(.0434+.001+.005)*9.81/4;x[:,97]=.2
x[:,:42].reshape(4,14,3)[:,:,2]=.6-torch.arange(14)*.6/13
x[1,2]+=.2;x[2,44]=.5;x[3,44]=-.5
before=old(x)[0].mean.detach();values=old(x)[1].detach();configure_collective_response(p,2);after=p(x)[0].mean.detach()
assert torch.allclose(after[0,0],before[0,0],atol=1e-6)
assert torch.allclose(after[:,0]-after[0,0],64*(before[:,0]-before[0,0]),atol=1e-6)
assert torch.equal(after[:,1:],before[:,1:]) and torch.equal(p(x)[1].detach(),values)
assert torch.equal(p.log_std,old.log_std)
print('Production collective initialization preserves anchor, differential means, critic and noise; scales only collective state-response64x')
