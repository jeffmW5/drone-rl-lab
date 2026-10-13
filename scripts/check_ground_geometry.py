import json,torch
from tether_sim.cuda_task import training_starts
p=torch.zeros(4,14,3,dtype=torch.float64);p[:,:,0]=torch.arange(14).double()*.6/13;p[:,:,2]=.025;p[:,0,2]=.05
assert torch.equal(training_starts(p,1,'takeoff'),p)
s=training_starts(p,3,'takeoff'); lengths=torch.linalg.vector_norm(s[:,1:]-s[:,:-1],dim=-1)
assert (lengths-.6/13).abs().max()<1e-12
assert torch.allclose(s[:,0,2],torch.full((4,),.05,dtype=torch.float64))
assert torch.allclose(s[:,-1,2],torch.full((4,),.025,dtype=torch.float64))
assert torch.equal(p[:,:,2],torch.cat([torch.full((4,1),.05,dtype=torch.float64),torch.full((4,13),.025,dtype=torch.float64)],dim=1))
mixed=training_starts(p,2,'takeoff');assert torch.equal(mixed[::2],p[::2])
print(json.dumps(dict(status='pass',scope='Version3 valid rest lengths, carrier/payload floor heights; legacy and mixed starts preserved, input not mutated')))
