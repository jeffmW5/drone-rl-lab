import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import training_starts,initial_motor_forces
s=torch.zeros(4,14,3,dtype=torch.float64);s[:,:,0]=torch.arange(14)*.6/13;s[:,:,2]=.025;s[:,0,2]=.05
for version in (4,5):
 p=training_starts(s,version,'takeoff')
 assert torch.allclose(torch.linalg.vector_norm(p[:,1:]-p[:,:-1],dim=-1),torch.full((4,13),.6/13,dtype=torch.float64),atol=1e-12)
 assert torch.allclose(p[1::2,0,2],torch.tensor(.8,dtype=torch.float64))
 assert torch.equal(p[::2],training_starts(s,3,'takeoff')[::2])
m=torch.tensor([.0434]+[.001/12]*12+[.005]).repeat(4,1)
a=initial_motor_forces(m,4);b=initial_motor_forces(m,5)
assert not bool(a.any()) and not bool(b[::2].any())
assert torch.allclose(b[1::2].sum(1),m[1::2].sum(1)*9.81)
assert torch.equal(training_starts(s,4,'takeoff'),training_starts(s,5,'takeoff'))
print('Matched mixed geometry, exact links, ground motors-off and airborne total-weight motor initialization passed')
