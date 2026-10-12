import json
import torch
from tether_sim.cuda_task import LandingBeforeSuccess
p=torch.zeros(2,14,3);v=torch.zeros_like(p);g=torch.zeros(2,3);g[:,2]=.2
f=torch.zeros(2,dtype=torch.int32);gate=LandingBeforeSuccess(2,'cpu')
def step():return gate.update(p,v,g,f)
p[:,0,2]=.05
assert not step().any() # initial ground is exempt
p[0,0,2]=.4
assert not step().any()
p[0,0,2]=.05
assert step().tolist()==[True,False]
assert step().tolist()==[True,False] # remaining grounded stays penalized
gate.reset(torch.tensor([True,False]));assert not step().any()
p[0,0,2]=.8;p[0,-1,2]=.2
for _ in range(49):step()
assert not gate.success[0]
v[0,-1,0]=.3;step();assert gate.hold[0]==0
v.zero_()
for _ in range(50):step()
assert gate.success[0]
p[0,0,2]=.05
assert not step().any() # successful landing exempt from new term
gate.reset();assert not gate.success.any() and not gate.airborne.any()
print(json.dumps(dict(status='pass',scope='CPU per-agent takeoff/landing latch, interrupted hold, 1s airborne prototype success, reset and initial/post-success exemptions')))
