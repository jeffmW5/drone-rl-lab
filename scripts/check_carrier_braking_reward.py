import torch
from tether_sim.cuda_task import navigation_priority_reward
p=torch.zeros(1,14,3);p[:,-1,2]=.2
v=torch.zeros_like(p);q=torch.zeros(1,7);q[:,0]=1
g=torch.tensor([[0.,0.,.2]]);f=torch.zeros(1,dtype=torch.int32)
def score(z,speed):
    p[:,0,2]=z;v[:,0,2]=speed
    return navigation_priority_reward(p,v,q,g,f,12).item()
assert score(1.4,-.35)>score(1.4,.35)
assert score(.4,.3)>score(.4,-.3)
assert score(.8,0)>score(.8,.3)
print('PASS: reward12 carrier approaches below goal, brakes above goal, stops at goal; scoring only')
