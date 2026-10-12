import json
import torch
from tether_sim.cuda_task import navigation_priority_reward
p=torch.zeros(1,14,3);p[:,0,2]=.8;p[:,-1,2]=.2
v=torch.zeros_like(p);q=torch.zeros(1,7);q[:,0]=1
g=torch.tensor([[0.,0.,.2]]);f=torch.zeros(1,dtype=torch.int32)
def score(version):return navigation_priority_reward(p,v,q,g,f,version).item()
assert abs(score(10)-score(11))<1e-6
p[:,-1,0]=5;far=score(11);assert far>score(10)
p[:,-1,0]=4;assert score(11)>far
v[:,-1,0]=1;assert score(11)<navigation_priority_reward(p,torch.zeros_like(v),q,g,f,11).item()
p[:,-1,2]=.025;assert abs(score(11)-.05)<1e-6
print(json.dumps(dict(status='pass',scope='Reward11 same target maximum and ground gate; useful distance signal at 4-5m, stopping preference retained')))
