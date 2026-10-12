import json
import torch
from tether_sim.cuda_task import navigation_priority_reward
p=torch.zeros(1,14,3);p[:,0,2]=.8
v=torch.zeros_like(p);q=torch.zeros(1,7);q[:,0]=1
goal=torch.tensor([[0.,0.,.2]]);f=torch.zeros(1,dtype=torch.int32)
def score():return navigation_priority_reward(p,v,q,goal,f).item()
p[:,-1,2]=.025;assert abs(score()-.05)<1e-6
p[:,-1,2]=.2;held=score();assert abs(held-.9)<1e-6
v[:,-1,0]=1;assert score()<held
v.zero_();p[:,-1,0]=1;assert score()<held
p[:,-1,0]=0;p[:,-1,2]=.025;ground=score()
p[:,-1,2]=.0625;assert ground<score()<held
q[:,4:]=1000;p[:,-1,2]=.2;assert abs(score()-held)<1e-6
f[:]=1;assert score()<-1
print(json.dumps(dict(status='pass',scope='CPU reward10: pre-lift suppression, smooth release, target hold dominates drift, no angular penalty, crash sign')))
