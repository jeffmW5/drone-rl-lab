"""CPU reward9 ablation: remove stationary-drone preference away from goal."""
import json
import torch
from tether_sim.cuda_task import approach_stop_reward
p=torch.zeros(1,14,3);p[:,0,2]=.8
v=torch.zeros_like(p);q=torch.zeros(1,7);q[:,0]=1
goal=torch.zeros(1,3);goal[:,2]=.2
failed=torch.zeros(1,dtype=torch.int32)
def score(version):return approach_stop_reward(p,v,q,goal,failed,version).item()
near=score(9);p[:,0,0]=1;far=score(9)
assert near>far
old_stopped=score(8)
v[:,0,0]=.5
assert score(9)==far # drone motion alone is no longer penalized
assert score(8)<old_stopped # old speed bonus decreases, new position bonus unchanged here
v.zero_();p[:,0,0]=0
assert abs(score(8)-score(9))<1e-6 # drone at target and stopped has identical bonus
q[:,4:]=1000
assert abs(score(9)-near)<1e-6
print(json.dumps(dict(status='pass',scope='Reward9 drone position bonus, no unconditional drone-speed preference; other reward terms unchanged')))
