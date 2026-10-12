import json
import torch
from tether_sim.cuda_task import approach_stop_reward
p=torch.zeros(1,14,3);p[:,0,2]=.8;p[:,-1,0]=1
v=torch.zeros_like(p);q=torch.zeros(1,7);q[:,0]=1;goal=torch.zeros(1,3);fail=torch.zeros(1,dtype=torch.int)
def score(x):v[:,-1,0]=x;return float(approach_stop_reward(p,v,q,goal,fail))
assert score(-.4)>score(.4) # moving toward the target beats moving away
assert score(-.4)>score(0) # at one metre, correct approach beats standing still
p[:,-1]=goal
assert score(0)>score(.4) # at the target, stopped beats fly-through
base=score(0);q[:,4:]=1000;assert score(0)==base # no angular-rate penalty
p[:,0,2]=.02;assert score(0)<base # ground reward cannot count as flight
fail[:]=1;assert score(0)<-1
# Noncrash normalized-quaternion states remain within learner clipping range.
fail[:]=0
for distance in [0,.1,1,10]:
 p[:,-1,0]=distance
 for speed in [-20,0,20]:
  r=score(speed);assert -1<=r<=1
print(json.dumps(dict(status='pass',scope='CPU reward semantics, approach vs retreat, brake at goal, no angular cost, ground/crash and noncrash clipping bounds'),indent=2))
