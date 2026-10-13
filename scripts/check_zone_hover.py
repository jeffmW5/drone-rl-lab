"""CPU tests of production zone reward and true failure boundaries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import zone_hover_reward
p=torch.zeros(8,14,3,dtype=torch.float64);p[:,0,2]=.8
p[:,0,0]=torch.tensor([0,1,1.0001,1.9999,2,-2,0,0])
p[6,0,1]=2;p[7,0,:2]=.9
v=torch.zeros_like(p);q=torch.zeros(8,7);q[:,0]=1
reward,escaped,band=zone_hover_reward(p,v,q,torch.zeros(8,3),torch.zeros(8,dtype=torch.int32))
assert escaped.tolist()==[False,False,False,False,True,True,True,False]
assert band.tolist()==[False,False,True,True,False,False,False,False]
assert bool((reward[2:7]<0).all())
assert reward[0]>0 and reward[1]>0 and reward[7]>0
p[0,0,2]=.05
assert zone_hover_reward(p,v,q,torch.zeros(8,3),torch.zeros(8,dtype=torch.int32))[0][0]<0
print('Passed production zone signs, square corners, exact1m/2m boundaries and ground reward checks')

p[:,0,2]=.8
new=zone_hover_reward(p,v,q,torch.zeros(8,3),torch.zeros(8,dtype=torch.int32),version=2)[0]
old=zone_hover_reward(p,v,q,torch.zeros(8,3),torch.zeros(8,dtype=torch.int32),version=1)[0]
assert old[0]==old[1] and new[0]>new[1] and new[0]>new[7]
assert bool((new[2:7]==-1).all())
assert new[0]==old[0]
print('Reward15 grades horizontal offset inside zone; reward14/center and negative band/failure preserved')
