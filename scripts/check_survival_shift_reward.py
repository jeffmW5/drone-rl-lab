import json,torch
from tether_sim.cuda_task import navigation_priority_reward
p=torch.zeros(5,14,3);v=torch.zeros_like(p);q=torch.zeros(5,4);q[:,0]=1;g=torch.zeros(5,3);g[:,2]=.2;f=torch.zeros(5,dtype=torch.int32)
p[:,0,2]=.8;p[:,-1,2]=.2
p[1,-1,0]=1;v[2,0,0]=1;p[3,0,2]=.05;p[3,-1,2]=.025;f[4]=1
old=navigation_priority_reward(p,v,q,g,f,12);new=navigation_priority_reward(p,v,q,g,f,13)
assert torch.allclose(new[:4],(old[:4]+.8)/1.7)
assert new[0]>new[1] and new[0]>new[2] and new[0]>new[3]
assert new[4]==old[4] and new[4]<-1
assert (new[:4]>=0).all() and (new[:4]<=1).all()
print(json.dumps(dict(status='pass',scope='Production reward13 affine nonfailure transform, proximity/speed/ground ordering and crash negative preservation',values=new.tolist())))
