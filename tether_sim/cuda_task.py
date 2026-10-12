"""Experimental PufferLib vector adapter; all physics runs in C/CUDA."""
import numpy as np
import torch
from gymnasium.spaces import Box
from .cuda_backend import CudaTether


class CudaVector:
    def __init__(self,count=256,direct=False,reward_version=1,phase='destination'):
        self.direct=direct
        self.reward_version=reward_version;self.phase=phase
        self.num_agents=self.agents_per_batch=count
        self.single_observation_space=Box(-np.inf,np.inf,(98,),np.float32)
        self.single_action_space=self.action_space=Box(-1,1,(4,),np.float32)
        self.sim=CudaTether(count)
        if direct:self.sim.params[:,8]=1
        self.ids=np.arange(count);self.mask=np.ones(count,dtype=bool)
        self.steps=torch.zeros(count,device='cuda',dtype=torch.int32)
        self.goal=torch.zeros(count,3,device='cuda')
        self.reward=torch.zeros(count,device='cuda');self.done=torch.zeros(count,device='cuda',dtype=torch.bool)
        self.truncated=torch.zeros_like(self.done)
        self.start=self.sim.position.clone()
        if direct:
            self.start.zero_();self.start[:,:,0]=torch.arange(14,device='cuda')*.6/13
            self.start[:,:,2]=.025;self.start[:,0,2]=.05
        self.allocation=torch.tensor([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[-.006,.006,-.006,.006]],device='cuda')

    def async_reset(self,seed=41):
        torch.manual_seed(seed);self.sim.reset();self.steps.zero_()
        self.goal[:,:2]=torch.rand_like(self.goal[:,:2])*.8-.4;self.goal[:,2]=1.4
        # Six boxes below or beside the route; randomized domains for prototype.
        self.sim.boxes=torch.zeros(self.num_agents,6,6,device='cuda')
        self.sim.boxes[:,:,:2]=torch.rand(self.num_agents,6,2,device='cuda')*3-1.5
        self.sim.boxes[:,:,2]=.35;self.sim.boxes[:,:,3:]=torch.tensor([.1,.1,.35],device='cuda')
        self.sim.wind[:,:2]=(torch.rand(self.num_agents,2,device='cuda')-.5)*.012
        if self.direct:
            self.sim.position[:]=self.start;self.sim.motors.zero_()
            self.sim.boxes[:,:,:2]+=2  # Keep takeoff strip clear in first experiment.
            if self.phase in ('takeoff','hover'):
                self.goal[:,:2]=self.start[:,-1,:2]
                self.goal[:,2]=.2 if self.phase=='takeoff' else .6
        self.previous_error=torch.linalg.vector_norm(self.sim.position[:,-1]-self.goal,dim=1)

    def obs(self):
        s=self.sim
        return torch.cat([(s.position-self.goal[:,None]).flatten(1),s.velocity.flatten(1),s.orientation,s.motors,self.goal],1).float()

    def baseline(self):
        s=self.sim;q=s.orientation[:,:4];w,x,y,z=q.unbind(1)
        R=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
          2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
          2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
        target=self.goal.clone();target[:,2]+=.6
        acc=(2*(target-s.position[:,0])-2.8*s.velocity[:,0]).float();acc[:,2]+=9.81
        desired=acc/torch.linalg.vector_norm(acc,dim=1,keepdim=True).clamp_min(1e-8)
        torque=(R.transpose(1,2)@(.012*torch.linalg.cross(R[:,:,2],desired)-.0012*(R@s.orientation[:,4:,None]).squeeze(2))[:,:,None]).squeeze(2)
        torque[:,2]=-.0002*s.orientation[:,6]
        force=s.mass.sum(1)*(acc*R[:,:,2]).sum(1)
        return (torch.linalg.solve(self.allocation,torch.cat([force[:,None],torque],1).T).T/s.params[:,2:3]).clamp(0,1)

    def send(self,actions):
        s=self.sim;residual=torch.as_tensor(actions,device='cuda',dtype=torch.float32)
        command=(.5*(residual+1)).clamp(0,1) if self.direct else self.baseline()+.03*residual
        s.step(command.contiguous())
        if bool((s.failed==2).any()):
            raise FloatingPointError('CUDA physics numerical failure; trial invalid')
        self.steps+=1
        error=torch.linalg.vector_norm(s.position[:,-1]-self.goal,dim=1)
        self.reward=(-error-.02*residual.square().sum(1)).float()
        if self.direct:
            self.reward=(20*(self.previous_error-error)-.2*error-.002*s.orientation[:,4:].square().sum(1)-5*(s.failed!=0)).float()
            if self.reward_version==2:
                speed=s.velocity[:,-1].square().sum(1)
                upright=1-2*(s.orientation[:,1:3].square().sum(1))
                self.reward=(.8*torch.exp(-4*error.square())+.2*upright-
                  .15*speed-.002*s.orientation[:,4:].square().sum(1)-
                  .8*(s.position[:,0,2]<.15)-5*(s.failed!=0)).float()
        self.previous_error=error
        self.truncated=self.steps>=500;self.done=(s.failed!=0)|self.truncated
        # Preserve terminal reward before resetting failed/timed-out states.
        mask=self.done
        if bool(mask.any()):
            initial=self.start
            s.position[mask]=initial[mask];s.velocity[mask]=0;s.orientation[mask]=0;s.orientation[mask,0]=1
            s.motors[mask]=0 if self.direct else s.mass[mask].sum(1,keepdim=True)*9.81/4;s.failed[mask]=0;self.steps[mask]=0
            self.previous_error[mask]=torch.linalg.vector_norm(s.position[mask,-1]-self.goal[mask],dim=1)

    def recv(self):
        return self.obs(),self.reward,self.done,self.truncated,[],self.ids,self.mask

    def close(self): pass


class CudaPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(98,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh())
        self.actor=torch.nn.Linear(128,4);self.value=torch.nn.Linear(128,1)
        torch.nn.init.zeros_(self.actor.weight);torch.nn.init.zeros_(self.actor.bias)
        self.log_std=torch.nn.Parameter(torch.full((4,),-2.))
    def forward(self,obs,state=None):
        h=self.body(obs);return torch.distributions.Normal(self.actor(h),self.log_std.exp()),self.value(h)
    def forward_eval(self,obs,state=None):return self.forward(obs,state)
