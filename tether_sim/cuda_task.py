"""Experimental PufferLib vector adapter; all physics runs in C/CUDA."""
import numpy as np
import torch
from gymnasium.spaces import Box
from .cuda_backend import CudaTether


def bounded_reward(reward):
    """Strictly monotone compression; avoid downstream hard-clip plateaus."""
    return reward/(1+reward.abs())


def position_reward_error(positions, goal, reward_version):
    if reward_version == 4:
        target = goal.clone()
        target[:, 2] += .6
        return torch.linalg.vector_norm(positions[:, 0] - target, dim=1)
    return torch.linalg.vector_norm(positions[:, -1] - goal, dim=1)


def position_reward(error, reward_version):
    # Bounded alternative with the same value and quadratic curvature at zero.
    if reward_version == 5:return .8/(1+4*error.square())
    return .8*torch.exp(-4*error.square())


def configure_exploration(policy, log_std):
    if log_std is None:return
    values=np.asarray(log_std,dtype=float)
    if values.shape not in ((),(4,)) or not np.isfinite(values).all() or (values < -6).any() or (values > 0).any():
        raise ValueError('initial_log_std must be a scalar or four finite values within [-6,0]')
    with torch.no_grad():policy.log_std.copy_(torch.as_tensor(values,device=policy.log_std.device,dtype=policy.log_std.dtype).expand_as(policy.log_std))


def drone_velocity_cost(velocity, reward_version):
    return .05*velocity[:,0].square().sum(1) if reward_version==6 else torch.zeros_like(velocity[:,0,0])


def angular_velocity_cost(angular_velocity,reward_version):
    if reward_version==7:return torch.zeros_like(angular_velocity[:,0])
    return .002*angular_velocity.square().sum(1)


def approach_stop_reward(positions, velocity, orientation, goal, failed, reward_version=8):
    """Reward8: bounded distance + goal-directed velocity matching, no PID/actions."""
    delta=goal-positions[:,-1]
    distance=torch.linalg.vector_norm(delta,dim=1)
    desired_velocity=.5*delta/(.25+distance[:,None])
    velocity_error=(velocity[:,-1]-desired_velocity).square().sum(1)
    drone_speed=velocity[:,0].square().sum(1)
    drone_bonus=.1/(1+4*drone_speed)
    if reward_version==9:
        drone_goal=goal.clone();drone_goal[:,2]+=.6
        drone_error=(positions[:,0]-drone_goal).square().sum(1)
        drone_bonus=.1/(1+4*drone_error)
    upright=1-2*orientation[:,1:3].square().sum(1)
    return (.45/(1+4*distance.square())+.35/(1+4*velocity_error)+
        .1*upright+drone_bonus-.8*(positions[:,0,2]<.15)-5*(failed!=0)).float()


def navigation_priority_reward(positions,velocity,orientation,goal,failed,reward_version=10):
    """Reward10: suppress pre-lift task reward; proximity and stopping dominate."""
    distance2=(positions[:,-1]-goal).square().sum(1)
    speed2=velocity[:,-1].square().sum(1)
    if reward_version in (12,13):
        carrier_goal=goal.clone();carrier_goal[:,2]+=.6
        delta=carrier_goal-positions[:,0]
        desired=.5*delta/(.25+torch.linalg.vector_norm(delta,dim=1)[:,None])
        speed2=(velocity[:,0]-desired).square().sum(1)
    # Smooth release after payload lifts from its 25mm ground-contact radius.
    flight_gate=((positions[:,-1,2]-.025)/.075).clamp(0,1)
    proximity=1/(1+4*distance2) if reward_version in (11,12,13) else torch.exp(-4*distance2)
    navigation=.85*proximity*(.25+.75/(1+4*speed2))*flight_gate
    upright=1-2*orientation[:,1:3].square().sum(1)
    base=(navigation+.05*upright-.8*(positions[:,0,2]<.15)-5*(failed!=0)).float()
    if reward_version==13:
        # Shift nonfailure rewards; intentionally changes episodic survival bias.
        return torch.where(failed!=0,base,(base+.8)/1.7)
    return base

def training_starts(start,version,phase):
    if version not in (1,2,3):raise ValueError('Unsupported training reset version')
    if version==1:return start
    if version==3:
        if phase not in ('takeoff','hover'):raise ValueError('Valid ground reset requires takeoff/hover phase')
        result=start.clone()
        fraction=torch.arange(14,device=start.device,dtype=start.dtype)/13
        result[:,:,0]=fraction*((.6*.6-.025*.025)**.5)
        result[:,:,2]=.05-fraction*.025
        return result
    if phase not in ('takeoff','hover'):raise ValueError('Mixed starts require takeoff/hover phase')
    result=start.clone()
    result[1::2,:,0]=start[1::2,-1,0,None]
    result[1::2,:,1]=start[1::2,-1,1,None]
    height=.2 if phase=='takeoff' else .6
    result[1::2,:,2]=height+.6-torch.arange(14,device=start.device,dtype=start.dtype)*.6/13
    return result


class LandingBeforeSuccess:
    """Episode-local reward gate; does not terminate or control the drone."""
    def __init__(self,count,device,version=1):
        if version not in (1,2):raise ValueError('Unsupported landing gate version')
        self.version=version
        self.airborne=torch.zeros(count,device=device,dtype=torch.bool)
        self.penalized_episode=torch.zeros_like(self.airborne)
        self.success=torch.zeros_like(self.airborne)
        self.hold=torch.zeros(count,device=device,dtype=torch.int32)
        self.penalized_steps=torch.zeros((),device=device,dtype=torch.int64)
        self.takeoffs=torch.zeros_like(self.penalized_steps)
        self.successes=torch.zeros_like(self.penalized_steps)
    def reset(self,mask=None):
        if mask is None:
            self.airborne.zero_();self.success.zero_();self.hold.zero_();self.penalized_episode.zero_()
        else:
            self.airborne[mask]=False;self.success[mask]=False;self.hold[mask]=0;self.penalized_episode[mask]=False
    def update(self,positions,velocity,goal,failed):
        height=positions[:,0,2]
        self.takeoffs+=((height>.3)&~self.airborne).sum()
        self.airborne|=height>.3
        stable=(torch.linalg.vector_norm(positions[:,-1]-goal,dim=1)<.1)&(
            torch.linalg.vector_norm(velocity[:,-1],dim=1)<.2)&(
            height>.3)&(positions[:,-1,2]>.05)&(failed==0)
        self.hold=torch.where(stable,self.hold+1,0)
        self.successes+=((self.hold>=50)&~self.success).sum()
        self.success|=self.hold>=50 # 1 second at 50 Hz, prototype criterion
        premature=self.airborne&~self.success&(height<=.051)&(failed==0)
        if self.version==2:
            premature=premature&~self.penalized_episode
            self.penalized_episode|=premature
        self.penalized_steps+=premature.sum()
        return premature
    def diagnostics(self):
        return dict(penalized_agent_steps=int(self.penalized_steps),episode_takeoffs=int(self.takeoffs),episode_successes=int(self.successes))


def zone_hover_reward(positions,velocity,orientation,center,failed):
    """Diagnostic14: carrier hover, square XY bounds, preserved full tether."""
    displacement=positions[:,0,:2]-center[:,:2]
    extent=displacement.abs().amax(1)
    escaped=extent>=2.
    band=(extent>1.)&~escaped
    height_error=positions[:,0,2]-.8
    speed2=velocity[:,0].square().sum(1)
    upright=1-2*orientation[:,1:3].square().sum(1)
    airborne=((positions[:,0,2]-.05)/.25).clamp(0,1)
    reward=airborne/(1+4*height_error.square()+2*speed2)+.05*upright-.8*(positions[:,0,2]<.15)
    reward=torch.where(band,-torch.ones_like(reward),reward)
    reward=torch.where(escaped|(failed!=0),-torch.ones_like(reward),reward)
    return reward.float(),escaped,band


class CudaVector:
    def __init__(self,count=256,direct=False,reward_version=1,phase='destination',control_version=1,training_region=None,physics_version=1,training_reset_version=1,landing_penalty_version=0,diagnostic_zone_version=0):
        if diagnostic_zone_version not in (0,1):raise ValueError("Unsupported diagnostic zone")
        self.diagnostic_zone_version=diagnostic_zone_version
        self.zone_counts=dict(band_steps=0,escapes=0)
        self.direct=direct
        self.terminal_observer=None
        self.training_region=training_region
        self.control_version=control_version
        self.reward_version=reward_version;self.phase=phase
        self.num_agents=self.agents_per_batch=count
        self.single_observation_space=Box(-np.inf,np.inf,(98,),np.float32)
        self.single_action_space=self.action_space=Box(-1,1,(4,),np.float32)
        if physics_version not in (1,2,3,4):raise ValueError('Unsupported physics version')
        self.sim=CudaTether(count,library=f'tmp/cuda-tether-v{physics_version:03d}/libtether.so')
        if landing_penalty_version not in (0,1,2):raise ValueError('Unsupported landing penalty version')
        self.landing_gate=LandingBeforeSuccess(count,'cuda',landing_penalty_version) if landing_penalty_version else None
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
            self.start=training_starts(self.start,training_reset_version,phase)
        self.allocation=torch.tensor([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[-.006,.006,-.006,.006]],device='cuda')

    def async_reset(self,seed=41):
        torch.manual_seed(seed);self.sim.reset();self.steps.zero_()
        if self.landing_gate is not None:self.landing_gate.reset()
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
        if self.diagnostic_zone_version:
            self.sim.boxes=torch.empty(self.num_agents,0,6,device="cuda")
            self.sim.wind.zero_()
            self.goal[:,:2]=self.start[:,0,:2]
            self.goal[:,2]=.2 # Carrier .8m, hanging payload .2m.
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
        if self.direct and self.control_version==2:
            # Learned collective + learned roll/pitch/yaw, fixed motor geometry.
            # No state-feedback controller; all four channels come from policy.
            mix=torch.tensor([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]],device='cuda',dtype=torch.float32)
            command=(.5*(residual[:,0:1]+1)+.02*(residual[:,1:]@mix.T)).clamp(0,1)
        s.step(command.contiguous())
        if bool((s.failed==2).any()):
            raise FloatingPointError('CUDA physics numerical failure; trial invalid')
        self.steps+=1
        error=torch.linalg.vector_norm(s.position[:,-1]-self.goal,dim=1)
        self.reward=(-error-.02*residual.square().sum(1)).float()
        if self.direct:
            self.reward=(20*(self.previous_error-error)-.2*error-.002*s.orientation[:,4:].square().sum(1)-5*(s.failed!=0)).float()
            if self.reward_version in (2,3,4,5,6,7):
                speed=s.velocity[:,-1].square().sum(1)
                upright=1-2*(s.orientation[:,1:3].square().sum(1))
                reward_error=position_reward_error(s.position,self.goal,self.reward_version)
                self.reward=(position_reward(reward_error,self.reward_version)+.2*upright-
                  .15*speed-angular_velocity_cost(s.orientation[:,4:],self.reward_version)-
                  .8*(s.position[:,0,2]<.15)-5*(s.failed!=0)).float()
                self.reward-=drone_velocity_cost(s.velocity,self.reward_version).float()
                if self.reward_version==3:self.reward=bounded_reward(self.reward)
        if self.direct and self.reward_version in (8,9):
            self.reward=approach_stop_reward(s.position,s.velocity,s.orientation,self.goal,s.failed,self.reward_version)
        if self.direct and self.reward_version in (10,11,12,13):
            self.reward=navigation_priority_reward(s.position,s.velocity,s.orientation,self.goal,s.failed,self.reward_version)
        if getattr(self,'landing_gate',None) is not None:
            premature=self.landing_gate.update(s.position,s.velocity,self.goal,s.failed)
            self.reward=torch.where(premature,torch.full_like(self.reward,-1),self.reward)
        if self.diagnostic_zone_version:
            self.reward,escaped,band=zone_hover_reward(s.position,s.velocity,s.orientation,self.start[:,0],s.failed)
            self.zone_counts["band_steps"]+=int(band.sum())
            self.zone_counts["escapes"]+=int(escaped.sum())
            s.failed=torch.where(escaped & (s.failed==0),torch.full_like(s.failed,3),s.failed)
        self.previous_error=error
        self.truncated=self.steps>=500;self.done=(s.failed!=0)|self.truncated
        if self.training_region is not None:
            escaped=(torch.linalg.vector_norm(s.position[:,0,:2]-self.goal[:,:2],dim=1)>self.training_region['radius_m']) | (s.position[:,0,2]>self.training_region['ceiling_m'])
            self.truncated=self.truncated|escaped
            self.done=self.done|escaped
        # Preserve terminal reward before resetting failed/timed-out states.
        mask=self.done
        if bool(mask.any()):
            if self.terminal_observer is not None:
                self.terminal_observer(self.obs(),self.done,self.truncated,s.failed,self.reward)
            if getattr(self,'landing_gate',None) is not None:self.landing_gate.reset(mask)
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
        self.value_gradient_version=1
        self.register_buffer('obs_scale',torch.ones(98))
    def configure_input_conditioning(self,version):
        if version not in (1,2):raise ValueError('Unsupported input conditioning version')
        scale=torch.ones_like(self.obs_scale)
        if version==2:
            scale[:42]=.6  # Relative particle positions, metres.
            scale[88:91]=10  # Angular rates, radians/second.
            scale[91:95]=.2  # Motor forces, newtons.
        # Preserve the loaded initial function while changing optimizer geometry.
        with torch.no_grad():
            ratio=scale/self.obs_scale
            self.body[0].weight.mul_(ratio)
            if hasattr(self,'value_body'):self.value_body[0].weight.mul_(ratio)
            self.obs_scale.copy_(scale)
    def enable_separate_value(self):
        import copy
        if not hasattr(self,'value_body'):self.value_body=copy.deepcopy(self.body)
        self.value_gradient_version=3
    def forward(self,obs,state=None):
        obs=obs/self.obs_scale
        h=self.body(obs)
        value_h=self.value_body(obs) if self.value_gradient_version==3 else (h.detach() if self.value_gradient_version==2 else h)
        return torch.distributions.Normal(self.actor(h),self.log_std.exp()),self.value(value_h)
    def forward_eval(self,obs,state=None):return self.forward(obs,state)
