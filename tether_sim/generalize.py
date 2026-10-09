"""Versioned task with seeded domains, routes and physical gate obstacles."""
import numpy as np
import mujoco
from gymnasium import spaces
from .env import TetherEnv, Config
from .task import LiftCarryTask


class GeneralLiftCarryTask(LiftCarryTask):
    def __init__(self,residual_scale=.03,episode_seconds=16):
        super().__init__(residual_scale,episode_seconds)
        self.duration=episode_seconds
        # Two obstacles x seven geometric parameters, plus domain/route variables.
        self.extra=np.zeros(27)
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.env.observation_space.shape[0]+6+27,),np.float64)
        self.domain={}

    def _task_obs(self,obs):
        return np.r_[super()._task_obs(obs),self.extra]

    def reset(self,*,seed=None,options=None):
        # Keep the seeded generator across model rebuilds; reset(seed=None) advances it.
        if seed is not None or not hasattr(self,'rng'):
            self.rng=np.random.default_rng(seed)
        r=self.rng
        heading=r.uniform(-np.pi,np.pi)
        distance=r.uniform(.45,.85)
        direction=np.array([np.cos(heading),np.sin(heading)])
        perpendicular=np.array([-direction[1],direction[0]])
        self.goal=distance*direction
        gap=r.uniform(.42,.68)
        center=self.goal*r.uniform(.4,.65)
        obstacles=[]
        for sign in (-1,1):
            xy=center+sign*perpendicular*(gap/2+.06)
            obstacles.append((*xy,.75,.075,.06,.75,heading))
        c=Config(payload_mass=r.uniform(.003,.008),cable_mass=r.uniform(.0007,.0013),length=r.uniform(.45,.75),motor_tau=r.uniform(.02,.04),duration=self.duration,obstacles=tuple(obstacles))
        self.env.close()
        self.env=TetherEnv(c)
        self.env.reset(seed=int(r.integers(0,2**31)),options={'scenario':'lift_carry'})
        self.env.scenario='general_lift_carry'
        self.phase=float(r.uniform(0,2*np.pi))
        self.amplitude=float(r.uniform(0,.018))
        self.period=float(r.uniform(2,4))
        self.gains=r.uniform(.96,1.04,4)
        self.env.model.actuator_gainprm[:,0]=self.gains
        self.altitude=float(r.uniform(1.15,1.4))
        self.start_z=c.length+.048
        angle=r.uniform(-.12,.12)
        self.env.data.qpos[7:11]=[np.cos(angle/2),0,np.sin(angle/2),0]
        self.env.data.qpos[:2]=r.uniform(-.035,.035,2)
        self.previous[:]=0
        self.external_wind[:]=0
        self.extra=np.r_[np.asarray(obstacles).ravel(),c.payload_mass,c.cable_mass,c.length,c.motor_tau,self.gains,self.amplitude,self.period,self.goal,self.altitude]
        assert self.extra.shape==(27,)
        self.obstacle_ids={self.env.model.geom(f'obstacle{i}').id for i in range(2)}
        self.domain=dict(payload_mass=c.payload_mass,cable_mass=c.cable_mass,length=c.length,motor_tau=c.motor_tau,motor_gains=self.gains.tolist(),wind_amplitude_N=self.amplitude,wind_period_s=self.period,goal_xy=self.goal.tolist(),gate_gap_m=gap,obstacles=obstacles)
        self._trajectory()
        mujoco.mj_forward(self.env.model,self.env.data)
        return self._task_obs(self.env._obs()),self.env.metrics()|{'domain':self.domain}

    def _trajectory(self):
        t=self.env.data.time
        lift=np.clip(t/3,0,1)
        carry=np.clip((t-4)/6,0,1)
        smooth=lambda f:3*f*f-2*f*f*f
        self.env.target[:]=[* (self.goal*smooth(carry)),self.start_z+(self.altitude-self.start_z)*smooth(lift)]
        self.env._update_target()

    def step(self,action):
        self._trajectory()
        if any(c.geom1 in self.obstacle_ids or c.geom2 in self.obstacle_ids for c in self.env.data.contact[:self.env.data.ncon]):
            info=self.env.metrics()
            info.update(obstacle_collision=True,obstacle_clearance_m=self.clearance(),domain=self.domain,payload_error_m=float(np.linalg.norm(self.env.data.xpos[self.env.payload]-(self.env.target-[0,0,self.env.config.length+.02]))),drone_error_m=float(np.linalg.norm(self.env.data.qpos[:3]-self.env.target)))
            return self._task_obs(self.env._obs()),-50.,True,False,info
        t=self.env.data.time
        desired=self.amplitude*np.array([np.sin(2*np.pi*t/self.period+self.phase),np.cos(2*np.pi*t/self.period+self.phase)*.6,0])
        # Parent supplies its original gust. Cancel it, preserving external wind pulses.
        pulse=self.external_wind.copy()
        self.external_wind[:]=pulse+desired-[.012*np.sin(2*np.pi*t/2.5+self.phase),0,0]
        _,reward,done,truncated,info=super().step(action)
        self.external_wind[:]=pulse
        collision=any(c.geom1 in self.obstacle_ids or c.geom2 in self.obstacle_ids for c in self.env.data.contact[:self.env.data.ncon])
        clearance=self.clearance()
        reward-=.04*np.square(action).sum()+2*max(0,.12-clearance)**2
        if collision:
            reward-=50
            done=True
        self._trajectory()
        info.update(obstacle_collision=collision,obstacle_clearance_m=clearance,domain=self.domain)
        return self._task_obs(self.env._obs()),float(reward),bool(done),truncated,info

    def clearance(self):
        ids=[self.env.drone,self.env.payload]+[self.env.model.body(f'link{i}').id for i in range(self.env.config.segments)]
        points=self.env.data.xpos[ids]
        radii=np.r_[.05,.025,np.full(self.env.config.segments,.0015)]
        clearance=float('inf')
        for o in self.env.config.obstacles:
            center=np.array(o[:3]);half=np.array(o[3:6]);a=o[6]
            R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            q=np.abs((points-center)@R)-half
            sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)-radii
            clearance=min(clearance,float(sdf.min()))
        return clearance


def make_task(config):
    if config.get('task')=='random_navigation_v1':
        from .navigation import NavigationTask
        return NavigationTask(config['residual_scale'],config['episode_seconds'])
    cls=GeneralLiftCarryTask if config.get('task')=='general_lift_carry_v1' else LiftCarryTask
    return cls(config['residual_scale'],config['episode_seconds'])
