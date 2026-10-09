"""Seeded lift/carry residual task, kept separate from the simulator contract."""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .env import TetherEnv, Config


class LiftCarryTask(gym.Wrapper):
    def __init__(self, residual_scale=.05, episode_seconds=12):
        super().__init__(TetherEnv(Config(duration=episode_seconds)))
        self.residual_scale=residual_scale
        self.action_space=spaces.Box(-1.,1.,(4,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.env.observation_space.shape[0]+6,),np.float64)
        self.phase=0.
        self.previous=np.zeros(4)

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed,options={'scenario':'lift_carry'})
        rng=self.env.np_random
        # A small initial top-joint tilt changes the hanging configuration.
        angle=rng.uniform(-.12,.12)
        self.env.data.qpos[7:11]=[np.cos(angle/2),0,np.sin(angle/2),0]
        self.env.data.qpos[:2]=rng.uniform(-.05,.05,2)
        self.phase=float(rng.uniform(0,2*np.pi))
        self.previous[:]=0
        mujoco.mj_forward(self.env.model,self.env.data)
        return self._task_obs(self.env._obs()),self.env.metrics()

    def _task_obs(self,obs):
        return np.r_[obs,self.previous,np.sin(self.phase),np.cos(self.phase)]

    def step(self,action):
        correction=np.asarray(action,dtype=float)
        if correction.shape!=(4,) or not np.isfinite(correction).all():
            raise ValueError('Four finite residual commands required')
        correction=np.clip(correction,-1,1)
        self.env.wind[:]=[.012*np.sin(2*np.pi*self.env.data.time/2.5+self.phase),0,0]
        motors=np.clip(self.env.baseline()+self.residual_scale*correction,0,1)
        obs,_,terminated,truncated,info=self.env.step(motors)
        payload_target=self.env.target-[0,0,self.env.config.length+.02]
        payload_error=float(np.linalg.norm(self.env.data.xpos[self.env.payload]-payload_target))
        drone_error=float(np.linalg.norm(self.env.data.qpos[:3]-self.env.target))
        swing=np.radians(info['swing_deg'])
        reward=-1.5*payload_error-.5*drone_error-.1*swing*swing-.01*np.square(correction).sum()-.01*np.square(correction-self.previous).sum()
        if terminated:
            reward-=25
        self.previous[:]=correction
        info.update(payload_error_m=payload_error,drone_error_m=drone_error)
        return self._task_obs(obs),float(reward),terminated,truncated,info
