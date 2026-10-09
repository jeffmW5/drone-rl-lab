"""Random clutter navigation: destination -> A* route -> PD + learned residual."""
import numpy as np
import mujoco
from dataclasses import replace
from gymnasium import spaces
from .env import TetherEnv
from .generalize import GeneralLiftCarryTask
from .planning import plan_route


class NavigationTask(GeneralLiftCarryTask):
    def __init__(self,residual_scale=.03,episode_seconds=30):
        super().__init__(residual_scale,episode_seconds)
        self.route=None
        # Six boxes x seven parameters, 13 domain variables, current waypoint xy.
        self.extra=np.zeros(57)
        self.observation_space=spaces.Box(-np.inf,np.inf,(self.env.observation_space.shape[0]+6+57,),np.float64)

    def reset(self,*,seed=None,options=None):
        self.route=None
        # Parent initialization uses its own 27-value intermediate contract.
        self.extra=np.zeros(27)
        super().reset(seed=seed)
        r=self.rng
        for _ in range(100):
            heading=r.uniform(-np.pi,np.pi)
            self.goal=r.uniform(1.1,1.65)*np.array([np.cos(heading),np.sin(heading)])
            boxes=[]
            # Deliberately block the straight route; the other boxes add clutter.
            for i in range(6):
                xy=self.goal*.5+r.uniform(-.06,.06,2) if i==0 else r.uniform(-1.7,1.7,2)
                if np.linalg.norm(xy)<.45 or np.linalg.norm(xy-self.goal)<.45:
                    break
                boxes.append((*xy,.75,r.uniform(.10,.20),r.uniform(.10,.20),.75,0.))
            if len(boxes)!=6:
                continue
            try:
                route=plan_route([0,0],self.goal,boxes)
            except ValueError:
                continue
            if np.linalg.norm(np.diff(route,axis=0),axis=1).sum()<3.5:
                break
        else:
            raise RuntimeError('Could not sample a reachable clutter layout')
        c=replace(self.env.config,obstacles=tuple(boxes))
        self.env.close();self.env=TetherEnv(c)
        self.env.reset(seed=seed,options={'scenario':'lift_carry'})
        self.env.scenario='navigation'
        self.env.model.actuator_gainprm[:,0]=self.gains
        self.route=route
        self.route_start_time=4.
        self.distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        self.obstacle_ids={self.env.model.geom(f'obstacle{i}').id for i in range(6)}
        self.previous[:]=0;self.external_wind[:]=0
        self.extra=np.r_[np.asarray(boxes).ravel(),c.payload_mass,c.cable_mass,c.length,c.motor_tau,self.gains,self.amplitude,self.period,self.goal,self.altitude,[0,0]]
        self.domain.update(obstacles=boxes,goal_xy=self.goal.tolist(),route=route.tolist(),planner='inflated 2D A*',planner_margin_m=.18)
        self.domain.pop('gate_gap_m',None)
        self._trajectory();mujoco.mj_forward(self.env.model,self.env.data)
        return self._task_obs(self.env._obs()),self.env.metrics()|{'domain':self.domain}

    def _trajectory(self):
        if self.route is None:
            return super()._trajectory()
        t=self.env.data.time
        lift=np.clip(t/3,0,1);lift=3*lift*lift-2*lift*lift*lift
        distance=min(max(0,t-self.route_start_time)*.16,self.distances[-1])
        i=min(np.searchsorted(self.distances,distance,side='right')-1,len(self.route)-2)
        f=(distance-self.distances[i])/max(self.distances[i+1]-self.distances[i],1e-8)
        xy=self.route[i]*(1-f)+self.route[i+1]*f
        self.env.target[:]=[*xy,self.start_z+(self.altitude-self.start_z)*lift]
        self.extra[-2:]=self.route[i+1]
        self.env._update_target()

    def set_goal(self,xy):
        xy=np.asarray(xy,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():
            raise ValueError('Goal requires two finite coordinates')
        route=plan_route(self.env.data.qpos[:2],xy,self.env.config.obstacles)
        self.goal=xy;self.route=route
        self.distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        self.route_start_time=max(4.,self.env.data.time)
        self.extra[-5:-3]=xy
        self.domain.update(goal_xy=xy.tolist(),route=route.tolist())
        self._trajectory()
