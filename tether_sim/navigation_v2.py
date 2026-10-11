"""Versioned navigation with rotor/deck collisions and tracking-aware setpoints.

Privileged geometry/state remains required. Model improvements are not hardware
calibration. Observation dimensions stay compatible with V005 transfer.
"""
from dataclasses import replace
import mujoco
import numpy as np
from .env import TetherEnv
from .navigation import NavigationTask
from .planning import plan_route


class NavigationTaskV2(NavigationTask):
    def __init__(self,residual_scale=.03,episode_seconds=45,navigation=None,physics=None):
        self.navigation=navigation or {}
        self.physics=physics or {}
        self._tracking_ready=False
        super().__init__(residual_scale,episode_seconds)
        self.margin=float(self.navigation.get('planner_margin_m',.26))
        self.cruise=float(self.navigation.get('speed_m_s',.12))
        self.acceleration=float(self.navigation.get('acceleration_m_s2',.08))
        if min(self.margin,self.cruise,self.acceleration)<=0:
            raise ValueError('Positive navigation margin, speed, acceleration required')

    def reset(self,*,seed=None,options=None):
        self._tracking_ready=False
        for attempt in range(100):
            super().reset(seed=seed if attempt==0 else None)
            try:
                route=plan_route([0,0],self.goal,self.env.config.obstacles,margin=self.margin)
            except ValueError:
                continue
            if np.linalg.norm(np.diff(route,axis=0),axis=1).sum()<3.5:
                break
        else:
            raise RuntimeError('No layout reachable with v2 clearance margin')
        ranges=self.physics.get('randomization',{})
        def sample(name,default):
            bounds=ranges.get(name,default)
            return float(self.rng.uniform(*bounds))
        mass=sample('drone_mass_kg',[.033,.046])
        inertia_scale=sample('inertia_scale',[.85,1.15])
        overrides=dict(self.physics.get('overrides',{}))
        if 'segments' in overrides:
            raise ValueError('Changing segment count changes V005 observation dimensions')
        cfg=replace(self.env.config,drone_mass=mass,
                    drone_inertia=tuple(v*inertia_scale for v in self.env.config.drone_inertia),
                    max_motor_thrust=sample('max_motor_thrust_N',[.17,.20]),
                    linear_drag=sample('linear_drag_N_s_m',[.012,.018]),
                    rotor_collisions=True,deck_collisions=True,
                    substep_collision_checks=True,**overrides)
        self.env.close();self.env=TetherEnv(cfg)
        self.env.reset(seed=seed,options={'scenario':'lift_carry'})
        self.env.scenario='navigation'
        self.env.model.actuator_gainprm[:,0]=self.gains
        # The v1 rebuild discarded these initial-state perturbations.
        angle=self.rng.uniform(-.12,.12)
        self.env.data.qpos[7:11]=[np.cos(angle/2),0,np.sin(angle/2),0]
        self.env.data.qpos[:2]=self.rng.uniform(-.02,.02,2)
        self.route=route
        self.distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        self.segment=0;self.progress=0.;self.speed=0.;self.last_target_time=0.
        self.route_start_time=4.;self._tracking_ready=True
        self.obstacle_ids=self.env.obstacle_geom_ids
        self.previous[:]=0;self.external_wind[:]=0
        self.domain.update(drone_mass=cfg.drone_mass,drone_inertia=cfg.drone_inertia,
                           max_motor_thrust_N=cfg.max_motor_thrust,linear_drag_N_s_m=cfg.linear_drag,
                           route=route.tolist(),planner_margin_m=self.margin,
                           physics_version='rotor_deck_substep_v2',
                           clearance_method='MuJoCo geometry distance',
                           hardware='Crazyflie 2.1 + Flow deck + AI deck',
                           calibration='uncalibrated; mass/thrust/drag ranges are assumptions')
        self._trajectory();mujoco.mj_forward(self.env.model,self.env.data)
        return self._task_obs(self.env._obs()),self._metrics()

    def _trajectory(self):
        if not self._tracking_ready or self.route is None:
            return super()._trajectory()
        t=float(self.env.data.time)
        dt=max(0.,t-self.last_target_time)
        self.last_target_time=t
        if t>=self.route_start_time and dt>0:
            end=self.distances[self.segment+1]
            if end-self.progress<1e-8 and self.segment<len(self.route)-2:
                # Hold the corner until the real simulated drone catches up.
                if np.linalg.norm(self.env.data.qpos[:2]-self.route[self.segment+1])<.10 and np.linalg.norm(self.env.data.qvel[:2])<.10:
                    self.segment+=1
                    end=self.distances[self.segment+1]
            error=np.linalg.norm(self.env.data.qpos[:2]-self.env.target[:2])
            tracking_factor=float(np.clip((.22-error)/.14,0.,1.))
            desired=min(self.cruise*tracking_factor,np.sqrt(2*self.acceleration*max(0,end-self.progress)))
            self.speed+=float(np.clip(desired-self.speed,-self.acceleration*dt,self.acceleration*dt))
            self.progress=min(end,self.progress+self.speed*dt)
        a,b=self.distances[self.segment:self.segment+2]
        f=float(np.clip((self.progress-a)/max(b-a,1e-8),0,1))
        xy=(1-f)*self.route[self.segment]+f*self.route[self.segment+1]
        lift=float(np.clip(t/3,0,1));lift=3*lift*lift-2*lift**3
        self.env.target[:]=[*xy,self.start_z+(self.altitude-self.start_z)*lift]
        self.extra[-2:]=self.route[self.segment+1]
        self.env._update_target()

    def clearance(self):
        # Include complete cable capsules, propeller disks and deck volumes,
        # instead of sampling only link origins with approximate radii.
        value=2.
        for obstacle in self.obstacle_ids:
            for geom in self.env.dynamic_collision_geoms:
                value=min(value,mujoco.mj_geomDistance(self.env.model,self.env.data,geom,obstacle,value,None))
        return float(value)

    def _metrics(self):
        info=self.env.metrics()
        payload_goal=np.r_[self.goal,self.altitude-self.env.config.length-.02]
        info.update(domain=self.domain,goal_distance_m=float(np.linalg.norm(self.env.data.qpos[:2]-self.goal)),
                    payload_goal_error_m=float(np.linalg.norm(self.env.data.xpos[self.env.payload]-payload_goal)),
                    route_remaining_m=float(max(0,self.distances[-1]-self.progress)),
                    goal_reached=bool(np.linalg.norm(self.env.data.qpos[:2]-self.goal)<.12 and self.distances[-1]-self.progress<.02))
        return info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info.update(self._metrics())
        # Destination accuracy becomes relevant after completing the route.
        if info['route_remaining_m']<.02:
            reward-=float(self.navigation.get('goal_error_weight',1.5))*info['payload_goal_error_m']
        return obs,float(reward),done,truncated,info

    def set_goal(self,xy):
        xy=np.asarray(xy,dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():
            raise ValueError('Goal requires two finite coordinates')
        route=plan_route(self.env.data.qpos[:2],xy,self.env.config.obstacles,margin=self.margin)
        self.goal=xy;self.route=route
        self.distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        self.segment=0;self.progress=0.;self.speed=0.
        self.last_target_time=float(self.env.data.time)
        self.route_start_time=max(4.,self.last_target_time)
        self.extra[-5:-3]=xy
        self.domain.update(goal_xy=xy.tolist(),route=route.tolist())
        self._trajectory()
