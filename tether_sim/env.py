from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from .visuals import visual_assets, visual_drone


@dataclass
class Config:
    drone_mass: float = .0434
    payload_mass: float = .005
    cable_mass: float = .001
    length: float = .6
    segments: int = 12
    timestep: float = .002
    control_dt: float = .02
    motor_tau: float = .025
    max_motor_thrust: float = .20
    duration: float = 20.
    obstacles: tuple = ()


def model_xml(c):
    if not (2 <= c.segments <= 64 and c.length > 0 and min(c.drone_mass, c.payload_mass, c.cable_mass) > 0):
        raise ValueError('Positive masses/length and 2..64 cable segments required')
    ds = c.length / c.segments
    rope = ''
    for i in range(c.segments):
        pos = '0 0 -.02' if i == 0 else f'0 0 {-ds}'
        rope += f'<body name="link{i}" pos="{pos}"><joint type="ball" damping="0.000002" armature="0.0000001"/><geom type="capsule" fromto="0 0 0 0 0 {-ds}" size=".0015" mass="{c.cable_mass/c.segments}" rgba=".95 .65 .15 1"/>'
    rope += f'<body name="payload" pos="0 0 {-ds}"><geom type="sphere" size=".025" mass="{c.payload_mass}" rgba=".95 .25 .15 1"/><site name="payload_site"/></body>'
    rope += '</body>' * c.segments
    motors = [( .032, .032), (-.032, .032), (-.032,-.032), (.032,-.032)]
    sites = ''.join(f'<site name="motor{i}" pos="{x} {y} 0" size=".018" type="cylinder" rgba=".2 .7 .9 0"/>' for i,(x,y) in enumerate(motors))
    actuators = ''.join(f'<general name="motor{i}" site="motor{i}" gear="0 0 1 0 0 {(.006 if i%2==0 else -.006)}" ctrllimited="true" ctrlrange="0 {c.max_motor_thrust}"/>' for i in range(4))
    obstacles = ''.join(f'<geom name="obstacle{i}" type="box" pos="{o[0]} {o[1]} {o[2]}" size="{o[3]} {o[4]} {o[5]}" euler="0 0 {o[6]}" rgba=".32 .48 .66 1"/>' for i,o in enumerate(c.obstacles))
    return f'''<mujoco model="Crazyflie tether lab"><compiler angle="radian"/><option timestep="{c.timestep}" integrator="implicitfast" gravity="0 0 -9.81" iterations="80" tolerance="1e-10"/><visual><quality shadowsize="4096" offsamples="4"/><rgba haze=".12 .16 .22 1"/><map zfar="30"/><global offwidth="1280" offheight="720"/><headlight ambient=".4 .4 .4"/></visual><default><geom friction=".7 .01 .001" solref=".01 1"/></default><asset>{visual_assets()}<texture type="skybox" builtin="gradient" rgb1=".12 .17 .24" rgb2=".025 .035 .055" width="512" height="3072"/><texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".12 .16 .2" rgb2=".18 .23 .28"/><material name="floor" texture="grid" texrepeat="12 12"/></asset><worldbody><light pos="1 -1 4" diffuse=".9 .9 .9"/><light pos="-2 1 2" diffuse=".35 .45 .6" castshadow="false"/><geom name="ground" type="plane" size="5 5 .1" material="floor"/><site name="target" type="cylinder" pos="0 0 .003" size=".04 .001" rgba=".2 1 .4 .6"/>{obstacles}<body name="drone" pos="0 0 1.2"><freejoint/><inertial pos="0 0 0" mass="{c.drone_mass}" diaginertia=".000025 .000025 .00004"/><geom type="box" size=".035 .035 .012" mass="0" rgba=".2 .65 .85 0" group="3"/>{visual_drone(motors)}{sites}<camera name="onboard" pos=".02 0 -.015" xyaxes="0 -1 0 1 0 0" fovy="90"/>{rope}</body></worldbody><actuator>{actuators}</actuator></mujoco>'''


class TetherEnv(gym.Env):
    """Four normalized motor commands; observations are privileged state, not vision."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, config=None, render_mode=None):
        self.config = c = config or Config()
        if c.timestep <= 0 or c.control_dt < c.timestep or c.motor_tau <= 0:
            raise ValueError('Invalid integration/motor time constants')
        self.model = mujoco.MjModel.from_xml_string(model_xml(c))
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(0., 1., (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.model.nq+self.model.nv+11,), np.float64)
        self.render_mode = render_mode
        self.renderer = None
        self.camera_renderer = None
        self.motors = np.zeros(4)
        self.target = np.array([0., 0., 1.2])
        self.wind = np.zeros(3)
        self.drone = self.model.body('drone').id
        self.payload = self.model.body('payload').id
        self.substeps = round(c.control_dt/c.timestep)
        self.reset()

    def _obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.motors, self.target, self.wind, [self.data.time]]).copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.motors[:] = 0
        self.wind[:] = 0
        self.target[:] = [0, 0, 1.2]
        scenario = (options or {}).get('scenario', 'hover')
        self.scenario = scenario
        if scenario == 'swing':
            a = .35
            self.data.qpos[7:11] = [np.cos(a/2), 0, np.sin(a/2), 0]
        elif scenario in ('lift', 'lift_carry'):
            self.data.qpos[2] = self.config.length + .048
        elif scenario == 'carry':
            self.target[0] = .6
        elif scenario != 'hover':
            raise ValueError('Unknown scenario')
        self._update_target()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self.metrics()

    def baseline(self):
        """Privileged-state PD baseline; not a learned/vision controller."""
        d, c = self.data, self.config
        self._update_target()
        R = d.xmat[self.drone].reshape(3,3)
        total = c.drone_mass+c.payload_mass+c.cable_mass
        acc = 2*(self.target-d.qpos[:3])-2.8*d.qvel[:3]+[0,0,9.81]
        desired_z = acc / max(np.linalg.norm(acc), 1e-8)
        attitude_error = np.cross(R[:,2], desired_z)
        omega_world = R @ d.qvel[3:6]
        torque = R.T @ (.015*attitude_error-.0012*omega_world)
        torque[2] = -.0002*d.qvel[5]
        force = total*np.dot(acc, R[:,2])
        allocation = np.array([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[.006,-.006,.006,-.006]])
        return np.clip(np.linalg.solve(allocation, np.r_[force,torque])/c.max_motor_thrust, 0, 1).astype(np.float32)

    def _update_target(self):
        if self.scenario == 'lift_carry':
            fraction = np.clip((self.data.time-3)/4, 0, 1)
            self.target[0] = .6*(3*fraction**2-2*fraction**3)
        self.model.site('target').pos[:] = [self.target[0],self.target[1],.003]

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all():
            raise ValueError('Action must contain four finite motor commands')
        command = np.clip(a,0,1)*self.config.max_motor_thrust
        for _ in range(self.substeps):
            self.motors += (1-np.exp(-self.config.timestep/self.config.motor_tau))*(command-self.motors)
            self.data.ctrl[:] = self.motors
            self.data.xfrc_applied[:] = 0
            self.data.xfrc_applied[self.drone,:3] = self.wind-.015*self.data.qvel[:3]
            mujoco.mj_step(self.model,self.data)
            if any(self.data.warning[i].number for i in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                raise FloatingPointError('MuJoCo numerical instability; rollout is invalid')
        self._update_target()
        mujoco.mj_forward(self.model, self.data)
        distance = np.linalg.norm(self.target-self.data.qpos[:3])
        terminated = bool(self.data.qpos[2]<.03 or not np.isfinite(self.data.qpos).all())
        truncated = bool(self.data.time >= self.config.duration)
        return self._obs(), float(-distance-.01*np.square(a).sum()), terminated, truncated, self.metrics()

    def metrics(self):
        delta = self.data.xpos[self.payload]-self.data.xpos[self.drone]
        mujoco.mj_rnePostConstraint(self.model,self.data)
        first = self.model.body('link0').id
        axis = -self.data.xmat[first].reshape(3,3)[:,2]
        axial_load = -float(np.dot(self.data.cfrc_int[first,3:],axis))
        return dict(time=float(self.data.time), drone=self.data.xpos[self.drone].tolist(), payload=self.data.xpos[self.payload].tolist(), target=self.target.tolist(), endpoint_distance=float(np.linalg.norm(delta)), cable_length=self.config.length, attachment_axial_load_N=axial_load, swing_deg=float(np.degrees(np.arctan2(np.linalg.norm(delta[:2]),-delta[2]))), contacts=int(self.data.ncon), motor_thrust_N=self.motors.tolist())

    def camera_frame(self, size=324):
        """Synthetic pinhole grayscale; 324x244 or 64x64. Not calibrated HM01B0."""
        from PIL import Image
        if self.camera_renderer is None:
            self.camera_renderer = mujoco.Renderer(self.model,height=244,width=324)
        self.camera_renderer.update_scene(self.data,camera='onboard')
        gray = Image.fromarray(self.camera_renderer.render()).convert('L')
        return np.asarray(gray if size == 324 else gray.resize((size,size))).copy()

    def render(self, camera=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=540,width=960)
        self.renderer.update_scene(self.data, camera=camera if camera is not None else -1)
        return self.renderer.render().copy()

    def close(self):
        if self.camera_renderer:
            self.camera_renderer.close()
            self.camera_renderer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
