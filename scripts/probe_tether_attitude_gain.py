"""Single-seed controller intervention, no training or promotion."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import types
import numpy as np
from tether_sim.generalize import make_task

config = json.loads(Path('configs/tether_research_recovery_v001.json').read_text())
results = []
for gain in (.015, .006):
    task = make_task(config)
    task.reset(seed=51001)
    def baseline(env):
        env._update_target()
        d, c = env.data, env.config
        R = d.xmat[env.drone].reshape(3,3)
        acc = 2*(env.target-d.qpos[:3])-2.8*d.qvel[:3]+[0,0,9.81]
        z = acc/max(np.linalg.norm(acc),1e-8)
        torque = R.T @ (gain*np.cross(R[:,2],z)-.0012*(R@d.qvel[3:6]))
        torque[2] = -.0002*d.qvel[5]
        force = (c.drone_mass+c.payload_mass+c.cable_mass)*np.dot(acc,R[:,2])
        A = np.array([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[.006,-.006,.006,-.006]])
        return np.clip(np.linalg.solve(A,np.r_[force,torque])/c.max_motor_thrust,0,1).astype(np.float32)
    task.env.baseline = types.MethodType(baseline, task.env)
    rows = []
    for _ in range(4500):
        _, _, done, truncated, info = task.step(np.zeros(4))
        if task.env.data.time >= 80:
            R = task.env.data.xmat[task.env.drone].reshape(3,3)
            rows.append([info['payload_goal_error_mm'],
                         task.env.data.qpos[2]-task.env.target[2],
                         np.degrees(np.arccos(np.clip(R[2,2],-1,1)))])
        if done or truncated:
            break
    results.append(dict(gain=gain, seed=51001, crashed=bool(done),
                        route_remaining_m=info['route_remaining_m'],
                        final_error_mm=info['payload_goal_error_mm'],
                        tail_mean_error_mm=float(np.mean(rows,axis=0)[0]) if rows else None,
                        tail_mean_z_error_m=float(np.mean(rows,axis=0)[1]) if rows else None,
                        tail_mean_tilt_deg=float(np.mean(rows,axis=0)[2]) if rows else None))
    task.close()
    print(json.dumps(results[-1]),flush=True)
Path('tmp/tether-research/attitude_gain_probe.json').write_text(json.dumps(
    dict(scope='single-seed 1ms PD gain intervention; not a trained-policy benchmark',results=results),indent=2))
