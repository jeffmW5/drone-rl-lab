"""Bounded paired PD controller diagnostics; never changes the live policy."""
import hashlib
import json
from pathlib import Path
import sys
import time
import types
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from tether_sim.generalize import make_task


def run(config, seed, gain, damping):
    task = make_task(config)
    task.reset(seed=seed)
    clipped = []
    def baseline(env):
        env._update_target()
        d, c = env.data, env.config
        R = d.xmat[env.drone].reshape(3,3)
        acc = 2*(env.target-d.qpos[:3])-2.8*d.qvel[:3]+[0,0,9.81]
        z = acc/max(np.linalg.norm(acc),1e-8)
        torque = R.T @ (gain*np.cross(R[:,2],z)-damping*(R@d.qvel[3:6]))
        torque[2] = -.0002*d.qvel[5]
        force = (c.drone_mass+c.payload_mass+c.cable_mass)*np.dot(acc,R[:,2])
        A = np.array([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[.006,-.006,.006,-.006]])
        command = np.linalg.solve(A,np.r_[force,torque])/c.max_motor_thrust
        clipped.append(bool(np.any(command<0) or np.any(command>1)))
        return np.clip(command,0,1).astype(np.float32)
    task.env.baseline = types.MethodType(baseline, task.env)
    rows = []
    failure = None
    done = False
    info = {}
    try:
        for _ in range(round(config['episode_seconds']/.02)):
            _, _, done, truncated, info = task.step(np.zeros(4))
            if task.env.data.time >= config['episode_seconds']-10:
                R = task.env.data.xmat[task.env.drone].reshape(3,3)
                rows.append([info['payload_goal_error_mm'],
                             task.env.data.qpos[2]-task.env.target[2],
                             np.degrees(np.arccos(np.clip(R[2,2],-1,1)))])
            if done or truncated:
                break
    except FloatingPointError as error:
        failure = str(error)
    result = dict(gain=gain,damping=damping,seed=seed,crashed=bool(done),
                  numerical_failure=failure,precision_success=info.get('precision_success',False),
                  route_remaining_m=info.get('route_remaining_m'),
                  final_error_mm=info.get('payload_goal_error_mm') if not failure else None,
                  saturation_fraction=float(np.mean(clipped)) if clipped else None,
                  tail_mean_error_mm=float(np.mean(rows,axis=0)[0]) if rows else None,
                  tail_mean_z_error_m=float(np.mean(rows,axis=0)[1]) if rows else None,
                  tail_mean_tilt_deg=float(np.mean(rows,axis=0)[2]) if rows else None,
                  domain=task.domain)
    task.close()
    return result


def main():
    plan_path = Path('configs/tether_controller_diagnostic_v001.json')
    plan = json.loads(plan_path.read_text())
    base_path = Path(plan['base_config'])
    config = json.loads(base_path.read_text())
    output = Path('tmp/tether-research/controller_sweep_v001.json')
    assert not output.exists(), 'Existing diagnostic retained; create a new version'
    result = dict(scope=plan['scope'],plan=plan,base_config_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),runs=[])
    started = time.monotonic()
    for gain in plan['attitude_gains']:
        for seed in plan['seeds']:
            if time.monotonic()-started > plan['budget_seconds']:
                result['status'] = 'budget exhausted'
                output.write_text(json.dumps(result,indent=2))
                return
            row = run(config,seed,gain,plan['angular_damping'])
            result['runs'].append(row)
            output.write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='domain'}),flush=True)
    result.update(status='complete',elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
