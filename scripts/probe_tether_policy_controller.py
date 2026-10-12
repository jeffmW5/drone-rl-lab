"""Evaluate frozen residual weights against two PD attitude gains on CUDA."""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import types
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from stable_baselines3 import PPO
import tether_sim.train as evaluator
from tether_sim.training_lock import acquire_training_lease


def bind_controller(task, gain, damping):
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
        return np.clip(np.linalg.solve(A,np.r_[force,torque])/c.max_motor_thrust,0,1).astype(np.float32)
    task.env.baseline = types.MethodType(baseline, task.env)
    return task


def main():
    plan_path = Path(sys.argv[1] if len(sys.argv)>1 else 'configs/tether_policy_controller_diagnostic_v001.json')
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan['base_config'])
    config = json.loads(config_path.read_text())
    source = Path(plan['source_run'])
    checkpoint = plan['source_checkpoint']
    policy_file = source/(checkpoint+'.zip')
    norm_file = source/(checkpoint+'_normalize.pkl')
    output = Path('tmp/tether-research/'+plan['name'].replace('tether_policy_controller_diagnostic','policy_controller_probe')+'.json')
    assert not output.exists(), 'Preserve existing diagnostic; create a new version'
    assert torch.cuda.is_available(), 'CUDA evaluation required'
    torch.set_num_threads(1)
    lease = acquire_training_lease()
    try:
        model = PPO.load(policy_file, device='cuda')
        with norm_file.open('rb') as file:
            normalizer = pickle.load(file)
        normalizer.training = False
        result = dict(status='running',plan=plan,
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      device=str(model.device),gpu=torch.cuda.get_device_name(0),
                      policy_sha256=hashlib.sha256(policy_file.read_bytes()).hexdigest(),
                      normalizer_sha256=hashlib.sha256(norm_file.read_bytes()).hexdigest(),
                      config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),results={})
        started = time.monotonic()
        original_factory = evaluator.make_task
        for gain in plan['attitude_gains']:
            if time.monotonic()-started > plan['budget_seconds']:
                result['status'] = 'budget exhausted'
                break
            def factory(cfg):
                task = original_factory(cfg)
                original_reset = task.reset
                def reset(*args, **kwargs):
                    observation = original_reset(*args, **kwargs)
                    bind_controller(task,gain,plan['angular_damping'])
                    return observation
                task.reset = reset
                return task
            evaluator.make_task = factory
            values = evaluator.evaluate(model,normalizer,config,seeds=plan['seeds'])
            result['results'][str(gain)] = values
            output.write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(gain=gain,scores={k:dict(
                crashes=v['crashes'],successes=v['successes'],
                route_completions=sum(r['route_remaining_m']<=1e-6 for r in v['runs']),
                mean_final_error_mm=float(np.mean([r['final_payload_goal_error_mm'] for r in v['runs']])))
                for k,v in values.items()})),flush=True)
        else:
            result['status'] = 'complete'
        evaluator.make_task = original_factory
        result['elapsed_seconds'] = time.monotonic()-started
        assert hashlib.sha256(policy_file.read_bytes()).hexdigest()==result['policy_sha256']
        assert hashlib.sha256(norm_file.read_bytes()).hexdigest()==result['normalizer_sha256']
        output.write_text(json.dumps(result,indent=2))
    finally:
        lease.close()


if __name__ == '__main__':
    main()
