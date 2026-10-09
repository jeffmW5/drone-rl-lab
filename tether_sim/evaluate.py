"""Repeatable baseline evaluation and optional vision demonstration dataset."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import mujoco
from .env import TetherEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='tmp/tether-sim/evaluation.json')
    p.add_argument('--dataset',help='Optional .npz demonstration: 64x64 images, state, actions')
    args=p.parse_args()
    env=TetherEnv()
    result={'engine':'MuJoCo','version':mujoco.__version__,'config':asdict(env.config),'controller':'privileged PD','scenarios':{},'limitations':['uncalibrated motors/inertia/cable','rigid segmented cable can transmit compression','no vision policy','CPU physics; GPU training not benchmarked']}
    frames,states,actions=[],[],[]
    for scenario in ('hover','swing','lift','carry','lift_carry'):
        obs,_=env.reset(seed=7,options={'scenario':scenario})
        errors,angles=[],[]
        for i in range(600):
            action=env.baseline()
            if args.dataset and scenario=='lift_carry' and i%5==0:
                frames.append(env.camera_frame(64)); states.append(obs); actions.append(action)
            obs,_,terminated,_,info=env.step(action)
            if terminated:
                raise RuntimeError(f'{scenario}: baseline crashed')
            errors.append(float(np.linalg.norm(env.data.qpos[:3]-env.target)))
            angles.append(info['swing_deg'])
        result['scenarios'][scenario]={'final_error_m':errors[-1],'rms_error_m':float(np.sqrt(np.mean(np.square(errors)))),'peak_swing_deg':max(angles),'final':info}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    if args.dataset:
        path=Path(args.dataset); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,images=np.asarray(frames),states=np.asarray(states),actions=np.asarray(actions))
    env.close()
    print(json.dumps({k:v['final_error_m'] for k,v in result['scenarios'].items()},indent=2))


if __name__=='__main__':
    main()
