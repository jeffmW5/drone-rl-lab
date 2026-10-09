"""Bounded residual PPO training with paired seeded baseline evaluations."""
import argparse
from dataclasses import asdict
import functools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import mujoco
import gymnasium
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv,VecNormalize
from stable_baselines3.common.logger import configure
from .task import LiftCarryTask
from .generalize import make_task


def factory(config):
    return Monitor(make_task(config))


def evaluate(model,normalizer,config,seeds=None):
    results={}
    for label in ('PD','PPO_residual'):
        runs=[]
        for seed in seeds or config['eval_seeds']:
            env=make_task(config)
            obs,_=env.reset(seed=seed)
            errors,swings,settled,clearances=[],[],[],[]
            total=0.
            for _ in range(round(config['episode_seconds']/.02)):
                action=np.zeros(4,dtype=np.float32)
                if label=='PPO_residual':
                    action,_=model.predict(normalizer.normalize_obs(obs.copy()),deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                errors.append(info['payload_error_m']);swings.append(info['swing_deg']);total+=reward
                if env.env.data.time>=10:
                    settled.append(info['payload_error_m'])
                clearances.append(info.get('obstacle_clearance_m',float('inf')))
                if terminated or truncated:
                    break
            runs.append(dict(seed=seed,return_=total,crashed=terminated,steps=len(errors),payload_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),settled_rmse_m=float(np.sqrt(np.mean(np.square(settled)))) if settled else None,final_payload_error_m=errors[-1],peak_swing_deg=max(swings),obstacle_collision=info.get('obstacle_collision',False),minimum_clearance_m=min(clearances) if np.isfinite(min(clearances)) else None,success=bool(not terminated and errors[-1]<.08 and swings[-1]<10)))
            env.close()
        results[label]=dict(runs=runs,mean_payload_rmse_m=float(np.mean([r['payload_rmse_m'] for r in runs])),crashes=sum(r['crashed'] for r in runs))
    return results


class Progress(BaseCallback):
    def __init__(self,config,out):
        super().__init__()
        self.config,self.out=config,out
        self.started=time.monotonic()
        self.next_save=config['checkpoint_every_steps']
        self.next_eval=config['eval_every_steps']
        self.best=float('inf')

    def save_pair(self,name):
        self.model.save(self.out/name)
        self.training_env.save(self.out/(name+'_normalize.pkl'))

    def _on_step(self):
        elapsed=time.monotonic()-self.started
        if self.num_timesteps>=self.next_save:
            self.save_pair(f'checkpoint_{self.num_timesteps}')
            self.next_save+=self.config['checkpoint_every_steps']
        if self.num_timesteps>=self.next_eval:
            result=evaluate(self.model,self.training_env,self.config)
            result.update(timesteps=self.num_timesteps,elapsed_seconds=elapsed)
            (self.out/f'eval_{self.num_timesteps}.json').write_text(json.dumps(result,indent=2)+'\n')
            score=result['PPO_residual']['mean_payload_rmse_m']
            if result['PPO_residual']['crashes']==0 and score<self.best:
                self.best=score
                self.save_pair('best')
            print('EVAL '+json.dumps(result),flush=True)
            self.next_eval+=self.config['eval_every_steps']
        if self.n_calls%128==0:
            state=dict(status='training',timesteps=self.num_timesteps,elapsed_seconds=elapsed,steps_per_second=self.num_timesteps/max(elapsed,1),device=str(self.model.device),gpu=torch.cuda.get_device_name(0) if self.model.device.type=='cuda' else None,updated_at=time.time())
            temporary=self.out/'status.tmp'
            temporary.write_text(json.dumps(state,indent=2)+'\n')
            temporary.replace(self.out/'status.json')
        return elapsed<self.config['budget_seconds']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/tether_v002.json')
    p.add_argument('--output',default='tmp/tether-runs/tether_v002')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.smoke:
        config=dict(config,total_timesteps=2048,n_envs=2,n_steps=128,n_epochs=2,batch_size=128,eval_every_steps=2048,checkpoint_every_steps=1024)
        config.update(eval_seeds=[41001],test_seeds=[42001])
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.set_num_threads(1)
    if config['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required by frozen config')
    # Exercise a real CUDA matrix multiply, not just device enumeration.
    tensor=torch.randn(512,512,device=config['device'])
    _=(tensor@tensor).sum().item()
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),python=platform.python_version(),torch=torch.__version__,mujoco=mujoco.__version__,gymnasium=gymnasium.__version__,sb3=sb3.__version__,gpu=torch.cuda.get_device_name(0) if config['device']=='cuda' else None,pid=os.getpid(),model='residual motor PPO with privileged PD',limitations=config['notes'])
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)
    factories=[functools.partial(factory,config) for _ in range(config['n_envs'])]
    env=DummyVecEnv(factories) if config.get('vec_backend')=='dummy' else SubprocVecEnv(factories,start_method='spawn')
    env.seed(config['seed'])
    env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.)
    model=PPO('MlpPolicy',env,device=config['device'],seed=config['seed'],n_steps=config['n_steps'],batch_size=config['batch_size'],n_epochs=config['n_epochs'],learning_rate=config['learning_rate'],gamma=config['gamma'],policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),log_std_init=config.get('log_std_init',-2),ortho_init=True),target_kl=.03,verbose=1)
    if config.get('zero_residual_init'):
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
    model.set_logger(configure(str(out),['stdout','csv']))
    callback=Progress(config,out)
    try:
        model.learn(total_timesteps=config['total_timesteps'],callback=callback)
        callback.save_pair('final')
        result=evaluate(model,env,config,seeds=config.get('test_seeds'))
        (out/'evaluation_final.json').write_text(json.dumps(result,indent=2)+'\n')
        state=dict(status='complete',timesteps=model.num_timesteps,elapsed_seconds=time.monotonic()-callback.started,evaluation=result)
        (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    except BaseException as error:
        callback.save_pair('interrupted')
        (out/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error),timesteps=model.num_timesteps),indent=2)+'\n')
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
