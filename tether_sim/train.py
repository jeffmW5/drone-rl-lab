"""Bounded residual PPO training with paired seeded baseline evaluations."""
import argparse
from dataclasses import asdict
import functools
import hashlib
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
            success=bool(not terminated and errors[-1]<.08 and swings[-1]<10)
            if config.get('task')=='random_navigation_v2':
                success=bool(not terminated and info.get('goal_reached',False) and info['payload_goal_error_m']<.08 and swings[-1]<10)
            runs.append(dict(goal_distance_m=info.get('goal_distance_m'),payload_goal_error_m=info.get('payload_goal_error_m'),route_remaining_m=info.get('route_remaining_m'),seed=seed,return_=total,crashed=terminated,steps=len(errors),payload_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),settled_rmse_m=float(np.sqrt(np.mean(np.square(settled)))) if settled else None,final_payload_error_m=errors[-1],peak_swing_deg=max(swings),obstacle_collision=info.get('obstacle_collision',False),minimum_clearance_m=min(clearances) if np.isfinite(min(clearances)) else None,success=success))
            env.close()
        results[label]=dict(runs=runs,mean_payload_rmse_m=float(np.mean([r['payload_rmse_m'] for r in runs])),crashes=sum(r['crashed'] for r in runs),successes=sum(r['success'] for r in runs),mean_goal_distance_m=float(np.mean([r['goal_distance_m'] for r in runs])) if runs and runs[0]['goal_distance_m'] is not None else None)
    return results


class Progress(BaseCallback):
    def __init__(self,config,out):
        super().__init__()
        self.config,self.out=config,out
        self.started=time.monotonic()
        self.next_save=config['checkpoint_every_steps']
        self.next_eval=config['eval_every_steps']
        self.best=float('inf')
        self.best_rank=None

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
            rank=(result['PPO_residual']['crashes'],-result['PPO_residual']['successes'],score) if self.config.get('task')=='random_navigation_v2' else (score,)
            eligible=result['PPO_residual']['crashes']==0 if self.config.get('task')!='random_navigation_v2' else True
            if eligible and (self.best_rank is None or rank<self.best_rank):
                self.best_rank=rank
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
    (out/'status.json').write_text(json.dumps(dict(status='initializing',timesteps=0,updated_at=time.time()))+'\n')
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
    raw_env=env
    env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.)
    model=PPO('MlpPolicy',env,device=config['device'],seed=config['seed'],n_steps=config['n_steps'],batch_size=config['batch_size'],n_epochs=config['n_epochs'],learning_rate=config['learning_rate'],gamma=config['gamma'],policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256]),log_std_init=config.get('log_std_init',-2),ortho_init=True),target_kl=.03,verbose=1)
    if config.get('resume_run'):
        source=Path(config['resume_run']);checkpoint=config.get('resume_checkpoint','best')
        if checkpoint not in ('best','final','interrupted'):
            raise ValueError('Resume checkpoint must be best, final or interrupted')
        model_file=source/(checkpoint+'.zip');norm_file=source/(checkpoint+'_normalize.pkl')
        if not model_file.is_file() or not norm_file.is_file():
            raise ValueError('Resume requires policy ZIP and matching normalizer PKL')
        env=VecNormalize.load(norm_file,raw_env)
        env.training=True;env.norm_reward=True
        model=PPO.load(model_file,env=env,device=config['device'],seed=config['seed'],
                       learning_rate=config['learning_rate'],n_steps=config['n_steps'],
                       batch_size=config['batch_size'],n_epochs=config['n_epochs'],gamma=config['gamma'])
        provenance['transfer']={'run':str(source),'checkpoint':checkpoint,'source_timesteps':model.num_timesteps,
                                'policy_sha256':hashlib.sha256(model_file.read_bytes()).hexdigest(),
                                'normalizer_sha256':hashlib.sha256(norm_file.read_bytes()).hexdigest(),
                                'mode':'loaded policy, optimizer and normalization; additional timestep counter resets'}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print('TRANSFER '+json.dumps(provenance['transfer']),flush=True)
        (out/'status.json').write_text(json.dumps(dict(status='evaluating initial transfer',timesteps=0,updated_at=time.time()))+'\n')
        callback_initial=evaluate(model,env,config)
        (out/'evaluation_initial.json').write_text(json.dumps(callback_initial,indent=2)+'\n')
        model.save(out/'initial');env.save(out/'initial_normalize.pkl')
    if config.get('zero_residual_init') and not config.get('resume_run'):
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
    model.set_logger(configure(str(out),['stdout','csv']))
    callback=Progress(config,out)
    try:
        model.learn(total_timesteps=config['total_timesteps'],callback=callback)
        callback.save_pair('final')
        if not (out/'best.zip').exists():
            callback.save_pair('best')
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
