"""Inference using the training action/observation contract and saved statistics."""
import json
from pathlib import Path
import numpy as np
from .task import LiftCarryTask


class PolicyPlayback:
    def __init__(self,run,checkpoint='best',seed=10001):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
        torch.set_num_threads(1)
        self.run=Path(run)
        self.config=json.loads((self.run/'config.json').read_text())
        self.task=LiftCarryTask(self.config['residual_scale'],self.config['episode_seconds'])
        self.env=self.task.env
        self.seed=seed
        self.models={}
        self.normalizers={}
        for name in ('best','final'):
            self.models[name]=PPO.load(self.run/(name+'.zip'),device='cpu')
            normalizer=VecNormalize.load(self.run/(name+'_normalize.pkl'),DummyVecEnv([lambda:self.task]))
            normalizer.training=False
            normalizer.norm_reward=False
            self.normalizers[name]=normalizer
        self.controller=checkpoint
        self.inferences=0
        self.reset()

    def reset(self):
        self.obs,_=self.task.reset(seed=self.seed)
        self.last_action=np.zeros(4)

    def step(self):
        action=np.zeros(4,dtype=np.float32)
        if self.controller!='PD':
            action,_=self.models[self.controller].predict(self.normalizers[self.controller].normalize_obs(self.obs.copy()),deterministic=True)
            self.inferences+=1
        self.last_action=np.asarray(action).copy()
        result=self.task.step(action)
        self.obs=result[0]
        return result
