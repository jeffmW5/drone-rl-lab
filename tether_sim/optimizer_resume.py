"""Opt-in matched Adam state restoration; refuse mismatched policy weights."""
import hashlib
from pathlib import Path
import torch

def restore_optimizer(optimizer,policy,state_path):
    path=Path(state_path);state=torch.load(path,map_location='cpu',weights_only=True)
    model=path.parent/state['model_name']
    weights=torch.load(model,map_location='cpu',weights_only=True)
    current=policy.state_dict()
    if weights.keys()!=current.keys() or any(not torch.equal(weights[k],current[k].detach().cpu()) for k in weights):
        raise ValueError('Optimizer checkpoint does not match loaded policy weights')
    saved=state['optimizer_state_dict']
    if len(saved['param_groups'])!=len(optimizer.param_groups):raise ValueError('Optimizer group mismatch')
    for old,new in zip(saved['param_groups'],optimizer.param_groups):
        if len(old['params'])!=len(new['params']):raise ValueError('Optimizer parameter count mismatch')
        for key in ('lr','betas','eps','weight_decay','amsgrad'):
            if old.get(key)!=new.get(key):raise ValueError('Optimizer setting mismatch: '+key)
    optimizer.load_state_dict(saved)
    return dict(version=2,state_path=str(path),model_path=str(model),weights_exact_match=True,source_global_step=state['global_step'],optimizer_steps=sorted({float(v['step']) for v in optimizer.state.values()}),state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),scope='Adam moments and counters only; environment, RNG and PPO rollout/global counters restart')
