"""Matched read-only policy queries; no simulation or sampled actions."""
import torch

def query(policy,observations,actions):
    shape=actions.shape
    with torch.no_grad():
        dist,value=policy(observations.reshape(-1,98))
        logprob=dist.log_prob(actions.reshape(-1,4)).sum(-1)
    return {'mean_action':dist.mean.reshape(shape).detach().cpu().clone(),'logprob':logprob.reshape(shape[:-1]).detach().cpu().clone(),'value':value.reshape(shape[:-1]).detach().cpu().clone(),'log_std':policy.log_std.detach().cpu().clone()}
