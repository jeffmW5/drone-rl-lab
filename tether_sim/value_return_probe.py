"""Version1 realized stochastic-policy return probe; no learning changes."""
import math

def completed_returns(rewards, dones, gamma):
    result=[None]*len(rewards);tail=0.;complete=False
    for i in range(len(rewards)-1,-1,-1):
        if dones[i]:tail=0.;complete=True
        if complete:
            tail=rewards[i]+gamma*tail;result[i]=tail
    return result

def summary(pairs):
    if not pairs:return dict(count=0,rmse=None,explained_variance=None)
    values,returns=zip(*pairs);n=len(pairs)
    errors=[v-r for v,r in pairs];mean=sum(returns)/n;mean_error=sum(errors)/n
    variance=sum((r-mean)**2 for r in returns)/n
    error_variance=sum((e-mean_error)**2 for e in errors)/n
    return dict(count=n,mean_prediction=sum(values)/n,mean_return=mean,
        mean_error=mean_error,mae=sum(abs(e) for e in errors)/n,
        rmse=math.sqrt(sum(e*e for e in errors)/n),
        explained_variance=1-error_variance/variance if variance>1e-12 else None)

def probe(policy, plan, seeds, gamma):
    import torch
    from .cuda_task import CudaVector
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();runs=[]
    try:
        for seed in seeds:
            torch.manual_seed(seed+1000000)
            vec=CudaVector(8,direct=True,reward_version=plan.get('reward_version',7),
                phase=plan.get('phase','takeoff'),control_version=plan.get('control_version',2),
                physics_version=plan.get('physics_version',1));vec.async_reset(seed)
            values=[];rewards=[];dones=[]
            for step in range(500):
                with torch.no_grad():
                    distribution,value=policy(vec.obs())
                    action=distribution.sample().clamp(-1,1).cpu().numpy()
                vec.send(action)
                values.append(value.flatten().cpu().tolist())
                rewards.append(vec.reward.clamp(-1,1).cpu().tolist())
                dones.append(vec.done.cpu().tolist())
            pairs=[];returns=[]
            for agent in range(8):
                realized=completed_returns([r[agent] for r in rewards],[d[agent] for d in dones],gamma)
                returns.append(realized)
                pairs.extend((values[t][agent],r) for t,r in enumerate(realized) if r is not None)
            runs.append(dict(seed=seed,summary=summary(pairs),values=values,
                clipped_rewards=rewards,dones=dones,completed_returns_by_agent=returns))
            vec.close()
    finally:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
    return dict(version=1,gamma=gamma,steps_per_domain=500,runs=runs,
        scope='Single realized stochastic-policy trajectory per randomized domain; censored incomplete tails excluded. Correlated samples, not expected-value calibration or causal proof. Reward7, no return adaptations.')
