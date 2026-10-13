"""Diagnostic recovery from signed perturbations and captured visited states.
No helper control, action sampling, learning or benchmark replacement.
Caller holds the existing serial GPU lease.
"""
import copy,hashlib,math
from pathlib import Path
import torch
from .cuda_task import CudaVector

def probe(policy,plan):
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    path=Path(plan['recovery_rollout']);saved=torch.load(path,map_location='cpu',weights_only=True)
    obs=saved['observations'].reshape(-1,98);h=obs[:,2]+obs[:,97];speed=obs[:,42:45].norm(dim=1)
    eligible=(h-.8).abs()<.1
    picks=[]
    for label,mask in (('visited_slow',eligible&(speed<.2)),('visited_fast',eligible&(speed>1))):
        ids=mask.nonzero().flatten();ids=ids[saved['terminals'].flatten()[ids]==0]
        if ids.numel():picks.append((label,int(ids[len(ids)//2])))
    names=['ground','ideal_hover','roll_plus10','roll_minus10','pitch_plus10','pitch_minus10','vx_plus','vx_minus','wx_plus','wx_minus']+[a for a,b in picks]
    result={'version':1,'scope':'Recovery diagnostic from artificial and actually visited states; no reset counted as recovery; standard benchmark unchanged','rollout':str(path),'rollout_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'visited_indices':dict(picks),'models':[]}
    try:
        for item in plan['recovery_models']:
            model=copy.deepcopy(policy);model.configure_collective_observation(item['collective_observation_version']);weights=torch.load(item['checkpoint'],weights_only=True);weights.setdefault('obs_scale',model.obs_scale);model.load_state_dict(weights)
            vec=CudaVector(len(names),direct=True,reward_version=15,phase='takeoff',control_version=2,physics_version=5,training_reset_version=3,diagnostic_zone_version=1);vec.async_reset(51011);s=vec.sim
            s.position[:,:,0:2]=0;s.position[:,:,2]=.8-torch.arange(14,device='cuda').double()*.6/13;s.velocity.zero_();s.orientation.zero_();s.orientation[:,0]=1;s.motors[:]=s.mass.sum(1,keepdim=True)*9.81/4
            s.position[0]=vec.start[0];s.motors[0]=0
            for row,axis,sign in ((2,1,1),(3,1,-1),(4,2,1),(5,2,-1)):
                s.orientation[row,0]=math.cos(math.radians(5));s.orientation[row,axis]=sign*math.sin(math.radians(5))
            s.velocity[6,:,0]=.5;s.velocity[7,:,0]=-.5;s.orientation[8,4]=1;s.orientation[9,4]=-1
            for row,(label,index) in enumerate(picks,10):
                o=obs[index].to('cuda');vec.goal[row]=o[95:98];s.position[row]=o[:42].reshape(14,3).double()+o[95:98].double();s.velocity[row]=o[42:84].reshape(14,3).double();s.orientation[row]=o[84:91];s.motors[row]=o[91:95]
            active=torch.ones(len(names),device='cuda',dtype=torch.bool);hold=torch.zeros(len(names),device='cuda');peak=hold.clone();landing=[None]*len(names);failures=[None]*len(names);frames=[]
            tookoff=s.position[:,0,2]>.3
            def terminal(o,done,truncated,failed,reward):
                ids=(done&active).nonzero().flatten().cpu().tolist()
                for i in ids:failures[i]={'time_s':(step+1)*.02,'failed':int(failed[i]),'observation':o[i].cpu().tolist()}
                active[done]=False
            vec.terminal_observer=terminal
            for step in range(1000):
                vec.steps.zero_()
                with torch.no_grad():dist,_=model(vec.obs());mean=dist.mean
                vec.send(mean.cpu().numpy());p=s.position[:,0];v=s.velocity[:,0];extent=p[:,:2].abs().amax(1);speed_now=v.norm(dim=1)
                tookoff|=(p[:,2]>.3)&active
                for i in ((p[:,2]<=.051)&tookoff&active).nonzero().flatten().cpu().tolist():
                    if landing[i] is None:landing[i]=(step+1)*.02
                stable=active&(extent<=1)&((p[:,2]-.8).abs()<.1)&(speed_now<.2)
                hold=torch.where(stable,hold+.02,0);peak=torch.maximum(peak,hold)
                if step%10==0:frames.append({'time_s':(step+1)*.02,'active':active.cpu().tolist(),'carrier':p.cpu().tolist(),'velocity':v.cpu().tolist(),'orientation':s.orientation.cpu().tolist(),'mean_action':mean.cpu().tolist()})
                if not bool(active.any()):break
            result['models'].append({'checkpoint':item['checkpoint'],'collective_observation_version':item['collective_observation_version'],'cases':[{'name':names[i],'peak_hold_s':float(peak[i]),'hover_success':bool(peak[i]>=1),'first_landing_s':landing[i],'first_failure':failures[i],'survived_20s':bool(active[i])} for i in range(len(names))],'frames':frames})
    finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
    return result
