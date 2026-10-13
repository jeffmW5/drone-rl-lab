"""Fixed-policy uninterrupted zone diagnostic; never a benchmark replacement."""
import copy
import torch
from .cuda_task import CudaVector

def probe(policy,plan):
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();result={'version':2,'scope':'Still-air full-tether fixed policies, up to20s with time limit disabled only here; stop first boundary/physics failure, no resets counted as recovery','policies':[]}
    try:
        for checkpoint in plan['zone_horizon_checkpoints']:
            model=copy.deepcopy(policy);weights=torch.load(checkpoint,weights_only=True);weights.setdefault('obs_scale',model.obs_scale);model.load_state_dict(weights)
            vec=CudaVector(1,direct=True,reward_version=14,control_version=2,physics_version=5,phase='takeoff',training_reset_version=3,diagnostic_zone_version=1)
            vec.async_reset(51011);frames=[];hold=0.;peak_hold=0.;tookoff=False;landed=None;terminal=[]
            def observe(obs,done,truncated,failed,reward):
                terminal.append(dict(time_s=(step+1)*.02,observation=obs[0].cpu().tolist(),failed=int(failed[0]),truncated=bool(truncated[0]),reward=float(reward[0])))
            vec.terminal_observer=observe
            for step in range(1000):
                vec.steps.zero_() # Disable500step timeout only in diagnostic.
                with torch.no_grad():
                    obs=vec.obs();dist,_=model(obs)
                    neutral=obs.clone()
                    neutral[:,:42].reshape(1,14,3)[:,:,:2]-=obs[:,:42].reshape(1,14,3)[:,:1,:2]
                    neutral[:,42:84].reshape(1,14,3)[:,:,:2]-=obs[:,42:84].reshape(1,14,3)[:,:1,:2]
                    neutral_dist,_=model(neutral)
                    orientation=vec.sim.orientation[0].cpu().tolist()
                    neutral_action=neutral_dist.mean[0].cpu().tolist()
                vec.send(dist.mean.cpu().numpy())
                if terminal:break # Terminal observer captures pre-reset state.
                p=vec.sim.position[0,0];v=vec.sim.velocity[0,0];speed=float(v.norm());extent=float(p[:2].abs().max())
                tookoff=tookoff or float(p[2])>.3
                if tookoff and float(p[2])<=.051 and landed is None:landed=(step+1)*.02
                stable=extent<=1 and abs(float(p[2])-.8)<.1 and speed<.2
                hold=hold+.02 if stable else 0.;peak_hold=max(peak_hold,hold)
                if step%5==0 or step in (249,499,999):frames.append(dict(time_s=(step+1)*.02,carrier=p.cpu().tolist(),velocity=v.cpu().tolist(),payload=vec.sim.position[0,-1].cpu().tolist(),speed_m_s=speed,hold_s=hold,mean_action=dist.mean[0].cpu().tolist(),orientation=orientation,neutral_horizontal_mean_action=neutral_action))
            result['policies'].append(dict(checkpoint=checkpoint,elapsed_simulated_s=(step+1)*.02,tookoff=tookoff,first_landing_s=landed,peak_zone_hover_hold_s=peak_hold,zone_hover_success=peak_hold>=1,terminal=terminal,frames=frames))
    finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
    return result
