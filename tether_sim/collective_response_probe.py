"""Offline deterministic collective perturbations; no policy updates."""
def probe(policy,plan):
    import torch
    from .cuda_task import CudaVector
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();rows=[]
    try:
        for seed in plan['validation_seeds']:
            for shift in [-.04,0.,.04]:
                vec=CudaVector(8,direct=True,reward_version=7,phase='takeoff',control_version=2)
                vec.async_reset(seed);commands=[];crashes=0
                for _ in range(250):
                    with torch.no_grad():distribution,_=policy(vec.obs());action=distribution.mean.clone()
                    commands.append(action[:,0].detach().cpu())
                    action[:,0]=(action[:,0]+shift).clamp(-1,1)
                    vec.send(action.cpu().numpy());crashes+=int((vec.done&~vec.truncated).sum())
                rows.append(dict(seed=seed,collective_action_shift=shift,
                    mean_unmodified_collective_action=float(torch.stack(commands).mean()),
                    final_payload_error_mm=float(torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1).mean()*1000),
                    mean_drone_height_m=float(vec.sim.position[:,0,2].mean()),
                    mean_payload_height_m=float(vec.sim.position[:,-1,2].mean()),crashes=crashes))
                vec.close()
    finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
    return dict(version=1,runs=rows,scope='Five-second deterministic offline perturbations on reused domains; action +/-0.04 means collective throttle +/-0.02 before clipping. No training, policy promotion, action-advantage ranking, hardware or long-horizon fidelity claim.')
