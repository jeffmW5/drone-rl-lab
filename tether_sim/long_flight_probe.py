"""Separate 30-second diagnostic; original five-second benchmark untouched."""
def probe(policy,plan):
    import torch
    from .cuda_task import CudaVector
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();rows=[]
    try:
        for seed in plan['validation_seeds']:
            vec=CudaVector(8,direct=True,reward_version=7,phase='takeoff',control_version=2)
            vec.async_reset(seed);crashes=0
            for step in range(1,1501):
                vec.steps.zero_() # only this diagnostic disables artificial timeout
                with torch.no_grad():distribution,_=policy(vec.obs())
                vec.send(distribution.mean.cpu().numpy())
                crashes+=int(vec.done.sum())
                if step in (250,500,1000,1500):
                    rows.append(dict(seed=seed,time_s=step*.02,crash_resets=crashes,
                        payload_error_mm=float(torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1).mean()*1000),
                        drone_height_m=float(vec.sim.position[:,0,2].mean()),
                        payload_height_m=float(vec.sim.position[:,-1,2].mean()),
                        payload_vertical_velocity_m_s=float(vec.sim.velocity[:,-1,2].mean())))
            vec.close()
    finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
    return dict(version=1,runs=rows,scope='Separate deterministic30s no-timeout diagnostic, reused domains, physical crashes still reset and are counted; not comparable to frozen5s benchmark or hardware fidelity claim')
