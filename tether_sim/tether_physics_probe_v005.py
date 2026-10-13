"""Version5 tether equilibrium/contact diagnostics. Caller owns GPU lease."""
import torch

def metrics(sim):
    lengths=torch.linalg.vector_norm(sim.position[:,1:]-sim.position[:,:-1],dim=-1)
    kinetic=.5*(sim.mass.double()*sim.velocity.square().sum(-1)).sum(-1)
    potential=(sim.mass.double()*9.81*sim.position[:,:,2]).sum(-1)
    return lengths,kinetic,potential

def probe():
    from .cuda_backend import CudaTether
    result={'version':5,'scope':'Corrected CUDA physics4, still-air ideal simulation diagnostics; no policy training, hardware calibration or strict navigation validation','cases':[]}
    for dt,substeps,iterations in ((.001,20,32),(.001,20,128),(.001,20,256),(.0005,40,128),(.0005,40,256)):
        sim=CudaTether(6,library="tmp/cuda-tether-v004/libtether.so");sim.params[:,8]=1;sim.wind.zero_()
        # Taut equilibrium at actual takeoff goal heights: carrier .8, payload .2.
        sim.position.zero_();sim.position[:,:,2]=.8-torch.arange(14,device='cuda').double()*.6/13
        # Valid slack zigzag: each link has rest length, endpoint separation < .6.
        rest=.6/13;vertical=rest*.5;horizontal=(rest*rest-vertical*vertical)**.5
        sim.position[1,:,0]=(torch.arange(14,device='cuda')%2).double()*horizontal
        sim.position[1,:,2]=1.0-torch.arange(14,device='cuda').double()*vertical
        # Existing production ground start, including its initial constraint preload.
        sim.position[2,:,0]=torch.arange(14,device='cuda').double()*.6/13
        sim.position[2,:,2]=.025;sim.position[2,0,2]=.05
        # Control start: all links exactly rest length, payload remains at floor.
        sim.position[3,:,0]=torch.arange(14,device='cuda').double()*((.6*.6-.025*.025)**.5)/13
        sim.position[3,:,2]=.05-torch.arange(14,device='cuda').double()*.025/13
        sim.position[4]=sim.position[2];sim.position[5]=sim.position[3]
        sim.velocity.zero_();sim.orientation.zero_();sim.orientation[:,0]=1
        thrust=sim.mass.sum(1)*9.81;sim.motors[:]=thrust[:,None]/4
        sim.motors[4:].zero_() # Actual training resets motors off;25ms lag retained.
        actions=(thrust/(4*sim.params[:,2]))[:,None].repeat(1,4).contiguous()
        initial=sim.position.clone();lengths,k,u=metrics(sim);energy0=k+u
        work=torch.zeros(6,device='cuda',dtype=torch.float64);peak_k=k.clone();peak_residual=torch.zeros_like(work)
        max_error=(lengths-rest).abs().amax(1);frames=[]
        for step in range(250):
            previous=sim.position.clone();before_v=sim.velocity[:,0].clone()
            sim.step(actions,dt=dt,substeps=substeps,iterations=iterations)
            lengths,k,u=metrics(sim)
            # Equal thrust retains identity orientation. Trapezoid drag work is
            # approximate at20ms; positive residual is a diagnostic, not proof.
            delta=sim.position[:,0]-previous[:,0]
            drag=-sim.params[:,3,None].double()*.5*(before_v+sim.velocity[:,0])
            force=drag.clone();force[:,2]+=sim.motors.sum(1).double()
            work+=(force*delta).sum(1)
            residual=k+u-energy0-work
            peak_residual=torch.maximum(peak_residual,residual);peak_k=torch.maximum(peak_k,k)
            max_error=torch.maximum(max_error,(lengths-rest).abs().amax(1))
            if step in (0,4,24,49,99,249):
                frames.append(dict(time_s=(step+1)*.02,carrier=sim.position[:,0].cpu().tolist(),payload=sim.position[:,-1].cpu().tolist(),carrier_velocity=sim.velocity[:,0].cpu().tolist(),payload_velocity=sim.velocity[:,-1].cpu().tolist(),kinetic_J=k.cpu().tolist(),energy_residual_J=residual.cpu().tolist(),max_link_error_m=(lengths-rest).abs().amax(1).cpu().tolist()))
        torch.cuda.synchronize()
        finite=bool(torch.isfinite(sim.position).all() and torch.isfinite(sim.velocity).all())
        result['cases'].append(dict(dt=dt,substeps=substeps,iterations=iterations,names=['taut_equilibrium','valid_slack_to_taut','production_ground_release','valid_ground_release','production_ground_motors_off','valid_ground_motors_off'],balanced_throttle=actions[:,0].cpu().tolist(),total_mass_kg=sim.mass.sum(1).cpu().tolist(),finite=finite,failed=sim.failed.cpu().tolist(),max_link_error_m=max_error.cpu().tolist(),peak_kinetic_J=peak_k.cpu().tolist(),peak_positive_energy_residual_J=peak_residual.cpu().tolist(),final_displacement_m=(sim.position-initial).cpu().tolist(),frames=frames))
    # Small independent differential pulses, tether retained, initially hanging.
    sim=CudaTether(3,library="tmp/cuda-tether-v004/libtether.so");sim.wind.zero_();thrust=sim.mass.sum(1)*9.81
    sim.motors[:]=thrust[:,None]/4
    base=(thrust/(4*sim.params[:,2]))[:,None].repeat(1,4)
    mix=torch.tensor([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]],device='cuda').float()
    actions=base+.005*mix.T
    sim.step(actions.contiguous(),dt=.001,substeps=20,iterations=32)
    result['pulses']=dict(duration_s=.02,differential_command=.005,rates_rad_s=sim.orientation[:,4:].cpu().tolist(),failed=sim.failed.cpu().tolist(),scope='Balanced initial hanging state, positive roll/pitch/yaw motor pulses; no attachment torque model')
    result['status']='measured'
    return result
