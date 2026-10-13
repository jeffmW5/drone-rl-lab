import json,math,hashlib
from pathlib import Path
import numpy as np
import torch
from tether_sim.cuda_task import CudaPolicy

torch.set_num_threads(1)
ROOT=Path('/home/n33du/ai/drone-rl-lab');OUT=ROOT/'tmp/deep_audit_physics'
existing=json.loads((OUT/'result.json').read_text());results=[]
# Rigid uniform lateral translation subspace of the14-node tether around ideal hover.
# This is an analytical reduced linearization, NOT the full PBD numerical Jacobian.
# Includes motor25ms lag, differential-force observation feedback, physical torque.
for trial,version in ((758,1),(759,2),(761,2),(763,2),(713,2)):
    path=ROOT/f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff/best.pt'
    model=CudaPolicy();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.configure_collective_observation(version);model.eval()
    p=torch.zeros(14,3);p[:,2]=.6-torch.arange(14)*.6/13
    q=torch.zeros(7);q[0]=1
    motor=torch.full((4,),(.0434+.001+.005)*9.81/4)
    obs=torch.cat((p.ravel(),torch.zeros(42),q,motor,torch.tensor([0,0,.2]))).requires_grad_()
    action,_=model(obs[None]);mean=action.mean[0]
    j=torch.stack([torch.autograd.grad(mean[i],obs,retain_graph=True)[0] for i in range(4)]).detach().numpy()
    # Columns:global x,y,vx,vy,rollrad,pitchrad,wx,wy,torqueXNm,torqueYNm.
    derivatives=np.zeros((4,10))
    derivatives[:,0]=j[:,:42:3].sum(1);derivatives[:,1]=j[:,1:42:3].sum(1)
    derivatives[:,2]=j[:,42:84:3].sum(1);derivatives[:,3]=j[:,43:84:3].sum(1)
    derivatives[:,4]=.5*j[:,85];derivatives[:,5]=.5*j[:,86]
    derivatives[:,6]=j[:,88];derivatives[:,7]=j[:,89]
    mix=np.array([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]])
    derivatives[:,8]=j[:,91:95]@mix[:,0]/.128
    derivatives[:,9]=j[:,91:95]@mix[:,1]/.128
    a=mean.detach().numpy();collective=.5*(float(a[0])+1)
    A=np.zeros((10,10));A[0,2]=1;A[1,3]=1;A[2,2]=A[3,3]=-.015/.0494
    accel=.8*collective/.0494
    A[2,5]=accel;A[3,4]=-accel;A[4,6]=A[5,7]=1
    A[6,8]=A[7,9]=1/.000025
    A[8,:]=.000512/.025*derivatives[1];A[8,8]-=1/.025
    A[9,:]=.000512/.025*derivatives[2];A[9,9]-=1/.025
    eig=np.linalg.eigvals(A)
    row=dict(trial=trial,collective_observation_version=version,mean_at_hover=a.tolist(),derivative_columns=['global_x','global_y','global_vx','global_vy','roll_rad','pitch_rad','wx_rad_s','wy_rad_s','motor_torqueX_Nm','motor_torqueY_Nm'],action_derivatives=derivatives.tolist(),linear_matrix=A.tolist(),eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in sorted(eig,key=lambda z:z.real,reverse=True)],scope='Reduced rigid lateral subspace local Jacobian. Not full cable mode/contact stability proof; anchor has small nonzero action/drift bias and is not an exact fixed point.')
    results.append(row)
    print(json.dumps(dict(trial=trial,mean=a.tolist(),roll_derivatives=derivatives[1,:8].tolist(),pitch_derivatives=derivatives[2,:8].tolist(),max_eigen_real=float(eig.real.max()))))
(OUT/'policy_derivatives.json').write_text(json.dumps(results,indent=2))
for row in existing['replays']:
    if row['scenario']=='replay_baseline':print(json.dumps({k:row[k] for k in ['trial','max_shared_frame_carrier_difference_m','cpu_gpu_failure_time_difference_s']}))
for row in existing['physical_checks']:
    if row['kind']=='20ms_max_channel_pulse' and row['physics']==5 and row['sign']==1:print(json.dumps(dict(channel=row['channel'],angle_rad=2*math.atan2(row['q'][row['channel']],row['q'][0]),rate_rad_s=row['q'][row['channel']+3])))
