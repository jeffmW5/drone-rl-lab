"""Read-only policy state-response diagnostic; caller owns serial GPU lease."""
import copy
import torch
from .cuda_task import CudaVector

def probe(policy,plan):
    vec=CudaVector(17,direct=True,reward_version=14,control_version=2,physics_version=5,phase='takeoff',training_reset_version=3,diagnostic_zone_version=1)
    s=vec.sim
    s.position.zero_();s.position[:,:,2]=.8-torch.arange(14,device='cuda').double()*.6/13
    s.velocity.zero_();s.orientation.zero_();s.orientation[:,0]=1
    s.motors[:]=s.mass.sum(1,keepdim=True)*9.81/4
    vec.goal.zero_();vec.goal[:,2]=.2
    s.position[1,:,2]-=.15;s.position[2,:,2]+=.2
    s.velocity[3,:,2]=.5;s.velocity[4,:,2]=-.5
    angle=torch.tensor(10*3.141592653589793/180/2,device='cuda')
    for row,axis,sign in ((5,1,1),(6,1,-1),(7,2,1),(8,2,-1)):
        s.orientation[row,0]=angle.cos();s.orientation[row,axis]=sign*angle.sin()
    for row,axis,sign in ((9,0,1),(10,0,-1),(11,1,1),(12,1,-1)):
        s.position[row,:,axis]+=sign*.5
    for row,axis,sign in ((13,0,1),(14,0,-1),(15,1,1),(16,1,-1)):
        s.velocity[row,:,axis]=sign*.5
    names=['hover','below','above','ascending','descending','roll_plus10','roll_minus10','pitch_plus10','pitch_minus10','x_plus','x_minus','y_plus','y_minus','vx_plus','vx_minus','vy_plus','vy_minus']
    weights=torch.load(plan['policy_response_reference'],weights_only=True)
    reference=copy.deepcopy(policy);weights.setdefault('obs_scale',reference.obs_scale);reference.load_state_dict(weights)
    mix=torch.tensor([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]],device='cuda').float()
    result={'version':2,'scope':'Synthetic matched full-tether states; policy response is not a stability proof or observed flight accuracy','reference_checkpoint':plan['policy_response_reference'],'models':{},'observations':vec.obs().cpu().tolist()}
    for name,model in (('current',policy),('reference',reference)):
        with torch.no_grad():dist,value=model(vec.obs());actions=dist.mean;command=(.5*(actions[:,:1]+1)+.02*(actions[:,1:]@mix.T)).clamp(0,1)
        collective=command.mean(1)
        result['models'][name]=dict(cases=[dict(name=names[i],mean_action=actions[i].cpu().tolist(),motor_command=command[i].cpu().tolist(),value=float(value[i,0])) for i in range(17)],collective_ascending_minus_hover=float(collective[3]-collective[0]),collective_descending_minus_hover=float(collective[4]-collective[0]),collective_above_minus_hover=float(collective[2]-collective[0]),collective_below_minus_hover=float(collective[1]-collective[0]),equilibrium_collective=float(s.mass[0].sum()*9.81/(4*s.params[0,2])))
    return result
