"""CPU invariants for reward-only velocity reference; no simulator or actions."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import ast
import numpy as np
from unittest.mock import patch
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tether_sim.fine_zone_reward import fine_zone_reward
from tether_sim.velocity_reference_reward import (
    horizontal_velocity_reference, velocity_reference_reward)


def states(count):
    positions = torch.zeros(count, 14, 3, dtype=torch.float64)
    positions[:, 0, 2] = .8
    velocity = torch.zeros_like(positions)
    orientation = torch.zeros(count, 7, dtype=torch.float64)
    orientation[:, 0] = 1
    center = torch.zeros(count, 3, dtype=torch.float64)
    failed = torch.zeros(count, dtype=torch.int32)
    return positions, velocity, orientation, center, failed


def require(condition, label):
    if not bool(condition):
        raise AssertionError(label)


def scalar_reward(x, y, z, vx, vy, vz, qx=0., qy=0., failed=0, version=17):
    """Independent scalar definitions of reward16 and its single reward17 change."""
    smooth = lambda d: math.sqrt(d*d+.01**2)-.01
    extent = max(abs(x),abs(y))
    if failed or extent >= 2:
        return -1.
    ground = .8*(z < .15)
    if extent > 1:
        return -(extent-1)-ground
    rx, ry = -x, -y
    norm = math.hypot(rx,ry)
    if norm > .5:
        rx, ry = rx*.5/norm, ry*.5/norm
    if version == 16:
        rx, ry = 0., 0.
    cost = 4*smooth(z-.8)+smooth(x)+smooth(y)+2*((vx-rx)**2+(vy-ry)**2+vz**2)
    airborne = min(1.,max(0.,(z-.05)/.25))
    core = .95*airborne/(1+cost)+.05*(1-2*(qx*qx+qy*qy))
    return (1-extent)*core-ground



def integration_checks(root):
    """Exact production reward dispatch/reset logic on an input-only CPU backend."""
    import tether_sim.cuda_task as task
    baseline_path=root/'state/cuda_research/methodology/review_0112/velocity_reference/cuda_task_before_reward17.py'
    baseline_text=baseline_path.read_text()
    node=next(n for n in ast.parse(baseline_text).body if isinstance(n,ast.FunctionDef) and n.name=='zone_hover_reward')
    code='\n'.join(baseline_text.splitlines()[node.lineno-1:node.end_lineno])
    scope={'torch':torch}
    exec(compile(code,str(baseline_path),'exec'),scope)
    old_zone=scope['zone_hover_reward']
    generator=torch.Generator().manual_seed(41)
    inputs=states(32)
    inputs[0][:,0,:2]=torch.rand(32,2,generator=generator,dtype=torch.float64)*5-2.5
    inputs[0][:,0,2]=torch.rand(32,generator=generator,dtype=torch.float64)*1.5
    inputs[1][:,0]=torch.randn(32,3,generator=generator,dtype=torch.float64)
    inputs[2][:,:4]=torch.randn(32,4,generator=generator,dtype=torch.float64)
    inputs[2][:,:4]/=inputs[2][:,:4].norm(dim=1,keepdim=True)
    inputs[3][:,:2]=torch.rand(32,2,generator=generator,dtype=torch.float64)-.5
    inputs[4][::7]=1
    legacy={}
    for version in (1,2,3):
        expected=old_zone(*inputs,version=version)
        actual=task.zone_hover_reward(*inputs,version=version)
        require(all(torch.equal(a,b) for a,b in zip(actual,expected)), 'exact old14/15/16 reward function preservation')
        legacy[str(version)]=True
    require(all(torch.equal(a,b) for a,b in zip(task.zone_hover_reward(*inputs,version=4),velocity_reference_reward(*inputs))), 'zoneversion4 exact new helper dispatch')
    invalid=[]
    for version in (-1,0,99):
        try:task.zone_hover_reward(*inputs,version=version)
        except ValueError:invalid.append(version)
        else:raise AssertionError('unsupported zone version accepted')

    # Velocity and displacement are world XY in the simulator. Pure yaw should
    # not rotate this scoring reference or alter upright's existing value.
    frame=states(2);frame[0][:,0,:2]=torch.tensor([[.2,-.1],[.2,-.1]],dtype=torch.float64)
    frame[1][:,0,:2]=torch.tensor([[-.1,.05],[-.1,.05]],dtype=torch.float64)
    frame[2][1,:4]=torch.tensor([math.sqrt(.5),0,0,math.sqrt(.5)],dtype=torch.float64)
    frame_reward=velocity_reference_reward(*frame)[0]
    require(frame_reward[0]==frame_reward[1], 'world-frame reference is invariant to pure yaw')

    class FakePhysics:
        def __init__(self,count):
            self.position,self.velocity,self.orientation,_,self.failed=states(count)
            self.motors=torch.zeros(count,4,dtype=torch.float64)
            self.mass=torch.ones(count,14,dtype=torch.float64)*.001
            self.step_calls=[]
        def step(self,command,dt=.001,substeps=20):
            self.step_calls.append({'command':command.clone(),'dt':dt,'substeps':substeps})
    def vector(count,version):
        vec=object.__new__(task.CudaVector)
        vec.sim=FakePhysics(count);vec.direct=True;vec.control_version=2
        vec.reward_version=version;vec.integration_version=1;vec.diagnostic_zone_version=1
        vec.zone_counts={'band_steps':0,'escapes':0};vec.termination_version=1
        vec.training_region=None;vec.landing_gate=None
        vec.goal=torch.zeros(count,3,dtype=torch.float64);vec.goal[:,2]=.2
        vec.steps=torch.zeros(count,dtype=torch.int32)
        vec.start=torch.zeros(count,14,3,dtype=torch.float64);vec.start[:,:,2]=.025;vec.start[:,0,2]=.05
        vec.reset_motor_forces=torch.zeros(count,4,dtype=torch.float64)
        vec.reset_orientation=None;vec.terminal_observer=None
        vec.previous_error=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
        vec.ids=np.arange(count);vec.mask=np.ones(count,dtype=bool)
        return vec
    def cpu_call(fn):
        def wrapped(*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return fn(*args,**kwargs)
        return wrapped
    real_tensor,real_as_tensor=torch.tensor,torch.as_tensor
    def forbidden_cuda(*args,**kwargs):raise AssertionError('CPU tests must never allocate CUDA physics')
    terminal=[]
    with patch.object(task,'CudaTether',forbidden_cuda),patch('torch.tensor',cpu_call(real_tensor)),patch('torch.as_tensor',cpu_call(real_as_tensor)):
        vec=vector(8,17)
        xy=real_tensor([[.2,0],[1,0],[1.3,0],[2,0],[-2,0],[0,2],[.01,.01],[.1,.1]],dtype=torch.float64)
        vec.sim.position[:,0,:2]=xy;vec.sim.velocity[0,0,0]=-.1;vec.sim.failed[6]=1
        expected=velocity_reference_reward(vec.sim.position,vec.sim.velocity,vec.sim.orientation,vec.start[:,0],vec.sim.failed)
        pre_position=vec.sim.position.clone()
        def observe(obs,done,truncated,failed,reward):
            terminal.append({'obs':obs.clone(),'done':done.clone(),'truncated':truncated.clone(),'failed':failed.clone(),'reward':reward.clone()})
        vec.terminal_observer=observe
        actions=np.zeros((8,4),dtype=np.float32);actions[:,0]=.2;actions_before=actions.copy()
        vec.send(actions)
        require(np.array_equal(actions,actions_before), 'adapter does not mutate supplied actions')
        require(torch.equal(vec.reward,expected[0]), 'real reward17 adapter dispatch matches pure helper')
        require(vec.done.tolist()==[False,False,False,True,True,True,True,False] and not vec.truncated.any(), 'true escape/physical failure flags retained')
        require(vec.zone_counts=={'band_steps':1,'escapes':3}, 'production zone counters retained')
        require(terminal[0]['failed'].tolist()==[0,0,0,3,3,3,1,0], 'pre-reset failure codes retained')
        require(torch.allclose(terminal[0]['obs'][:,:3].double(),pre_position[:,0]-vec.goal,rtol=0,atol=1e-7), 'pre-reset observation retained')
        require(torch.equal(terminal[0]['reward'],vec.reward), 'terminal rewards persist through reset')
        require(torch.equal(vec.sim.position[vec.done],vec.start[vec.done]) and torch.equal(vec.sim.position[~vec.done],pre_position[~vec.done]), 'only failed states reset')
        require(not vec.sim.failed.any() and torch.all(vec.steps[vec.done]==0), 'failed state flags/steps clear after observer')
        require(torch.all(vec.sim.velocity[vec.done]==0) and torch.all(vec.sim.motors[vec.done]==0), 'failed velocity/motor state reset')
        require(torch.allclose(vec.sim.step_calls[0]['command'],torch.full((8,4),.6),rtol=0,atol=1e-7), 'learned motor command independent of reward reference')
        require(vec.sim.step_calls[0]['dt']==.001 and vec.sim.step_calls[0]['substeps']==20, 'same integration parameters reach backend')
        observed,reward,done,truncated,*rest=vec.recv()
        require(observed.shape==(8,98) and torch.equal(reward,expected[0]) and torch.equal(done,vec.done) and not truncated.any(), 'Puffer-facing reward/flags survive autoreset')
        old_vectors={}
        for reward_version,zone_version in ((14,1),(15,2),(16,3)):
            old_vec=vector(3,reward_version)
            old_vec.sim.position[:,0,:2]=real_tensor([[0,0],[.2,.3],[1.2,0]],dtype=torch.float64)
            expected_old=old_zone(old_vec.sim.position,old_vec.sim.velocity,old_vec.sim.orientation,old_vec.start[:,0],old_vec.sim.failed,version=zone_version)[0]
            old_vec.send(np.zeros((3,4),dtype=np.float32))
            require(torch.equal(old_vec.reward,expected_old), 'exact legacy adapter reward dispatch retained')
            old_vectors[str(reward_version)]=True
        timeout=vector(1,17);timeout.steps[:]=499;timeout.send(np.zeros((1,4),dtype=np.float32))
        require(timeout.done.item() and timeout.truncated.item() and timeout.steps.item()==0, 'timeout remains distinct and resets')
        numerical=vector(1,17);numerical.sim.failed[:]=2
        try:numerical.send(np.zeros((1,4),dtype=np.float32))
        except FloatingPointError:numerical_rejected=True
        else:raise AssertionError('numerical failure should reject trial')
    return {'scope':'Exact CudaVector.send/obs/recv with input-only CPU fake physics and CUDA constructor forbidden. Checks scoring/dispatch/state logic, not integration physics or learned flight.',
            'baseline_source':{'path':str(baseline_path),'sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest()},'old_zone_functions_exact':legacy,'old_adapter_dispatch_exact':old_vectors,
            'zoneversion4_helper_exact':True,'invalid_versions_rejected':invalid,'yaw_frame_rewards':frame_reward.tolist(),'actions_unmodified_and_no_reference_motor_helper':True,
            'adapter_rewards':vec.reward.tolist(),'done':vec.done.tolist(),'truncated':vec.truncated.tolist(),'pre_reset_failed':terminal[0]['failed'].tolist(),'zone_counts':vec.zone_counts,
            'command_for_collective_point2':vec.sim.step_calls[0]['command'].tolist(),'timeout_distinct_pass':True,'numerical_failure_rejected':numerical_rejected}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = []
    displacement = torch.tensor([[0,0],[.001,0],[-.001,0],[.3,.4],[.5,.5],[-2,1],[0,-1]],dtype=torch.float64)
    reference = horizontal_velocity_reference(displacement)
    require(torch.isfinite(reference).all(), 'zero-safe reference finite')
    require(torch.equal(reference[0],torch.zeros(2,dtype=torch.float64)), 'zero displacement means zero reference')
    require((reference.norm(dim=1) <= .5+1e-12).all(), 'vector norm cap0.5m/s')
    require(torch.equal(reference[1:4],-displacement[1:4]), 'gain1persecond below cap')
    require(torch.all((reference[1:]*displacement[1:]).sum(1)<0), 'reference points inward for either axis sign')
    checks.append('zero_safe_inward_one_per_second_reference_vector_norm_cap')

    # Matched magnitudes isolate direction, both signs and all diagonal quadrants.
    offsets = torch.tensor([[.2,0],[-.2,0],[0,.2],[0,-.2],[.25,.3],[-.25,.3],[.25,-.3],[-.25,-.3],[.8,.1],[-.8,-.1]],dtype=torch.float64)
    inward = states(len(offsets)); outward = states(len(offsets))
    direction = offsets/offsets.norm(dim=1,keepdim=True)
    inward[0][:,0,:2] = offsets; outward[0][:,0,:2] = offsets
    inward[1][:,0,:2] = -.2*direction; outward[1][:,0,:2] = .2*direction
    reward_in = velocity_reference_reward(*inward)[0]
    reward_out = velocity_reference_reward(*outward)[0]
    legacy_in = fine_zone_reward(*inward)[0]; legacy_out = fine_zone_reward(*outward)[0]
    require(torch.all(reward_in > reward_out), 'same-speed inward movement beats outward in each quadrant')
    require(torch.equal(legacy_in,legacy_out), 'reward16 controls are indifferent to same-speed direction')
    checks.append('inward_beats_outward_same_speed_both_axes_and_diagonal_signs')

    centered = states(6)
    centered[0][:,0,2] = torch.tensor([.8,.805,.9,.7,.05,.1],dtype=torch.float64)
    old = fine_zone_reward(*centered); new = velocity_reference_reward(*centered)
    require(all(torch.equal(x,y) for x,y in zip(old,new)), 'center stationary exactly preserves reward16')
    speeds = torch.tensor([0,.01,.1,.2,.5,1.],dtype=torch.float64)
    centered[0][:,0,2] = .8; centered[1][:,0,2] = speeds
    old_z = fine_zone_reward(*centered)[0]; new_z = velocity_reference_reward(*centered)[0]
    require(torch.equal(old_z,new_z), 'vertical speed cost unchanged at zero XY displacement')
    require(torch.all(new_z[:-1]>new_z[1:]), 'increasing vertical speed still costs reward')
    centered[1][:,0,2] *= -1
    require(torch.equal(velocity_reference_reward(*centered)[0],new_z), 'vertical speed sign symmetry preserved')
    checks.append('center_stationary_and_vertical_speed_cost_exact_reward16_parity')

    # Independent scalar parity checks both production16 baseline and candidate17.
    cases = [(.001,-.002,.805,-.01,.02,.01,0),(.2,.3,.8,-.2,-.1,.05,0),(.8,.1,.75,-.4,.1,-.2,0),(-.3,.4,.9,.1,-.2,.2,0),(1.3,-.1,.8,-.2,0,0,0),(1.3,0,.05,0,0,0,0),(.1,.1,.1,-.1,-.1,0,0),(.01,.01,.8,0,0,0,1),(2,0,.8,-.5,0,0,0)]
    parity=[]
    for x,y,z,vx,vy,vz,failed in cases:
        inputs=states(1)
        inputs[0][0,0] = torch.tensor([x,y,z],dtype=torch.float64)
        inputs[1][0,0] = torch.tensor([vx,vy,vz],dtype=torch.float64)
        inputs[4][0] = failed
        actual16=float(fine_zone_reward(*inputs)[0][0])
        actual17=float(velocity_reference_reward(*inputs)[0][0])
        expected16=scalar_reward(x,y,z,vx,vy,vz,failed=failed,version=16)
        expected17=scalar_reward(x,y,z,vx,vy,vz,failed=failed)
        require(abs(actual16-expected16)<1e-7, 'independent production16 baseline formula parity')
        require(abs(actual17-expected17)<1e-7, 'independent reward17 formula parity')
        parity.append({'case':list((x,y,z,vx,vy,vz,failed)),'expected16':expected16,'actual16':actual16,'expected17':expected17,'actual17':actual17})
    checks.append('independent_scalar_formula_and_production16_baseline_parity')

    xy=torch.tensor([[0,0],[.9,.9],[1,0],[1,1],[1.0001,0],[-1.5,0],[0,1.9999],[2,0],[-2,0],[0,2],[2,2]],dtype=torch.float64)
    boundary_inputs=states(len(xy));boundary_inputs[0][:,0,:2]=xy
    reward,escaped,band=velocity_reference_reward(*boundary_inputs)
    old_reward,old_escaped,old_band=fine_zone_reward(*boundary_inputs)
    require(torch.equal(escaped,old_escaped) and torch.equal(band,old_band), 'square masks unchanged from16')
    require(reward[1]>0 and not escaped[1], 'square0.9corner remains inside')
    require(reward[2]==reward[3]==0, 'airborne exact1m boundary0')
    require(torch.all(reward[4:7]<0), 'entire1to2m band negative')
    require(torch.equal(reward[2:],old_reward[2:]), 'boundary/band/failure values unchanged from16')
    require(torch.all(reward[7:]==-1), 'exact2m failure reward')
    epsilon=1e-6
    continuous=states(6)
    continuous[0][:,0,0]=torch.tensor([1-epsilon,1,1+epsilon,2-epsilon,2,2+epsilon],dtype=torch.float64)
    edge=velocity_reference_reward(*continuous)[0]
    require(abs(float(edge[0]-edge[1]))<2*epsilon and abs(float(edge[2]-edge[1]))<2*epsilon, 'airborne1m continuity')
    require(abs(float(edge[3]-edge[4]))<2*epsilon and edge[4]==edge[5]==-1, 'airborne2m continuity')
    checks.append('unchanged_square_bounds_negative_band_and_airborne_continuity')

    ground=states(len(xy));ground[0][:,0,:2]=xy;ground[0][:,0,2]=.05
    ground_reward=velocity_reference_reward(*ground)[0]
    require(torch.all(ground_reward<0), 'ground remains negative')
    require(torch.equal(ground_reward,fine_zone_reward(*ground)[0]), 'motors-off ground spatial/ground clauses unchanged')
    require(ground_reward[5]<-1 and ground_reward[5].clamp(-1,1)==-1, 'existing ground band lower clipping limitation preserved')
    failed=states(3);failed[4][:]=torch.tensor([0,1,3],dtype=torch.int32)
    failure_reward=velocity_reference_reward(*failed)[0]
    require(failure_reward[0]==1 and torch.all(failure_reward[1:]==-1), 'failure override retains16definition')
    checks.append('negative_ground_failure_override_and_ground_band_clip_limitation_retained')

    orientation=states(3)
    orientation[2][1,4:]=torch.tensor([100.,-100.,50.],dtype=torch.float64)
    angular=velocity_reference_reward(*orientation)[0]
    require(angular[0]==angular[1], 'angular rates unpenalized')
    orientation[2][2,0]=0.;orientation[2][2,1]=1.
    tilted=velocity_reference_reward(*orientation)[0]
    require(torch.equal(tilted,fine_zone_reward(*orientation)[0]), 'upright coefficient unchanged at centered state')
    require(tilted[2]<tilted[0], 'upright reward retained')
    checks.append('upright_terms_preserved_and_angular_rate_penalty_zero')

    # Test numeric bounds and pure behavior on a wider CPU tensor batch. An
    # explicit local generator avoids altering the process-global RNG state.
    generator=torch.Generator().manual_seed(41)
    mixed=states(256)
    mixed[0][:,0,:2]=torch.rand(256,2,generator=generator,dtype=torch.float64)*5-2.5
    mixed[0][:,0,2]=torch.rand(256,generator=generator,dtype=torch.float64)*2
    mixed[1][:,0]=torch.randn(256,3,generator=generator,dtype=torch.float64)*2
    mixed[2][:,:4]=torch.randn(256,4,generator=generator,dtype=torch.float64)
    mixed[2][:,:4]/=mixed[2][:,:4].norm(dim=1,keepdim=True)
    originals=tuple(x.clone() for x in mixed);rng=torch.get_rng_state().clone()
    output=velocity_reference_reward(*mixed)
    require(all(torch.equal(x,y) for x,y in zip(mixed,originals)), 'no input mutation')
    require(torch.equal(rng,torch.get_rng_state()), 'no global RNG mutation')
    require(torch.isfinite(output[0]).all(), 'all output rewards finite')
    require(torch.all(output[0]<=1), 'upper bound1 for normalized quaternion input')
    require(output[0].dtype==torch.float32 and output[1].dtype==torch.bool and output[2].dtype==torch.bool, 'same return dtypes')
    shifted=tuple(x.clone() for x in mixed)
    shifted[0][:,:,0]+=5.;shifted[0][:,:,1]-=3.
    shifted[3][:,0]+=5.;shifted[3][:,1]-=3.
    require(torch.allclose(output[0],velocity_reference_reward(*shifted)[0],rtol=0,atol=1e-7), 'reference uses takeoff-relative XY')
    checks.append('finite_unit_quaternion_upper_bound_no_input_or_RNG_mutation_translation')

    # Keep sub-centimetre stationary center height gradation; direction shaping
    # does not reintroduce the old upper-clip plateau at the hover target.
    millimetres=torch.tensor([0.,.001,.002,.005,.01],dtype=torch.float64)
    near=states(len(millimetres));near[0][:,0,2]+=millimetres
    near_reward=velocity_reference_reward(*near)[0]
    require(torch.equal(near_reward,fine_zone_reward(*near)[0]), 'center height gradation exactly16')
    require(torch.all(near_reward[:-1]>near_reward[1:]) and torch.equal(near_reward,near_reward.clamp(-1,1)), 'near-hover clipped fine resolution preserved')
    checks.append('stationary_center_millimetre_height_gradation_and_no_upper_clip_plateau')

    root=Path(__file__).resolve().parents[1]
    integration=integration_checks(root)
    checks.append('production17_dispatch_old14_15_16_parity_frame_actions_failure_and_autoreset')
    result={'integration':integration,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU-only pure reward functions. Reference velocity is used only in the reward, never in actions, controller or simulator. No GPU, training, physics, angular-rate penalty, exploration or queue change. Checked normalized quaternions for upper bound. Existing ground-band lower clip and exactlyzero1m reward remain. This does not establish learned hover, simulation fidelity or hardware precision.','checks_passed':checks,'reference_displacement_m':displacement.tolist(),'reference_velocity_m_s':reference.tolist(),'reference_norm_m_s':reference.norm(dim=1).tolist(),'inward_outward':{'offsets_m':offsets.tolist(),'matched_speed_m_s':.2,'inward_reward17':reward_in.tolist(),'outward_reward17':reward_out.tolist(),'same_speed_reward16':legacy_in.tolist()},'independent_scalar_parity':parity,'square_xy_m':xy.tolist(),'square_reward17':reward.tolist(),'square_reward16':old_reward.tolist(),'square_escaped':escaped.tolist(),'square_band':band.tolist(),'boundary_rewards':edge.tolist(),'ground_rewards':ground_reward.tolist(),'vertical_speed_m_s':speeds.tolist(),'vertical_speed_reward16_and17':new_z.tolist(),'near_center_height_error_m':millimetres.tolist(),'near_center_height_reward16_and17':near_reward.tolist(),'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tether_sim/fine_zone_reward.py',root/'tether_sim/velocity_reference_reward.py',root/'tether_sim/cuda_task.py',Path(__file__).resolve())}}
    if args.output:
        outputs=[Path(args.output)]
    else:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        name=f'velocity_reference_cpu_checks_{stamp}.json'
        outputs=[root/'state/cuda_research/methodology/review_0112/velocity_reference'/name,root/'tmp/deep_audit_learner'/name]
    if any(path.exists() for path in outputs):raise FileExistsError('Refuse existing reward evidence')
    for path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'checks_passed':len(checks),'outputs':[str(p) for p in outputs],'inward_advantage_min':float((reward_in-reward_out).min()),'center_stationary_exact16':True,'reference_max_norm_m_s':float(reference.norm(dim=1).max()),'no_input_or_rng_mutation':True},indent=2))


if __name__=='__main__':
    main()
