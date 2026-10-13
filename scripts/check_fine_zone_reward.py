"""CPU invariants of production fine reward, including the learner's hard clip."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import ast
import numpy as np
from unittest.mock import patch
import datetime
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tether_sim.fine_zone_reward import fine_zone_reward, soft_absolute
from tether_sim.cuda_task import zone_hover_reward


def states(count):
    p = torch.zeros(count, 14, 3, dtype=torch.float64)
    p[:, 0, 2] = .8
    v = torch.zeros_like(p)
    q = torch.zeros(count, 7, dtype=torch.float64)
    q[:, 0] = 1
    center = torch.zeros(count, 3, dtype=torch.float64)
    failed = torch.zeros(count, dtype=torch.int32)
    return p, v, q, center, failed


def require(condition, label):
    if not bool(condition):
        raise AssertionError(label)



def integration_checks(root):
    """Test real production dispatch/reset logic with CPU input-only fake physics."""
    import tether_sim.cuda_task as task
    reference_path = root/'state/cuda_research/methodology/review_0109/recovery_curriculum/cuda_task_before_reset6_source.py'
    reference_text = reference_path.read_text()
    node = next(n for n in ast.parse(reference_text).body if isinstance(n, ast.FunctionDef) and n.name=='zone_hover_reward')
    code = '\n'.join(reference_text.splitlines()[node.lineno-1:node.end_lineno])
    scope = {'torch':torch}
    exec(compile(code,str(reference_path),'exec'),scope)
    legacy_reward = scope['zone_hover_reward']
    generator = torch.Generator().manual_seed(41)
    inputs = states(32)
    inputs[0][:,0,:2] = torch.rand(32,2,generator=generator,dtype=torch.float64)*5-2.5
    inputs[0][:,0,2] = torch.rand(32,generator=generator,dtype=torch.float64)*1.5
    inputs[1][:,0] = torch.randn(32,3,generator=generator,dtype=torch.float64)
    inputs[2][:,:4] = torch.randn(32,4,generator=generator,dtype=torch.float64)
    inputs[2][:,:4] /= inputs[2][:,:4].norm(dim=1,keepdim=True)
    inputs[3][:,:2] = torch.rand(32,2,generator=generator,dtype=torch.float64)-.5
    inputs[4][::7] = 1
    old_exact={}
    for version in (1,2):
        expected=legacy_reward(*inputs,version=version)
        actual=task.zone_hover_reward(*inputs,version=version)
        require(all(torch.equal(x,y) for x,y in zip(actual,expected)), 'R14/15 exact frozen-source preservation')
        old_exact[str(version)] = True
    direct=fine_zone_reward(*inputs)
    dispatched=task.zone_hover_reward(*inputs,version=3)
    require(all(torch.equal(x,y) for x,y in zip(direct,dispatched)), 'zoneversion3 exact fine-helper dispatch')
    rejected=[]
    for version in (-1,0,4):
        try:task.zone_hover_reward(*inputs,version=version)
        except ValueError:rejected.append(version)
        else:raise AssertionError('invalid zone version accepted')

    class FakePhysics:
        def __init__(self,count):
            self.position,self.velocity,self.orientation,_,self.failed=states(count)
            self.motors=torch.zeros(count,4,dtype=torch.float64)
            self.mass=torch.ones(count,14,dtype=torch.float64)*.001
            self.step_calls=[]
        def step(self,command,dt=.001,substeps=20):
            self.step_calls.append({'command':command.clone(),'dt':dt,'substeps':substeps})
    def vector(count,reward_version):
        # Exact CudaVector.send/obs/recv; __new__ bypasses GPU constructor.
        vec=object.__new__(task.CudaVector)
        vec.sim=FakePhysics(count);vec.direct=True;vec.control_version=2
        vec.reward_version=reward_version;vec.integration_version=1
        vec.diagnostic_zone_version=1;vec.zone_counts={'band_steps':0,'escapes':0}
        vec.termination_version=1;vec.training_region=None;vec.landing_gate=None
        vec.goal=torch.zeros(count,3,dtype=torch.float64);vec.goal[:,2]=.2
        vec.steps=torch.zeros(count,dtype=torch.int32)
        vec.start=torch.zeros(count,14,3,dtype=torch.float64)
        vec.start[:,:,2]=.025;vec.start[:,0,2]=.05
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
    def forbidden_cuda(*args,**kwargs):
        raise AssertionError('CPU checks must never instantiate CUDA physics')
    terminal=[]
    with patch.object(task,'CudaTether',forbidden_cuda), patch('torch.tensor',cpu_call(real_tensor)), patch('torch.as_tensor',cpu_call(real_as_tensor)):
        vec=vector(8,16)
        coords=real_tensor([[.005,-.002],[1,0],[1.3,0],[2,0],[-2,0],[0,2],[.01,.01],[.1,.1]],dtype=torch.float64)
        vec.sim.position[:,0,:2]=coords;vec.sim.failed[6]=1
        expected=fine_zone_reward(vec.sim.position,vec.sim.velocity,vec.sim.orientation,vec.start[:,0],vec.sim.failed)
        pre_position=vec.sim.position.clone()
        def observe(obs,done,truncated,failed,reward):
            terminal.append({'obs':obs.clone(),'done':done.clone(),'truncated':truncated.clone(),'failed':failed.clone(),'reward':reward.clone()})
        vec.terminal_observer=observe
        vec.send(np.zeros((8,4),dtype=np.float32))
        require(torch.equal(vec.reward,expected[0]), 'CudaVector reward16 maps fine helper exactly')
        require(vec.done.tolist()==[False,False,False,True,True,True,True,False], 'true square escape/physicalfailure done flags')
        require(not vec.truncated.any(), 'true failures are not timeouts')
        require(vec.zone_counts=={'band_steps':1,'escapes':3}, 'actual production zone counters')
        require(len(terminal)==1 and terminal[0]['failed'].tolist()==[0,0,0,3,3,3,1,0], 'terminal observer captures true failure codes before reset')
        require(torch.allclose(terminal[0]['obs'][:,:3].double(),pre_position[:,0]-vec.goal,rtol=0,atol=1e-7), 'terminal observer captures pre-reset position')
        require(torch.equal(terminal[0]['reward'],vec.reward), 'terminal reward survives reset')
        require(torch.equal(vec.sim.position[vec.done],vec.start[vec.done]), 'failed particles reset to configured starts')
        require(torch.equal(vec.sim.position[~vec.done],pre_position[~vec.done]), 'healthy particle states are preserved')
        require(not vec.sim.failed.any() and torch.all(vec.steps[vec.done]==0), 'failure counters clear after observer')
        require(torch.all(vec.sim.velocity[vec.done]==0) and torch.all(vec.sim.motors[vec.done]==0), 'failed state resets velocities/motors')
        require(vec.sim.step_calls[0]['dt']==.001 and vec.sim.step_calls[0]['substeps']==20, 'production integration arguments reach fake backend')
        obs,reward,done,truncated,*rest=vec.recv()
        require(obs.shape==(8,98) and torch.equal(reward,expected[0]) and torch.equal(done,vec.done) and not truncated.any(), 'Puffer recv preserves terminal reward and flags')
        # Existing14/15 reach their exact old helper in the same real send path.
        old_vector={}
        for reward_version,helper_version in ((14,1),(15,2)):
            legacy_vec=vector(3,reward_version)
            legacy_vec.sim.position[:,0,:2]=real_tensor([[0,0],[.2,.3],[1.2,0]],dtype=torch.float64)
            old_expected=legacy_reward(legacy_vec.sim.position,legacy_vec.sim.velocity,legacy_vec.sim.orientation,legacy_vec.start[:,0],legacy_vec.sim.failed,version=helper_version)[0]
            legacy_vec.send(np.zeros((3,4),dtype=np.float32))
            require(torch.equal(legacy_vec.reward,old_expected), 'real CudaVector preserves R14/15 dispatch')
            old_vector[str(reward_version)] = True
        # Timeout at500 is separate from the boundary failure exercised above.
        timeout_vec=vector(1,16);timeout_vec.steps[:]=499
        timeout_vec.send(np.zeros((1,4),dtype=np.float32))
        require(timeout_vec.done.item() and timeout_vec.truncated.item() and timeout_vec.steps.item()==0, '500-step timeout stays a timeout and resets')
        numerical_vec=vector(1,16);numerical_vec.sim.failed[:]=2
        try:numerical_vec.send(np.zeros((1,4),dtype=np.float32))
        except FloatingPointError:numerical_rejected=True
        else:raise AssertionError('numerical physics failure must reject trial')
    return {'scope':'Exact current CudaVector.send/obs/recv on CPU fake backend, device-only tensor redirection. No CUDA constructor, physics integration or learned flight. Frozen reference AST extraction executes only pure old reward function.',
            'frozen_reference':{'path':str(reference_path),'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest()},'legacy_reward_function_exact':old_exact,'legacy_vector_dispatch_exact':old_vector,'fine_zoneversion3_helper_exact':True,'invalid_zone_versions_rejected':rejected,'numerical_failure_rejected':numerical_rejected,
            'reward16_vector_rewards':vec.reward.tolist(),'reward16_vector_done':vec.done.tolist(),'reward16_vector_truncated':vec.truncated.tolist(),'terminal_pre_reset_failed':terminal[0]['failed'].tolist(),'terminal_pre_reset_carrier_xy':terminal[0]['obs'][:,:2].tolist(),'reset_carrier_positions':vec.sim.position[:,0].tolist(),'zone_counts':vec.zone_counts,'timeout_preservation_pass':True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = []
    distances = torch.tensor([0., .001, .002, .005, .01, .02, .05, .1], dtype=torch.float64)
    graded = {}
    for axis, label in ((0, 'x'), (1, 'y'), (2, 'height')):
        rewards = []
        for sign in (1, -1):
            inputs = states(len(distances))
            inputs[0][:, 0, axis] += sign * distances
            raw = fine_zone_reward(*inputs)[0]
            clipped = raw.clamp(-1, 1)
            require(torch.all(clipped[:-1] > clipped[1:]), label + ' mm/cm strict clipped improvement')
            require(torch.all(raw <= 1), label + ' no upper clipping plateau')
            require(torch.equal(raw, clipped), label + ' near-hover unaffected by hard clip')
            rewards.append(raw)
        require(torch.equal(rewards[0], rewards[1]), label + ' positive/negative distance symmetry')
        graded[label] = {'distance_m': distances.tolist(), 'raw_reward': rewards[0].tolist(), 'clipped_reward': rewards[0].clamp(-1, 1).tolist()}
        checks.append(label + '_millimetre_centimetre_gradation_and_signs')

    # Existing production reward15 pays>1 near hover, hiding all sub-centimetre
    # improvement from the actual learner, which clamps incoming rewards to[-1,1].
    old_plateau = {}
    small = torch.tensor([0., .001, .005, .01], dtype=torch.float64)
    for axis, label in ((0, 'x'), (1, 'y'), (2, 'height')):
        inputs = states(len(small))
        inputs[0][:, 0, axis] += small
        raw = zone_hover_reward(*inputs, version=2)[0]
        require(abs(float(raw[0]) - 1.05) < 1e-6, 'production15 center raw1.05')
        require(torch.equal(raw.clamp(-1, 1), torch.ones_like(raw)), 'production15 actual clipped subcm plateau')
        old_plateau[label] = {'distance_m': small.tolist(), 'raw_reward': raw.tolist(), 'clipped_reward': raw.clamp(-1, 1).tolist()}
    checks.append('actual_production15_upper_clip_plateau_demonstrated')

    speeds = torch.tensor([0., .01, .05, .1, .2, .5, 1.], dtype=torch.float64)
    inputs = states(len(speeds))
    inputs[1][:, 0, 0] = speeds
    speed_reward = fine_zone_reward(*inputs)[0].clamp(-1, 1)
    require(torch.all(speed_reward[:-1] > speed_reward[1:]), 'speed increase reduces reward')
    # A pure1m/s speed hascost2; at center/height target, reward=.95/3+.05.
    require(abs(float(speed_reward[-1]) - (.95/3+.05)) < 1e-7, 'speed coefficient and m/s units')
    require(abs(float(soft_absolute(torch.tensor(1.,dtype=torch.float64))) - ((1+.01**2)**.5-.01)) < 1e-12, 'softabsolute metre units')
    checks.append('speed_penalty_and_metres_units')

    xy = torch.tensor([[0,0],[1,0],[-1,0],[0,1],[.9,.9],[1,1],[1.0001,0],[-1.5,0],[0,1.9999],[2,0],[-2,0],[0,2],[2,2]], dtype=torch.float64)
    inputs = states(len(xy)); inputs[0][:,0,:2] = xy
    reward, escaped, band = fine_zone_reward(*inputs)
    require(escaped.tolist() == [False]*9+[True]*4, 'square failure exactly2m')
    require(band.tolist() == [False]*6+[True]*3+[False]*4, 'square negative band1to2m')
    require(reward[4] > 0, 'square corner0.9,0.9 remains inside')
    require(reward[1] == 0 and reward[2] == 0 and reward[3] == 0 and reward[5] == 0, 'airborne1m boundary exactly0')
    require(torch.all(reward[6:9] < 0), 'all band rewards negative')
    require(reward[6] > reward[7] > reward[8], 'band distance gradation')
    require(torch.all(reward[9:] == -1), 'all exact2m escapes penalized')
    checks.append('square_corners_band_signs_and_true_failure_boundary')

    epsilon = 1e-6
    inputs = states(6)
    inputs[0][:,0,0] = torch.tensor([1-epsilon,1,1+epsilon,2-epsilon,2,2+epsilon], dtype=torch.float64)
    boundary = fine_zone_reward(*inputs)[0]
    require(abs(float(boundary[0]-boundary[1])) < 2*epsilon, 'airborne1m left continuity')
    require(abs(float(boundary[2]-boundary[1])) < 2*epsilon, 'airborne1m right continuity')
    require(abs(float(boundary[3]-boundary[4])) < 2*epsilon, 'airborne2m left continuity')
    require(boundary[4] == boundary[5] == -1, 'airborne2m outside continuity')
    checks.append('airborne_boundary_continuity')

    inputs = states(5)
    inputs[0][:,0,2] = .05
    inputs[0][:,0,0] = torch.tensor([0,.9,1.5,1.9999,2], dtype=torch.float64)
    ground = fine_zone_reward(*inputs)[0]
    require(torch.all(ground < 0), 'ground negative everywhere')
    require(ground[2] < -1 and ground[2].clamp(-1,1) == -1, 'ground band lower clip limitation explicitly retained')
    inputs = states(4); inputs[4][:] = torch.tensor([0,1,2,3], dtype=torch.int32)
    failures = fine_zone_reward(*inputs)[0]
    require(failures[0] == 1 and torch.all(failures[1:] == -1), 'true failure override independent of goal closeness')
    checks.append('negative_ground_and_real_failure_override')

    # Upright remains rewarded, but angular-rate channels have no reward cost.
    inputs = states(2); inputs[2][1,4:7] = torch.tensor([100.,-100.,50.],dtype=torch.float64)
    require(fine_zone_reward(*inputs)[0][0] == fine_zone_reward(*inputs)[0][1], 'angular rates unpenalized')
    inputs[2][1,0] = 0.; inputs[2][1,1] = 1.
    require(fine_zone_reward(*inputs)[0][1] < fine_zone_reward(*inputs)[0][0], 'upright coefficient retained')
    checks.append('upright_retained_and_angular_rate_penalty_zero')

    inputs = states(4)
    inputs[0][:,0,:2] = torch.tensor([[.001,-.002],[.05,.01],[.9,.9],[1.3,-.1]], dtype=torch.float64)
    originals = tuple(x.clone() for x in inputs)
    result = fine_zone_reward(*inputs)
    require(all(torch.equal(a,b) for a,b in zip(inputs,originals)), 'no input mutation')
    require(all(torch.isfinite(x).all() for x in result), 'finite production helper outputs')
    require(result[0].dtype == torch.float32 and result[1].dtype == torch.bool and result[2].dtype == torch.bool, 'production return dtypes')
    shifted = tuple(x.clone() for x in inputs)
    shifted[0][:,:,0] += 5.; shifted[0][:,:,1] -= 3.
    shifted[3][:,0] += 5.; shifted[3][:,1] -= 3.
    require(torch.allclose(fine_zone_reward(*shifted)[0],result[0],rtol=0,atol=1e-7), 'reward follows takeoff-relative XY, not world origin')
    checks.append('finite_dtype_translation_and_input_immutability')

    # Formula parity through an independently expanded scalar calculation,
    # including ground/band/failure cases rather than only the helper internals.
    cases = [(0.,0.,.8,0.,0),(.005,-.01,.805,.1,0),(.9,.9,.75,.25,0),(1.3,.1,.8,.2,0),(1.3,.1,.05,0.,0),(.01,.01,.8,0.,1),(2.,0.,.8,0.,0)]
    expected=[];actual=[]
    for x,y,z,speed,failed in cases:
        inputs=states(1);inputs[0][0,0,:]=torch.tensor([x,y,z],dtype=torch.float64);inputs[1][0,0,0]=speed;inputs[4][0]=failed
        smooth=lambda d:(d*d+.01**2)**.5-.01
        d=max(abs(x),abs(y));a=min(1,max(0,(z-.05)/.25));c=4*smooth(z-.8)+smooth(x)+smooth(y)+2*speed*speed;core=.95*a/(1+c)+.05;g=.8*(z<.15)
        value=-1. if failed or d>=2 else (-(d-1)-g if d>1 else (1-d)*core-g)
        measured=float(fine_zone_reward(*inputs)[0][0]);require(abs(measured-value)<1e-7,'independent scalar formula parity')
        expected.append(value);actual.append(measured)
    checks.append('independent_scalar_formula_parity')

    root=Path(__file__).resolve().parents[1]
    integration=integration_checks(root)
    checks.append('production_R14_R15_frozen_preservation_R16_dispatch_true_failure_and_autoreset')
    result={'integration':integration,'scope':'CPU-only production pure reward helpers; no simulator initialization, GPU, training, configuration or review edits. Exact unchanged old reward15 compared with new fine reward16 candidate. Fine resolution near hover does not establish learned hover or hardware precision. Ground band can still clip below-1.','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':checks,'graded_distance':graded,'production_reward15_plateau':old_plateau,'speed_m_s':speeds.tolist(),'speed_reward':speed_reward.tolist(),'square_xy_m':xy.tolist(),'square_rewards':reward.tolist(),'square_escaped':escaped.tolist(),'square_band':band.tolist(),'airborne_boundary_rewards':boundary.tolist(),'ground_rewards':ground.tolist(),'scalar_reference_rewards':expected,'scalar_production_rewards':actual,'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tether_sim/fine_zone_reward.py',root/'tether_sim/cuda_task.py',Path(__file__).resolve())}}
    if args.output:
        path=Path(args.output)
    else:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path=root/'tmp/deep_audit_learner'/f'fine_reward_cpu_checks_{stamp}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'checks_passed':len(checks),'evidence':str(path),'near_hover_raw_max':max(max(x['raw_reward']) for x in graded.values()),'production15_center_raw':old_plateau['height']['raw_reward'][0],'production15_clipped_plateau':old_plateau['height']['clipped_reward'],'height1mm_gain':graded['height']['raw_reward'][0]-graded['height']['raw_reward'][1],'x1mm_gain':graded['x']['raw_reward'][0]-graded['x']['raw_reward'][1]},indent=2))


if __name__=='__main__':
    main()
