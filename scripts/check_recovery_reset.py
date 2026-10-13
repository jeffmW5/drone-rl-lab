"""CPU-only reset6 distribution and production reset-cycle regression tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.recovery_reset import initial_orientations


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected invalid reset specification to fail')


counts = {}
for count in (1,2,3,7,8,9,128,129):
    start = torch.arange(count*42,dtype=torch.float64).reshape(count,14,3) / 100
    original = start.clone(); rng = torch.get_rng_state().clone()
    q5 = initial_orientations(start,5); q6 = initial_orientations(start,6)
    assert torch.equal(start,original) and torch.equal(rng,torch.get_rng_state())
    assert q5.shape == q6.shape == (count,7)
    assert q6.dtype == torch.float32 and q6.device == start.device
    assert not q6.requires_grad and torch.isfinite(q6).all()
    identity = torch.zeros(count,7); identity[:,0] = 1
    assert torch.equal(q5,identity) and torch.equal(q6[::2],identity[::2])
    torch.testing.assert_close(q6[:,:4].square().sum(1),torch.ones(count),rtol=0,atol=1.2e-7)
    assert torch.count_nonzero(q6[:,4:]) == 0 and torch.count_nonzero(q6[:,3]) == 0
    observed = []
    for warm_case, lane in enumerate(range(1,count,2)):
        axis = 1 if warm_case%4<2 else 2
        sign = 1 if warm_case%2==0 else -1
        angle = 2*math.atan2(float(q6[lane,axis]),float(q6[lane,0]))
        assert abs(angle-sign*math.radians(2)) < 2e-9
        assert torch.count_nonzero(q6[lane,1:3]) == 1
        # Quaternion body-Z direction validates signed roll/pitch construction.
        w,x,y,z = q6[lane,:4]
        thrust_x = 2*(x*z+w*y); thrust_y = 2*(y*z-w*x)
        if axis == 1:
            assert float(thrust_y)*sign < 0 and float(thrust_x) == 0
        else:
            assert float(thrust_x)*sign > 0 and float(thrust_y) == 0
        observed.append(warm_case%4)
    case_counts = [observed.count(case) for case in range(4)]
    assert sum(case_counts)==count//2 and max(case_counts)-min(case_counts)<=1
    counts[str(count)] = case_counts
rejected(lambda: initial_orientations(torch.zeros(1,14,3),7))
rejected(lambda: initial_orientations(torch.zeros(1,13,3),6))
rejected(lambda: initial_orientations(torch.zeros(0,14,3),6))


class CpuTorchProxy:
    def __getattr__(self,name):
        value = getattr(torch,name)
        if name in ('zeros','empty','ones','full','tensor','arange','as_tensor','rand','randn'):
            def create(*args,**kwargs):
                if kwargs.get('device') == 'cuda':kwargs['device'] = 'cpu'
                return value(*args,**kwargs)
            return create
        return value


class FakeSim:
    def __init__(self,count,library=None):
        self.count = count
        self.position = torch.zeros(count,14,3,dtype=torch.float64)
        self.velocity = torch.zeros_like(self.position)
        self.orientation = torch.zeros(count,7)
        self.motors = torch.zeros(count,4)
        self.mass = torch.tensor([.0434]+[.001/12]*12+[.005]).repeat(count,1)
        self.params = torch.tensor([.6,.025,.2,.015,.000025,.000025,.00004,0,0]).repeat(count,1)
        self.wind = torch.zeros(count,3); self.boxes = torch.empty(count,0,6)
        self.failed = torch.zeros(count,dtype=torch.int32)
        self.reset()
    def reset(self):
        self.position.zero_(); self.position[:,:,2] = 2-torch.arange(14)*.6/13
        self.velocity.zero_(); self.orientation.zero_(); self.orientation[:,0] = 1
        self.motors[:] = self.mass.sum(1,keepdim=True)*9.81/4
        self.failed.zero_()
    def step(self,actions,**kwargs):
        return self.position,self.failed


def load_task(path,name):
    spec = importlib.util.spec_from_file_location('tether_sim.'+name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.torch = CpuTorchProxy(); module.CudaTether = FakeSim
    return module


current_path = Path('tether_sim/cuda_task.py')
legacy_path = Path('state/cuda_research/methodology/review_0109/recovery_curriculum/cuda_task_before_reset6_source.py')
current = load_task(current_path,'recovery_task_current')
legacy = load_task(legacy_path,'recovery_task_legacy')


def vector(task,version):
    return task.CudaVector(16,direct=True,reward_version=15,phase='takeoff',
                           control_version=2,physics_version=5,training_reset_version=version,
                           diagnostic_zone_version=1,integration_version=1,termination_version=1)


def fields(v):
    result = {name:getattr(v,name).clone() for name in (
        'steps','goal','reward','done','truncated','previous_error','start','reset_motor_forces','allocation')}
    result.update({'sim.'+name:getattr(v.sim,name).clone() for name in (
        'position','velocity','orientation','motors','mass','params','wind','boxes','failed')})
    result['obs'] = v.obs().clone()
    return result


def identical(first,second,excluded=()):
    assert first.keys()==second.keys()
    for name in first:
        if name not in excluded:
            assert torch.equal(first[name],second[name]), 'State differs: '+name


legacy_checks = []
for version in range(1,6):
    old = vector(legacy,version); new = vector(current,version)
    old.async_reset(51011); old_initial = fields(old); old_rng = torch.get_rng_state().clone()
    new.async_reset(51011); identical(old_initial,fields(new))
    assert torch.equal(old_rng,torch.get_rng_state())
    # Force a real failure and a timeout; call unmodified production send with
    # no-op physics, allowing its actual observer/reset branches to execute.
    for v in (old,new):
        v.sim.orientation[:] = torch.tensor([.5,.5,.5,.5,.3,.4,.5])
        v.sim.velocity.fill_(.1); v.sim.motors.fill_(.17)
        v.sim.failed[1] = 1; v.steps[3] = 499
        v.send(torch.zeros(16,4).numpy())
    identical(fields(old),fields(new))
    assert old.done[1] and old.truncated[3]
    legacy_checks.append(dict(version=version,initial_states_equal=True,rng_equal=True,
                              failure_timeout_reset_states_equal=True))

v5 = vector(current,5); v6 = vector(current,6)
v5.async_reset(51012); state5 = fields(v5); rng5 = torch.get_rng_state().clone()
v6.async_reset(51012); state6 = fields(v6)
assert torch.equal(rng5,torch.get_rng_state())
identical(state5,state6,excluded=('sim.orientation','obs'))
assert torch.equal(state5['obs'][:,:84],state6['obs'][:,:84])
assert torch.equal(state5['obs'][:,91:],state6['obs'][:,91:])
expected = initial_orientations(v6.start,6)
assert torch.equal(v6.sim.orientation,expected)
assert torch.equal(v6.sim.orientation[::2],v5.sim.orientation[::2])
# Full particle geometry and initialized motor force are identical5/6. Warm
# forces retain weight/4 rather than compensating for the deliberate tilt.
links = torch.linalg.vector_norm(v6.start[:,1:]-v6.start[:,:-1],dim=2)
torch.testing.assert_close(links,torch.full_like(links,.6/13),rtol=0,atol=2e-16)
assert torch.count_nonzero(v6.sim.motors[::2]) == 0
warm_force = v6.sim.mass[1::2].sum(1,keepdim=True)*9.81/4
assert torch.equal(v6.sim.motors[1::2],warm_force.expand(-1,4))
assert torch.equal(v6.start[1::2,0,2],torch.full((8,),.8,dtype=torch.float64))
assert torch.count_nonzero(v6.sim.velocity) == 0
# Alter all orientations, force both true failures and timeout-only events,
# and verify only done lanes regain the reset6 orientations.
v6.sim.orientation[:] = torch.tensor([.5,.5,.5,.5,.3,.4,.5])
v6.sim.velocity.fill_(.1); v6.sim.motors.fill_(.17)
unchanged_orientation = v6.sim.orientation.clone()
v6.sim.failed[0] = 1; v6.sim.failed[1] = 3; v6.sim.failed[3] = 1
v6.steps[5] = 499; v6.steps[7] = 499
observed = []
def observer(obs,done,truncated,failed,reward):
    observed.append(dict(orientation=obs[:,84:91].clone(),done=done.clone(),
                         truncated=truncated.clone(),failed=failed.clone()))
v6.terminal_observer = observer
rng = torch.get_rng_state().clone()
v6.send(torch.zeros(16,4).numpy())
assert torch.equal(rng,torch.get_rng_state())
mask = v6.done
assert torch.nonzero(mask).flatten().tolist() == [0,1,3,5,7]
assert v6.truncated[5] and v6.truncated[7]
assert torch.equal(observed[0]['orientation'],unchanged_orientation)
assert torch.equal(v6.sim.orientation[mask],expected[mask])
assert torch.equal(v6.sim.orientation[~mask],unchanged_orientation[~mask])
assert torch.equal(v6.sim.position[mask],v6.start[mask])
assert torch.count_nonzero(v6.sim.velocity[mask]) == 0
assert torch.equal(v6.sim.motors[mask],v6.reset_motor_forces[mask])
assert torch.count_nonzero(v6.steps[mask]) == 0 and torch.count_nonzero(v6.sim.failed) == 0
# Explicit async-reset also restores every lane, without persistent learned
# commands or rates leaking across resets.
v6.async_reset(51013)
assert torch.equal(v6.sim.orientation,expected) and torch.count_nonzero(v6.sim.velocity)==0
assert torch.equal(v6.sim.motors,v6.reset_motor_forces)
assert not torch.cuda.is_initialized()

out = Path('tmp/deep_audit_experiment/recovery_reset_cpu_test.json')
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():
    for revision in range(2,1000):
        candidate = out.with_name(f'recovery_reset_cpu_test_v{revision:03d}.json')
        if not candidate.exists():out=candidate;break
    else:raise FileExistsError('Evidence suffix range exhausted')
result = dict(scope='CPU helper and actual production reset logic with no-op physics; no CUDA, dynamics or learned flight test.',
              cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=False,
              helper_sha256=hashlib.sha256(Path('tether_sim/recovery_reset.py').read_bytes()).hexdigest(),
              task_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
              pre_reset6_task_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              passed=True,warm_case_counts_by_num_agents=counts,
              old_versions_regression=legacy_checks,version5_6_geometry_motors_rng_equal=True,
              reset6_orientation_only=True,reset6_failure_timeout_lanes=[0,1,3,5,7],
              checks=['unit quaternions and signed roll/pitch axes','balanced four-case warm coverage',
                      'FP32 orientation with start-device allocation','finite state and no input mutation/RNG',
                      'old versions1..5 initial and failure/timeout reset parity',
                      'full taut-chain geometry and motor parity5vs6',
                      'pre-reset callback followed by restored orientation and zero rates',
                      'only done lanes reset','async reset restores all lanes'],
              counterevidence='Review0080 rejected warm motor initialization: worse three ground-evaluation pairs, zero hover. Reset6 separately tests signed orientation exposure, not a proven fix.',
              missing_evidence='Real GPU training/evaluation, perturbation recovery and longer uninterrupted ground-flight outcomes remain pending.')
out.write_text(json.dumps(result,indent=2,allow_nan=False))
print('Recovery reset CPU helper/production tests passed; evidence saved to '+str(out))

