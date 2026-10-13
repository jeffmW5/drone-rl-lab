import sys, json, hashlib, math
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
import tether_sim.cuda_task as task
OUT=Path('tmp/deep_audit_root')
OUT.mkdir(exist_ok=True)
class CpuTorchProxy:
    def __getattr__(self,name):
        obj=getattr(torch,name)
        if name in ('zeros','empty','tensor','arange','as_tensor','full','ones','rand'):
            def create(*args,**kwargs):
                if kwargs.get('device')=='cuda':kwargs['device']='cpu'
                return obj(*args,**kwargs)
            return create
        return obj
class FakeSim:
    def __init__(self,count,library=None):
        self.count=count
        self.position=torch.zeros(count,14,3,dtype=torch.float64)
        self.velocity=torch.zeros_like(self.position)
        self.orientation=torch.zeros(count,7)
        self.motors=torch.zeros(count,4)
        self.mass=torch.tensor([.0434]+[.001/12]*12+[.005]).repeat(count,1)
        self.params=torch.tensor([.6,.025,.2,.015,.000025,.000025,.00004,0,0]).repeat(count,1)
        self.wind=torch.zeros(count,3)
        self.boxes=torch.empty(count,0,6)
        self.failed=torch.zeros(count,dtype=torch.int32)
        self.reset()
    def reset(self):
        self.position.zero_(); self.position[:,:,2]=2-torch.arange(14)*.6/13
        self.velocity.zero_();self.orientation.zero_();self.orientation[:,0]=1
        self.motors[:]=self.mass.sum(1,keepdim=True)*9.81/4;self.failed.zero_()
    def step(self,actions,**kwargs):return self.position,self.failed
# CPU-only factory substitution; all task logic/methods/functions remain production.
task.torch=CpuTorchProxy(); task.CudaTether=FakeSim
hashes={}
vecs=[]
for seed in (51011,51012,51013):
    v=task.CudaVector(8,direct=True,reward_version=14,phase='takeoff',control_version=2,physics_version=5,training_reset_version=3,diagnostic_zone_version=1)
    v.async_reset(seed)
    tensors=[v.obs(),v.sim.position,v.sim.velocity,v.sim.orientation,v.sim.motors,v.sim.wind,v.sim.boxes,v.goal,v.start,v.sim.mass,v.sim.params]
    hashes[str(seed)]=hashlib.sha256(b''.join(x.numpy().tobytes() for x in tensors)).hexdigest();vecs.append(v)
assert len(set(hashes.values()))==1
assert torch.equal(vecs[0].obs()[0],vecs[0].obs()[1])
term=[]
v=vecs[0]
def observe(obs,done,truncated,failed,reward):
    term.append(dict(failed=failed.tolist(),done=done.tolist(),truncated=truncated.tolist(),carrier_xyz=(obs[:,:3]+v.goal).tolist(),reward=reward.tolist()))
v.terminal_observer=observe
v.sim.position[0,0]=torch.tensor([2.,0.,.8],dtype=torch.float64)
v.send(torch.zeros(8,4).numpy())
assert term[-1]['failed'][0]==3 and term[-1]['done'][0] and not term[-1]['truncated'][0]
assert torch.equal(v.sim.position[0],v.start[0])
escape_normal=term[-1]
v.steps[:]=499
v.sim.position[0,0]=torch.tensor([2.,0.,.8],dtype=torch.float64)
v.send(torch.zeros(8,4).numpy())
escape_timeout=term[-1]
assert escape_timeout['failed'][0]==3 and escape_timeout['done'][0] and escape_timeout['truncated'][0]
result=dict(scope='CPU production task logic with fake no-op physics; not a dynamics, GPU or flight test',source_sha256=hashlib.sha256(Path('tether_sim/cuda_task.py').read_bytes()).hexdigest(),validation_state_hashes=hashes,all_validation_states_identical=True,agents_identical=True,standard_escape=escape_normal,escape_at_timeout=escape_timeout,standard_evaluator_crash_flag_at_timeout=bool(v.done[0] and not v.truncated[0]),default_exposure=dict(transitions=65536,agents=128,steps_per_agent=65536/128,seconds_per_agent=65536/128*.02,batches=65536/4096,optimizer_steps=65536/4096*3*4,complete_max_length_episodes=math.floor((65536/128)/500),fragment_seconds=.02*32,gamma=.995,gae_lambda=.95,gae_decay_steps=1/(1-.995*.95),gae_decay_seconds=.02/(1-.995*.95)))
(OUT/'production_logic_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('standard_escape','escape_at_timeout')},indent=2))
