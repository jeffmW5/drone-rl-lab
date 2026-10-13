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

rows=[]
for version in (1,2):
 v=task.CudaVector(2,direct=True,reward_version=14,phase='takeoff',control_version=2,physics_version=5,training_reset_version=3,diagnostic_zone_version=1,termination_version=version);v.async_reset(51011);v.steps[:]=499;v.sim.position[0,0]=torch.tensor([2.,0.,.8]);events=[]
 v.terminal_observer=lambda obs,done,truncated,failed,reward:events.append({'done':done.clone(),'truncated':truncated.clone(),'failed':failed.clone(),'reward':reward.clone()})
 v.send(torch.zeros(2,4).numpy());e=events[0]
 assert bool(e['done'].all()) and int(e['failed'][0])==3 and int(e['failed'][1])==0
 assert bool(e['truncated'][1]) and bool(e['truncated'][0])==(version==1)
 assert float(e['reward'][0])==-1 and torch.equal(v.sim.position,v.start)
 rows.append({'version':version,'physical_failure_counted':bool(v.done[0]&~v.truncated[0]),'true_timeout_retained':bool(v.truncated[1]),'terminal_reward_preserved':True})
(OUT/'failure_priority_production_test.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
