import ctypes, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from tether_sim.cuda_task import CudaPolicy

torch.set_num_threads(1)
ROOT=Path('/home/n33du/ai/drone-rl-lab')
OUT=ROOT/'tmp/deep_audit_physics'
OUT.mkdir(exist_ok=True)

def cpu_kernel(version):
    original=ROOT/f'tether_sim/cuda/physics_v{version:03d}.cu'
    source=original.read_text()
    old='int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=n||failed[e])return;'
    assert source.count(old)==1
    rewritten=source.replace('__global__ ','').replace(old,'int e=0; if(e>=n||failed[e])return;')
    cpp=OUT/f'physics_v{version:03d}_cpu.cpp'
    cpp.write_text(rewritten)
    so=cpp.with_suffix('.so')
    subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
    library=ctypes.CDLL(str(so));function=library.step_kernel
    function.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_int,ctypes.c_int]
    return function,dict(version=version,original_path=str(original),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),translation_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),scope='Serial CPU translation of unchanged source arithmetic, differs from CUDA arithmetic; removes CUDA qualifier and sets environment index0 only')

class Sim:
    def __init__(self,function,start='hover'):
        self.function=function
        self.p=np.zeros((14,3),dtype=np.float64);self.v=np.zeros((14,3),dtype=np.float64)
        self.q=np.zeros(7,dtype=np.float32);self.q[0]=1
        self.motors=np.zeros(4,dtype=np.float32)
        self.mass=np.array([.0434]+[.001/12]*12+[.005],dtype=np.float32)
        self.params=np.array([.6,.025,.2,.015,.000025,.000025,.00004,0,1],dtype=np.float32)
        self.wind=np.zeros(3,dtype=np.float32);self.boxes=np.empty((0,6),dtype=np.float32);self.failed=np.zeros(1,dtype=np.int32)
        if start=='hover':
            self.p[:,2]=.8-np.arange(14,dtype=np.float64)*.6/13
            self.motors[:]=np.float32(torch.tensor(self.mass).sum().item()*9.81/4)
        elif start=='ground':
            f=np.arange(14,dtype=np.float64)/13
            self.p[:,0]=f*math.sqrt(.6*.6-.025*.025);self.p[:,2]=.05-f*.025
        self.equilibrium=np.float32(torch.tensor(self.mass).sum().item()*9.81/(4*float(self.params[2])))
    def step(self,action,dt=.001,substeps=20,iterations=32):
        a=np.asarray(action,dtype=np.float32)
        assert a.shape==(4,) and a.flags.c_contiguous
        self.function(*[x.ctypes.data for x in (self.p,self.v,self.q,self.motors,a,self.mass,self.params,self.wind,self.boxes,self.failed)],1,0,dt,substeps,iterations)
    def observation(self):
        goal=np.array([0,0,.2]);return torch.from_numpy(np.concatenate(((self.p-goal).ravel(),self.v.ravel(),self.q,self.motors,goal)).astype(np.float32))[None]
    def energy(self):return float(np.sum(.5*self.mass*np.sum(self.v**2,axis=1))+np.sum(self.mass*9.81*self.p[:,2]))

MIX=np.array([[1,-1,-1],[1,1,1],[-1,1,-1],[-1,-1,1]],dtype=np.float32)
ALLOCATION=np.array([[1,1,1,1],[.032,.032,-.032,-.032],[-.032,.032,.032,-.032],[-.006,.006,-.006,.006]])
functions={};sources=[]
for version in (5,6):
    functions[version],meta=cpu_kernel(version);sources.append(meta)
result=dict(scope='CPU-only independent arithmetic/control audit. Production sources/checkpoints read-only; no GPU/context/trainer/benchmark modification. Counterfactuals are diagnostics, not proposed production controllers.',sources=sources,allocation=ALLOCATION.tolist(),mix=MIX.tolist(),torque_per_action=(ALLOCATION[1:]@(.2*.02*MIX)).tolist(),physical_checks=[],replays=[])
assert np.allclose(ALLOCATION[1:]@MIX,np.diag([.128,.128,.024]))

# Positive/negative allocation and authority; steady versus25ms motor lag.
for version in (5,6):
    for channel in (1,2,3):
        for sign in (1,-1):
            s=Sim(functions[version]);u=np.array([float(2*s.equilibrium-1),0,0,0]);u[channel]=sign
            command=np.clip(.5*(u[0]+1)+.02*(u[1:]@MIX.T),0,1)
            s.step(command)
            result['physical_checks'].append(dict(kind='20ms_max_channel_pulse',physics=version,channel=channel,sign=sign,q=s.q.tolist(),carrier_velocity=s.v[0].tolist(),motor_N=s.motors.tolist(),failed=int(s.failed[0])))
    for dt,substeps in ((.001,20),(.0005,40)):
        s=Sim(functions[version]);s.params[3]=0
        first=s.p.copy();initial_energy=s.energy();peak_velocity=0
        for _ in range(250):
            s.step(np.full(4,s.equilibrium),dt,substeps)
            peak_velocity=max(peak_velocity,float(np.linalg.norm(s.v[0])))
        result['physical_checks'].append(dict(kind='taut_equilibrium5s',physics=version,dt=dt,carrier_delta=(s.p[0]-first[0]).tolist(),payload_delta=(s.p[-1]-first[-1]).tolist(),peak_carrier_speed=peak_velocity,energy_delta=s.energy()-initial_energy,failed=int(s.failed[0])))
    s=Sim(functions[version],start='ground');peak=.05
    for _ in range(50):
        s.step(np.zeros(4));peak=max(peak,float(s.p[0,2]))
    result['physical_checks'].append(dict(kind='no_thrust_ground1s',physics=version,peak_carrier_height=peak,carrier_velocity=s.v[0].tolist(),energy=s.energy(),failed=int(s.failed[0])))

# Unchanged policy replay and controlled perturbations; same intact tether.
models=[(758,1),(759,2),(761,2),(763,2),(713,2)]
known=json.loads((ROOT/'tmp/cuda-research/cuda_auto_0771_takeoff/zone_horizon.json').read_text())
for trial,obs_version in models:
    checkpoint=ROOT/f'tmp/cuda-research/cuda_auto_{trial:04d}_takeoff/best.pt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    policy=CudaPolicy();policy.load_state_dict(weights);policy.configure_collective_observation(obs_version);policy.eval()
    for start,scenario,physics in [('ground','replay_baseline',5),('hover','initial_exact_hover',5),('hover','lateral_vx_plus0.5',5),('hover','roll_plus10deg',5),('ground','joint_floor_v6_counterfactual',6)]:
        s=Sim(functions[physics],start=start)
        if scenario=='lateral_vx_plus0.5':s.v[:,0]=.5
        if scenario=='roll_plus10deg':s.q[0]=math.cos(math.radians(5));s.q[1]=math.sin(math.radians(5))
        frames=[];hold=0;peak_hold=0;boundary=None;landing=None;takeoff=False;peak_height=.05;total_commands=[]
        for step in range(1000):
            with torch.no_grad():dist,_=policy(s.observation());u=dist.mean[0].numpy()
            command=np.clip(.5*(u[0]+1)+.02*(u[1:]@MIX.T),0,1)
            s.step(command);t=(step+1)*.02;extent=float(np.max(np.abs(s.p[0,:2])));speed=float(np.linalg.norm(s.v[0]));height=float(s.p[0,2])
            stable=extent<=1 and abs(height-.8)<.1 and speed<.2 and not s.failed[0]
            hold=hold+.02 if stable else 0;peak_hold=max(peak_hold,hold);takeoff|=height>.3;peak_height=max(peak_height,height)
            if takeoff and height<=.051 and landing is None:landing=t
            if step%25==0:frames.append(dict(t=t,p=s.p[0].tolist(),v=s.v[0].tolist(),q=s.q.tolist(),action=u.tolist(),motor=s.motors.tolist()))
            if extent>=2:boundary=t;break
            if s.failed[0]:break
        row=dict(trial=trial,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),collective_observation_version=obs_version,start=start,scenario=scenario,physics=physics,elapsed_s=t,boundary_failure_s=boundary,kernel_failure=int(s.failed[0]),landing_s=landing,takeoff=takeoff,peak_height=peak_height,peak_hover_hold=peak_hold,final_p=s.p[0].tolist(),final_v=s.v[0].tolist(),frames=frames)
        if scenario=='replay_baseline':
            known_row=next(x for x in known['policies'] if x['checkpoint'].endswith(f'cuda_auto_{trial:04d}_takeoff/best.pt') and x['collective_observation_version']==obs_version)
            row['gpu_diagnostic_elapsed_s']=known_row['elapsed_simulated_s'];row['cpu_gpu_failure_time_difference_s']=t-known_row['elapsed_simulated_s']
            kframes={round(x['time_s'],5):x for x in known_row['frames']}
            errors=[]
            for frame in frames:
                if round(frame['t'],5) in kframes:errors.append(float(np.max(np.abs(np.array(frame['p'])-np.array(kframes[round(frame['t'],5)]['carrier'])))))
            row['max_shared_frame_carrier_difference_m']=max(errors,default=None)
        result['replays'].append(row)
        print(json.dumps({k:row[k] for k in ('trial','scenario','physics','elapsed_s','peak_hover_hold','final_p','final_v','landing_s')}) ,flush=True)
(OUT/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(result_path=str(OUT/'result.json'),source_hashes=sources)),flush=True)
