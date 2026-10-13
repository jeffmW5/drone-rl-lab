"""Build and validate physics3 only while caller holds the serial GPU lease."""
import hashlib,json,os,subprocess
from pathlib import Path
import torch
from .cuda_backend import CudaTether

def prepare_and_check(output,physics_version=3):
    if physics_version not in (3,4):raise ValueError("Unsupported checked physics")
    sources=('tether_sim/thrust_order_check.py',f'tether_sim/cuda/physics_v{physics_version:03d}.cu','tether_sim/cuda/physics_v001.cu','tether_sim/cuda/host.c')
    manifest={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    for p in sources: Path(output).parent.joinpath(Path(p).name).write_bytes(Path(p).read_bytes())
    Path(output).with_suffix('.sources.json').write_text(json.dumps(manifest,indent=2))
    build=Path(f'tmp/cuda-tether-v{physics_version:03d}')
    if not build.exists():
        build.mkdir()
        cuda=os.environ.get('CUDA_HOME','/home/n33du/ai/cuda-12.6-puffer')
        subprocess.run([cuda+'/bin/nvcc','-O3','-arch=sm_61','--cubin',f'tether_sim/cuda/physics_v{physics_version:03d}.cu','-o',str(build/'physics.cubin')],check=True)
        subprocess.run(['gcc','-O3','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC','-I'+cuda+'/include','tether_sim/cuda/host.c','-L/usr/lib/wsl/lib','-Wl,-rpath,/usr/lib/wsl/lib','-lcuda','-o',str(build/'libtether.so')],check=True)
        (build/'sources.sha256').write_text(''.join(hashlib.sha256(Path(p).read_bytes()).hexdigest()+'  '+p+'\n' for p in (f'tether_sim/cuda/physics_v{physics_version:03d}.cu','tether_sim/cuda/host.c')))
    sim=CudaTether(7,library=str(build/'libtether.so'))
    sim.params[:,3]=0
    sim.orientation[1,4]=10;sim.orientation[2,4]=-10
    sim.orientation[3,5]=10;sim.orientation[4,5]=-10
    sim.orientation[5,6]=10;sim.orientation[6,6]=-10
    thrust=sim.motors.sum(-1).double().clone()
    actions=(sim.motors/sim.params[:,2:3]).contiguous()
    sim.step(actions,substeps=1)
    q=sim.orientation[:,:4].double();w,x,y,z=q.unbind(-1)
    axis=torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),-1)
    expected=.001*thrust[:,None]*axis
    expected[:,2]-=.001*9.81*sim.mass.double().sum(-1)
    momentum=(sim.mass.double()[:,:,None]*sim.velocity).sum(1)
    error=float((momentum-expected).abs().max())
    assert error<2e-8,(error,momentum,expected)
    assert abs(float(momentum[1,1]))>4e-6,'Old quaternion would produce zero lateral impulse'
    assert float(momentum[1,1])<0 and float(momentum[2,1])>0
    assert float(momentum[3,0])>0 and float(momentum[4,0])<0
    assert float(momentum[5,:2].abs().max())<1e-10
    assert float((q.norm(dim=-1)-1).abs().max())<1e-6
    assert not bool(sim.failed.any())
    # Zero rotation must match the immutable baseline exactly within numerical tolerance.
    old=CudaTether(1);old.params[:,3]=0
    old.step(actions[:1].contiguous(),substeps=1)
    stationary=float((old.position[0]-sim.position[0]).abs().max())
    assert stationary<1e-12,stationary
    result=dict(status='passed',physics_version=physics_version,max_impulse_error_ns=error,zero_rotation_baseline_difference_m=stationary,checks='Positive/negative roll and pitch, yaw, quaternion normalization, updated-axis total impulse, no-rotation baseline',scope='One-substep full-tether regression; not learned navigation success or numerical convergence')
    Path(output).write_text(json.dumps(result,indent=2))
    return result

