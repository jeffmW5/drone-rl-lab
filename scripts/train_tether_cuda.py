"""Bounded experimental C/CUDA physics + PufferLib PPO, with live viewer."""
import argparse,json,time,sys,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from tether_sim.cuda_task import CudaVector,CudaPolicy
from tether_sim.training_lock import acquire_training_lease

parser=argparse.ArgumentParser();parser.add_argument('--output',default='tmp/cuda-tether-v001/training_v001');parser.add_argument('--port',type=int,default=8767);parser.add_argument('--direct',action='store_true');parser.add_argument('--initial-throttle',type=float,default=.5);parser.add_argument('--steps',type=int,default=131072);args=parser.parse_args()
if not 0<args.initial_throttle<1 or not 4096<=args.steps<=1048576:parser.error('Invalid bounded experiment settings')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
lease=acquire_training_lease();torch.set_num_threads(1);torch.manual_seed(41)
snapshot={'mode':'initializing','positions':[]};guard=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        if self.path=='/state':
            with guard: content=json.dumps(snapshot).encode()
            mime='application/json'
        elif self.path in ('/','/viewer'):
            content=Path('tether_sim/cuda/viewer.html').read_bytes();mime='text/html'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()

def publish(vec,mode,step):
    s=vec.sim
    with guard:
        snapshot.update(mode=mode+(' · direct learned motors' if args.direct else ' · PD + AI residual'),step=step,positions=s.position[0].cpu().tolist(),goal=vec.goal[0].cpu().tolist(),boxes=s.boxes[0].cpu().tolist(),failed=int(s.failed[0]),gpu=torch.cuda.get_device_name(0))

def evaluate(policy,seed):
    vec=CudaVector(8,direct=args.direct);vec.async_reset(seed);collisions=0;peak_height=torch.zeros(8,device='cuda')
    for step in range(250):
        with torch.no_grad():distribution,_=policy(vec.obs());action=distribution.mean.cpu().numpy()
        vec.send(action)
        peak_height=torch.maximum(peak_height,vec.sim.position[:,0,2].float())
        collisions+=int((vec.done&~vec.truncated).sum())
        if step%10==0:publish(vec,'deterministic evaluation',step)
    error=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
    return dict(seed=seed,mean_final_error_mm=float(error.mean()*1000),errors_mm=(error*1000).tolist(),crashes=collisions,takeoffs=int((peak_height>.3).sum()),peak_drone_height_m=peak_height.cpu().tolist(),
      scope='Prototype five-second destination episode; not original navigation or sustained1mm benchmark')

trainer=None
try:
    from pufferlib.pufferl import PuffeRL,ADVANTAGE_CUDA
    if not ADVANTAGE_CUDA:raise RuntimeError('CUDA advantage kernel required')
    vec=CudaVector(128,direct=args.direct);policy=CudaPolicy().cuda()
    if args.direct and args.initial_throttle!=.5:
        with torch.no_grad():policy.actor.bias.fill_(2*args.initial_throttle-1);policy.log_std.fill_(-4.)
    config=dict(env='cuda_tether_prototype_v001',seed=41,torch_deterministic=True,cpu_offload=False,device='cuda',optimizer='adam',anneal_lr=False,precision='float32',amp=False,total_timesteps=32768,learning_rate=3e-4,gamma=.995,gae_lambda=.95,update_epochs=3,clip_coef=.2,vf_coef=.5,vf_clip_coef=.2,max_grad_norm=.5,ent_coef=0.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-5,min_lr_ratio=1.,data_dir=str(out/'puffer'),checkpoint_interval=100,batch_size=4096,minibatch_size=1024,max_minibatch_size=1024,bptt_horizon=32,compile=False,compile_mode='default',compile_fullgraph=False,use_rnn=False,vtrace_rho_clip=1.,vtrace_c_clip=1.,prio_alpha=0.,prio_beta0=0.)
    if args.direct:config['total_timesteps']=args.steps
    config['control_mode']='direct_motor_policy' if args.direct else 'PD_plus_residual'
    config['initial_throttle']=args.initial_throttle
    (out/'config.json').write_text(json.dumps(config,indent=2))
    initial=evaluate(policy,51011)
    trainer=PuffeRL(config,vec,policy)
    started=time.monotonic()
    while trainer.global_step<config['total_timesteps'] and time.monotonic()-started<120:
        trainer.evaluate();trainer.train();publish(vec,'PufferLib training',trainer.global_step)
        (out/'status.json').write_text(json.dumps(dict(status='training',steps=trainer.global_step,elapsed=time.monotonic()-started)))
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    torch.save(policy.state_dict(),out/'policy.pt')
    final=evaluate(policy,51011)
    result=dict(status='complete',control_mode=config['control_mode'],initial_throttle=args.initial_throttle,steps=trainer.global_step,training_seconds=elapsed,training_transitions_per_second=trainer.global_step/elapsed,initial=initial,final=final,gpu=torch.cuda.get_device_name(0),advantage_cuda=ADVANTAGE_CUDA,
      limitations='Experimental particle tether; no MuJoCo parity, segment contact response, hardware calibration or original1mm success claim. PufferLib3 copies actions through CPU, but physics and learner are GPU; no CPU physics workers.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    trainer.utilization.stop();trainer=None
    # Live deterministic playback, bounded ten minutes; lease prevents overlap.
    view=CudaVector(1,direct=args.direct);view.async_reset(51011)
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        with torch.no_grad():distribution,_=policy(view.obs())
        view.send(distribution.mean.cpu().numpy());publish(view,'saved policy playback',int(view.steps[0]));time.sleep(.02)
finally:
    if trainer is not None:trainer.utilization.stop()
    server.shutdown();lease.close()
