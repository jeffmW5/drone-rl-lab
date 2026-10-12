"""Bounded experimental C/CUDA physics + PufferLib PPO, with live viewer."""
import argparse,json,time,sys,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from tether_sim.cuda_task import CudaVector,CudaPolicy
from tether_sim.training_lock import acquire_training_lease

parser=argparse.ArgumentParser();parser.add_argument('--output',default='tmp/cuda-tether-v001/training_v001');parser.add_argument('--port',type=int,default=8767);parser.add_argument('--direct',action='store_true');parser.add_argument('--initial-throttle',type=float,default=.5);parser.add_argument('--steps',type=int,default=131072);parser.add_argument('--plan');args=parser.parse_args()
plan=json.loads(Path(args.plan).read_text()) if args.plan else {}
for key in ('initial_throttle','steps','direct'):
    if key in plan:setattr(args,key,plan[key])
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
    vec=CudaVector(8,direct=args.direct,reward_version=plan.get('reward_version',1),phase=plan.get('phase','destination'),control_version=plan.get('control_version',1));vec.async_reset(seed);collisions=0;peak_height=torch.zeros(8,device='cuda');hold=torch.zeros(8,device='cuda');peak_hold=hold.clone();trajectory=[]
    for step in range(plan.get('evaluation_steps',250)):
        with torch.no_grad():distribution,_=policy(vec.obs());action=distribution.mean.cpu().numpy()
        vec.send(action)
        peak_height=torch.maximum(peak_height,vec.sim.position[:,0,2].float())
        collisions+=int((vec.done&~vec.truncated).sum())
        distance=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
        stable=(distance<.1)&(torch.linalg.vector_norm(vec.sim.velocity[:,-1],dim=1)<.2)&~vec.done
        hold=torch.where(stable,hold+.02,0);peak_hold=torch.maximum(peak_hold,hold)
        if step%10==0:publish(vec,'deterministic evaluation',step)
        if plan and step%25==0:
            trajectory.append(dict(step=step,positions=vec.sim.position[0].cpu().tolist(),orientation=vec.sim.orientation[0].cpu().tolist(),motors=vec.sim.motors[0].cpu().tolist(),terminal=bool(vec.done[0])))
    error=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
    return dict(seed=seed,mean_final_error_mm=float(error.mean()*1000),errors_mm=(error*1000).tolist(),crashes=collisions,takeoffs=int((peak_height>.3).sum()),hover_successes=int((peak_hold>=1).sum()),peak_hover_hold_s=peak_hold.cpu().tolist(),peak_drone_height_m=peak_height.cpu().tolist(),
      trajectory=trajectory,scope='Prototype evaluation; not original obstacle navigation or sustained1mm benchmark')

trainer=None
try:
    from pufferlib.pufferl import PuffeRL,ADVANTAGE_CUDA
    if not ADVANTAGE_CUDA:raise RuntimeError('CUDA advantage kernel required')
    vec=CudaVector(128,direct=args.direct,reward_version=plan.get('reward_version',1),phase=plan.get('phase','destination'),control_version=plan.get('control_version',1));policy=CudaPolicy().cuda()
    if args.direct and args.initial_throttle!=.5:
        with torch.no_grad():policy.actor.bias.fill_(2*args.initial_throttle-1);policy.log_std.fill_(-4.)
    if plan.get('control_version',1)==2:
        with torch.no_grad():policy.actor.bias.zero_();policy.actor.bias[0]=2*args.initial_throttle-1
    if plan.get('resume'):policy.load_state_dict(torch.load(plan['resume'],weights_only=True))
    config=dict(env='cuda_tether_prototype_v001',seed=41,torch_deterministic=True,cpu_offload=False,device='cuda',optimizer='adam',anneal_lr=False,precision='float32',amp=False,total_timesteps=32768,learning_rate=3e-4,gamma=.995,gae_lambda=.95,update_epochs=3,clip_coef=.2,vf_coef=.5,vf_clip_coef=.2,max_grad_norm=.5,ent_coef=0.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-5,min_lr_ratio=1.,data_dir=str(out/'puffer'),checkpoint_interval=100,batch_size=4096,minibatch_size=1024,max_minibatch_size=1024,bptt_horizon=32,compile=False,compile_mode='default',compile_fullgraph=False,use_rnn=False,vtrace_rho_clip=1.,vtrace_c_clip=1.,prio_alpha=0.,prio_beta0=0.)
    if args.direct:config['total_timesteps']=args.steps
    config['control_mode']='direct_motor_policy' if args.direct else 'PD_plus_residual'
    config['initial_throttle']=args.initial_throttle
    config['learning_rate']=plan.get('learning_rate',config['learning_rate'])
    config['seed']=plan.get('seed',41);torch.manual_seed(config['seed'])
    config['research_plan']=plan
    (out/'config.json').write_text(json.dumps(config,indent=2))
    seeds=plan.get('validation_seeds',[51011])
    def measured():
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        try:values=[evaluate(policy,seed) for seed in seeds]
        finally:torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
        return dict(runs=values,crashes=sum(v['crashes'] for v in values),takeoffs=sum(v['takeoffs'] for v in values),hover_successes=sum(v['hover_successes'] for v in values),mean_final_error_mm=float(np.mean([v['mean_final_error_mm'] for v in values]))) if plan else values[0]
    initial=measured();torch.save(policy.state_dict(),out/'initial.pt')
    best=initial;best_step=0;torch.save(policy.state_dict(),out/'best.pt')
    def rank(value):return (value['crashes'],-value.get('hover_successes',0),-value['takeoffs'],value['mean_final_error_mm'])
    trainer=PuffeRL(config,vec,policy)
    started=time.monotonic()
    next_eval=plan.get('eval_every_steps',65536)
    while trainer.global_step<config['total_timesteps'] and time.monotonic()-started<plan.get('budget_seconds',120):
        trainer.evaluate();trainer.train();publish(vec,'PufferLib training',trainer.global_step)
        (out/'status.json').write_text(json.dumps(dict(status='training',steps=trainer.global_step,elapsed=time.monotonic()-started)))
        if plan and trainer.global_step>=next_eval:
            value=measured();(out/('eval_'+str(trainer.global_step)+'.json')).write_text(json.dumps(value,indent=2))
            torch.save(policy.state_dict(),out/('checkpoint_'+str(trainer.global_step)+'.pt'))
            if rank(value)<rank(best):best=value;best_step=trainer.global_step;torch.save(policy.state_dict(),out/'best.pt')
            next_eval+=plan.get('eval_every_steps',65536)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    torch.save(policy.state_dict(),out/'policy.pt')
    final=measured()
    if rank(final)<rank(best):best=final;best_step=trainer.global_step;torch.save(policy.state_dict(),out/'best.pt')
    result=dict(status='complete',control_mode=config['control_mode'],initial_throttle=args.initial_throttle,steps=trainer.global_step,training_seconds=elapsed,training_transitions_per_second=trainer.global_step/elapsed,initial=initial,final=final,gpu=torch.cuda.get_device_name(0),advantage_cuda=ADVANTAGE_CUDA,
      limitations='Experimental particle tether; no MuJoCo parity, segment contact response, hardware calibration or original1mm success claim. PufferLib3 copies actions through CPU, but physics and learner are GPU; no CPU physics workers.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    (out/'selection.json').write_text(json.dumps(dict(best=best,best_step=best_step,promoted=rank(best)<rank(initial),phase=plan.get('phase','destination')),indent=2))
    (out/'status.json').write_text(json.dumps(dict(status='complete',steps=trainer.global_step)))
    trainer.utilization.stop();trainer=None
    # Live deterministic playback, bounded ten minutes; lease prevents overlap.
    view=CudaVector(1,direct=args.direct);view.async_reset(51011)
    deadline=time.monotonic()+plan.get('playback_seconds',600)
    while time.monotonic()<deadline:
        with torch.no_grad():distribution,_=policy(view.obs())
        view.send(distribution.mean.cpu().numpy());publish(view,'saved policy playback',int(view.steps[0]));time.sleep(.02)
finally:
    if trainer is not None:trainer.utilization.stop()
    server.shutdown();lease.close()
