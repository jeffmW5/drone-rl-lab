"""Bounded experimental C/CUDA physics + PufferLib PPO, with live viewer."""
import argparse,json,time,sys,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from tether_sim.cuda_task import CudaVector,CudaPolicy,configure_exploration
from tether_sim.training_lock import acquire_training_lease

parser=argparse.ArgumentParser();parser.add_argument('--output',default='tmp/cuda-tether-v001/training_v001');parser.add_argument('--port',type=int,default=8767);parser.add_argument('--direct',action='store_true');parser.add_argument('--initial-throttle',type=float,default=.5);parser.add_argument('--steps',type=int,default=131072);parser.add_argument('--plan');args=parser.parse_args()
plan=json.loads(Path(args.plan).read_text()) if args.plan else {}
for key in ('initial_throttle','steps','direct'):
    if key in plan:setattr(args,key,plan[key])
if not 0<args.initial_throttle<1 or not 4096<=args.steps<=1048576:parser.error('Invalid bounded experiment settings')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
lease=acquire_training_lease();torch.set_num_threads(1);torch.manual_seed(41)
if plan.get("physics_version",1)==3:
    from tether_sim.thrust_order_check import prepare_and_check
    prepare_and_check(out/"thrust_order_checks.json")
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
    vec=CudaVector(8,direct=args.direct,reward_version=plan.get('reward_version',1),phase=plan.get('phase','destination'),control_version=plan.get('control_version',1),physics_version=plan.get('physics_version',1));vec.async_reset(seed);collisions=0;peak_height=torch.zeros(8,device='cuda');hold=torch.zeros(8,device='cuda');peak_hold=hold.clone();trajectory=[]
    acquisition_hold=torch.zeros(8,device='cuda');peak_acquisition=acquisition_hold.clone()
    airborne_hold=acquisition_hold.clone();peak_airborne=acquisition_hold.clone()
    for step in range(plan.get('evaluation_steps',250)):
        with torch.no_grad():distribution,_=policy(vec.obs());action=distribution.mean.cpu().numpy()
        vec.send(action)
        peak_height=torch.maximum(peak_height,vec.sim.position[:,0,2].float())
        collisions+=int((vec.done&~vec.truncated).sum())
        distance=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
        stable=(distance<.1)&(torch.linalg.vector_norm(vec.sim.velocity[:,-1],dim=1)<.2)&~vec.done
        hold=torch.where(stable,hold+.02,0);peak_hold=torch.maximum(peak_hold,hold)
        acquiring=(distance<.25)&(torch.linalg.vector_norm(vec.sim.velocity[:,-1],dim=1)<.5)&~vec.done
        acquisition_hold=torch.where(acquiring,acquisition_hold+.02,0)
        peak_acquisition=torch.maximum(peak_acquisition,acquisition_hold)
        airborne=acquiring&(vec.sim.position[:,0,2]>.3)&(vec.sim.position[:,-1,2]>.05)
        airborne_hold=torch.where(airborne,airborne_hold+.02,0)
        peak_airborne=torch.maximum(peak_airborne,airborne_hold)
        if step%10==0:publish(vec,'deterministic evaluation',step)
        if plan and step%25==0:
            trajectory.append(dict(step=step,positions=vec.sim.position[0].cpu().tolist(),velocities=vec.sim.velocity[0].cpu().tolist(),orientation=vec.sim.orientation[0].cpu().tolist(),motors=vec.sim.motors[0].cpu().tolist(),terminal=bool(vec.done[0])))
    error=torch.linalg.vector_norm(vec.sim.position[:,-1]-vec.goal,dim=1)
    return dict(seed=seed,mean_final_error_mm=float(error.mean()*1000),errors_mm=(error*1000).tolist(),crashes=collisions,takeoffs=int((peak_height>.3).sum()),hover_successes=int((peak_hold>=1).sum()),peak_hover_hold_s=peak_hold.cpu().tolist(),peak_drone_height_m=peak_height.cpu().tolist(),
      airborne_acquisition_successes=int((peak_airborne>=.5).sum()),peak_airborne_acquisition_hold_s=peak_airborne.cpu().tolist(),airborne_acquisition_criteria=dict(version=2,distance_m=.25,speed_m_s=.5,hold_s=.5,drone_height_min_m=.3,payload_height_min_m=.05),acquisition_successes=int((peak_acquisition>=.5).sum()),peak_acquisition_hold_s=peak_acquisition.cpu().tolist(),acquisition_criteria=dict(distance_m=.25,speed_m_s=.5,hold_s=.5),trajectory=trajectory,scope='Prototype evaluation; not original obstacle navigation or sustained1mm benchmark')

trainer=None
try:
    from pufferlib.pufferl import PuffeRL,ADVANTAGE_CUDA
    if not ADVANTAGE_CUDA:raise RuntimeError('CUDA advantage kernel required')
    vec=CudaVector(128,direct=args.direct,reward_version=plan.get('reward_version',1),phase=plan.get('phase','destination'),control_version=plan.get('control_version',1),training_region=plan.get('training_region'),physics_version=plan.get('physics_version',1),training_reset_version=plan.get('training_reset_version',1),landing_penalty_version=plan.get('landing_penalty_version',0));policy=CudaPolicy().cuda()
    if args.direct and args.initial_throttle!=.5:
        with torch.no_grad():policy.actor.bias.fill_(2*args.initial_throttle-1);policy.log_std.fill_(-4.)
    if plan.get('control_version',1)==2:
        with torch.no_grad():policy.actor.bias.zero_();policy.actor.bias[0]=2*args.initial_throttle-1
    if plan.get('resume'):
        weights=torch.load(plan['resume'],weights_only=True)
        if any(k.startswith('value_body.') for k in weights):policy.enable_separate_value()
        weights.setdefault('obs_scale',policy.obs_scale)
        policy.load_state_dict(weights)
    configure_exploration(policy,plan.get('initial_log_std'))
    policy.value_gradient_version=plan.get('value_gradient_version',policy.value_gradient_version)
    if policy.value_gradient_version not in (1,2,3):raise ValueError('Unsupported value gradient version')
    if policy.value_gradient_version==3:policy.enable_separate_value()
    policy.configure_input_conditioning(plan.get('input_conditioning_version',1))
    config=dict(env='cuda_tether_prototype_v001',seed=41,torch_deterministic=True,cpu_offload=False,device='cuda',optimizer='adam',anneal_lr=False,precision='float32',amp=False,total_timesteps=32768,learning_rate=3e-4,gamma=.995,gae_lambda=.95,update_epochs=3,clip_coef=.2,vf_coef=.5,vf_clip_coef=.2,max_grad_norm=.5,ent_coef=0.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-5,min_lr_ratio=1.,data_dir=str(out/'puffer'),checkpoint_interval=100,batch_size=4096,minibatch_size=1024,max_minibatch_size=1024,bptt_horizon=32,compile=False,compile_mode='default',compile_fullgraph=False,use_rnn=False,vtrace_rho_clip=1.,vtrace_c_clip=1.,prio_alpha=0.,prio_beta0=0.)
    if args.direct:config['total_timesteps']=args.steps
    config['control_mode']='direct_motor_policy' if args.direct else 'PD_plus_residual'
    config['initial_throttle']=args.initial_throttle
    config['learning_rate']=plan.get('learning_rate',config['learning_rate'])
    config['batch_size']=plan.get('batch_size',config['batch_size'])
    config['bptt_horizon']=plan.get('bptt_horizon',config['bptt_horizon'])
    if config['batch_size'] not in (4096,16384) or config['bptt_horizon'] not in (32,128):raise ValueError('Unsupported rollout layout')
    if config['batch_size']<128*config['bptt_horizon']:raise ValueError('Rollout batch too small for 128 agents')
    config['gae_lambda']=plan.get('gae_lambda',config['gae_lambda'])
    if not .8<=config['gae_lambda']<=1:raise ValueError('gae_lambda outside bounded experiment range')
    config['max_grad_norm']=plan.get('max_grad_norm',config['max_grad_norm'])
    config['vf_clip_coef']=plan.get('vf_clip_coef',config['vf_clip_coef'])
    if not .2<=config['vf_clip_coef']<=1e6:raise ValueError('Invalid value clipping experiment bound')
    if not .05<=config['max_grad_norm']<=10:raise ValueError('max_grad_norm outside bounded experiment range')
    config['seed']=plan.get('seed',41);torch.manual_seed(config['seed'])
    config['research_plan']=plan
    (out/'config.json').write_text(json.dumps(config,indent=2))
    seeds=plan.get('validation_seeds',[51011])
    def measured():
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        try:values=[evaluate(policy,seed) for seed in seeds]
        finally:torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
        return dict(runs=values,crashes=sum(v['crashes'] for v in values),takeoffs=sum(v['takeoffs'] for v in values),hover_successes=sum(v['hover_successes'] for v in values),airborne_acquisition_successes=sum(v['airborne_acquisition_successes'] for v in values),acquisition_successes=sum(v['acquisition_successes'] for v in values),mean_final_error_mm=float(np.mean([v['mean_final_error_mm'] for v in values]))) if plan else values[0]
    def value_return_probe(stage):
        if plan.get('value_return_probe_version',0):
            if plan['value_return_probe_version']!=1 or plan.get('timeout_bootstrap_version',0) or plan.get('potential_shaping_version',0) or not args.direct:
                raise ValueError('Return probe requires direct baseline return version')
            from tether_sim.value_return_probe import probe
            (out/'value_return_probe_source.py').write_bytes(Path('tether_sim/value_return_probe.py').read_bytes())
            (out/('value_return_probe_'+stage+'.json')).write_text(json.dumps(probe(policy,plan,seeds,config['gamma']),indent=2))
    initial=measured();torch.save(policy.state_dict(),out/'initial.pt')
    if plan.get('tether_physics_probe_version',0):
        if plan['tether_physics_probe_version'] not in (1,2,3):raise ValueError('Unsupported tether physics probe')
        if plan['tether_physics_probe_version']==1:
            from tether_sim.tether_physics_probe import probe as tether_probe
            probe_source='tether_sim/tether_physics_probe.py'
        elif plan['tether_physics_probe_version']==2:
            from tether_sim.tether_physics_probe_v002 import probe as tether_probe
            probe_source='tether_sim/tether_physics_probe_v002.py'
        else:
            from tether_sim.tether_physics_probe_v003 import probe as tether_probe
            probe_source='tether_sim/tether_physics_probe_v003.py'
        (out/'tether_physics_probe_source.py').write_bytes(Path(probe_source).read_bytes())
        (out/'tether_physics_probe.json').write_text(json.dumps(tether_probe(),indent=2))
    if plan.get('long_flight_probe_version',0):
        if plan['long_flight_probe_version']!=1:raise ValueError('Unsupported long flight probe')
        from tether_sim.long_flight_probe import probe as long_probe
        (out/'long_flight_probe_source.py').write_bytes(Path('tether_sim/long_flight_probe.py').read_bytes())
        (out/'long_flight_probe.json').write_text(json.dumps(long_probe(policy,plan),indent=2))
    if plan.get('collective_response_probe_version',0):
        if plan['collective_response_probe_version']!=1:raise ValueError('Unsupported collective response probe')
        from tether_sim.collective_response_probe import probe as collective_probe
        (out/'collective_response_probe_source.py').write_bytes(Path('tether_sim/collective_response_probe.py').read_bytes())
        (out/'collective_response_probe.json').write_text(json.dumps(collective_probe(policy,plan),indent=2))
    if plan.get('compiled_dynamics_probe_version',0):
        if plan['compiled_dynamics_probe_version']!=1:raise ValueError('Unsupported dynamics probe')
        from tether_sim.cuda_dynamics_probe import probe as dynamics_probe
        (out/'cuda_dynamics_probe_source.py').write_bytes(Path('tether_sim/cuda_dynamics_probe.py').read_bytes())
        (out/'cuda_dynamics_probe.json').write_text(json.dumps(dynamics_probe(),indent=2))
    value_return_probe('initial')
    best=initial;best_step=0;torch.save(policy.state_dict(),out/'best.pt')
    def rank(value):return (value['crashes'],-value.get('hover_successes',0),-value['takeoffs'],value['mean_final_error_mm'])
    if plan.get('terminal_diagnostics'):
        def observe_terminal(obs,done,truncated,failed,reward):
            with torch.no_grad():
                _,terminal_value=policy(obs)
                ids=torch.nonzero(done).flatten()
                row=dict(ids=ids.cpu().tolist(),timeout_only=(truncated[ids]&(failed[ids]==0)).cpu().tolist(),
                    failed=failed[ids].cpu().tolist(),pre_reset_value=terminal_value.flatten()[ids].cpu().tolist(),
                    raw_reward=reward[ids].cpu().tolist(),clipped_reward=reward[ids].clamp(-1,1).cpu().tolist())
            with (out/'terminal_diagnostics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        vec.terminal_observer=observe_terminal
    bootstrap=None
    if plan.get('timeout_bootstrap_version',0):
        if plan['timeout_bootstrap_version']!=1:raise ValueError('Unsupported timeout bootstrap version')
        import hashlib, pufferlib.pufferl
        source=Path(pufferlib.pufferl.__file__).read_bytes()
        if hashlib.sha256(source).hexdigest()!=plan.get('pufferl_source_sha256'):
            raise RuntimeError('Timeout bootstrap requires audited PufferLib source hash')
        from tether_sim.timeout_bootstrap import TimeoutBootstrap
        (out/'timeout_bootstrap_source.py').write_bytes(Path('tether_sim/timeout_bootstrap.py').read_bytes())
        bootstrap=TimeoutBootstrap(vec,policy,config['gamma'])
    shaping=None
    if plan.get('potential_shaping_version',0):
        if plan['potential_shaping_version']!=1 or bootstrap is not None:
            raise ValueError('Unsupported shaping version or simultaneous return adaptation')
        import hashlib, pufferlib.pufferl
        if hashlib.sha256(Path(pufferlib.pufferl.__file__).read_bytes()).hexdigest()!=plan.get('pufferl_source_sha256'):
            raise RuntimeError('Shaping requires audited PufferLib source hash')
        from tether_sim.potential_shaping import PotentialShaping
        (out/'potential_shaping_source.py').write_bytes(Path('tether_sim/potential_shaping.py').read_bytes())
        shaping=PotentialShaping(vec,config['gamma'])
    trainer=PuffeRL(config,vec,policy)
    if shaping is not None:shaping.trainer=trainer
    if bootstrap is not None:bootstrap.trainer=trainer
    diagnostic=None
    if plan.get('gradient_diagnostics'):
        from tether_sim.gradient_diagnostics import GradientDiagnostics
        (out/'gradient_diagnostics_source.py').write_bytes(Path('tether_sim/gradient_diagnostics.py').read_bytes())
        diagnostic=GradientDiagnostics(policy,trainer.optimizer)
    started=time.monotonic()
    next_eval=plan.get('eval_every_steps',65536)
    while trainer.global_step<config['total_timesteps'] and time.monotonic()-started<plan.get('budget_seconds',120):
        trainer.evaluate()
        if bootstrap is not None:
            bootstrap.apply()
            (out/'timeout_bootstrap_status.json').write_text(json.dumps(dict(version=1,applied=bootstrap.applied)))
        if shaping is not None:
            shaping.apply()
            (out/'potential_shaping_status.json').write_text(json.dumps(dict(version=1,applied=shaping.applied)))
        if plan.get('rollout_capture_version',0):
            if plan['rollout_capture_version'] != 1:
                raise ValueError('Unsupported rollout capture version')
            import hashlib, pufferlib.pufferl
            if hashlib.sha256(Path(pufferlib.pufferl.__file__).read_bytes()).hexdigest() != plan.get('pufferl_source_sha256'):
                raise RuntimeError('Rollout capture requires audited PufferLib source')
            # Snapshot only; detached CPU copies do not mutate buffers or RNG.
            # Rewards/terminals here describe the arriving transition; actions
            # describe the action issued from the observation in that cell.
            if trainer.global_step == config['batch_size'] or trainer.global_step >= config['total_timesteps']:
                names=('observations','actions','values','logprobs','rewards','terminals','truncations')
                snapshot={name:getattr(trainer,name).detach().cpu().clone() for name in names}
                torch.save(snapshot,out/('rollout_'+str(trainer.global_step)+'.pt'))
                (out/('rollout_'+str(trainer.global_step)+'.json')).write_text(json.dumps(dict(
                    step=trainer.global_step, version=1, shapes={k:list(v.shape) for k,v in snapshot.items()},
                    scope='Pre-train learner buffers after reward adapters; incoming reward and outgoing action must not be paired at the same index without alignment review. No actual PPO advantage snapshot.',
                    source_sha256=hashlib.sha256(Path(pufferlib.pufferl.__file__).read_bytes()).hexdigest()),indent=2))
        trainer.train();publish(vec,'PufferLib training',trainer.global_step)
        if vec.landing_gate is not None:
            (out/'landing_diagnostics.json').write_text(json.dumps(dict(version=vec.landing_gate.version,training_step=trainer.global_step,**vec.landing_gate.diagnostics()),indent=2))
        if plan.get('learner_diagnostics'):
            # Public cached Puffer metrics; logging may refresh them less often
            # than each update. Retain repeated rows rather than invent freshness.
            with (out/'learner_diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=trainer.global_step,
                    cached_losses=dict(trainer.losses),
                    log_std=policy.log_std.detach().cpu().tolist()))+'\n')
        (out/'status.json').write_text(json.dumps(dict(status='training',steps=trainer.global_step,elapsed=time.monotonic()-started)))
        if plan and trainer.global_step>=next_eval:
            value=measured();(out/('eval_'+str(trainer.global_step)+'.json')).write_text(json.dumps(value,indent=2))
            torch.save(policy.state_dict(),out/('checkpoint_'+str(trainer.global_step)+'.pt'))
            if rank(value)<rank(best):best=value;best_step=trainer.global_step;torch.save(policy.state_dict(),out/'best.pt')
            next_eval+=plan.get('eval_every_steps',65536)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    if diagnostic is not None:
        (out/'gradient_diagnostics.json').write_text(json.dumps(diagnostic.summary(),indent=2));diagnostic.close()
    torch.save(policy.state_dict(),out/'policy.pt')
    final=measured()
    value_return_probe('final')
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
