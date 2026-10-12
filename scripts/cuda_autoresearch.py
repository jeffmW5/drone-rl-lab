"""Continuous serial bounded trials. No reward-based checkpoint promotion."""
import argparse,fcntl,json,os,signal,subprocess,time,threading,urllib.request,shutil,hashlib
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parent.parent
STATE=ROOT/'state/cuda_research/loop.json'
def read():return json.loads(STATE.read_text()) if STATE.exists() else dict(trials=[],phase='takeoff',paused=False)
def write(value):
    STATE.parent.mkdir(parents=True,exist_ok=True);p=STATE.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(STATE)
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','status','pause','resume']);args=parser.parse_args()
if args.action!='run':
    s=read()
    if args.action!='status':s['paused']=args.action=='pause';write(s)
    print(json.dumps(s,indent=2));raise SystemExit
os.chdir(ROOT);STATE.parent.mkdir(parents=True,exist_ok=True)
lock=(STATE.parent/'daemon.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Research daemon already running')
last_frame={'mode':'research initializing','positions':[]}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        global last_frame
        if self.path=='/state':
            try:last_frame=json.load(urllib.request.urlopen('http://127.0.0.1:8768/state',timeout=.5))
            except Exception:last_frame={**last_frame,'mode':'Between trials / paused'}
            data=json.dumps(last_frame).encode();mime='application/json'
        elif self.path in ('/','/viewer'):data=(ROOT/'tether_sim/cuda/viewer.html').read_bytes();mime='text/html'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',8767),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
stopping=False
def stop(*unused):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
env=dict(os.environ,CUDA_HOME='/home/n33du/ai/cuda-12.6-puffer',PATH='/home/n33du/ai/cuda-12.6-puffer/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
python='/home/n33du/ai/puffer-tether-venv/bin/python'
failures=0
try:
    while not stopping:
        state=read();state['daemon_pid']=os.getpid();write(state)
        if state.get('paused'):time.sleep(2);continue
        # Trainer lease also catches races after this read-only check.
        active=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                parts=p.read_bytes().decode().split('\0')
                if any(x in parts for x in ['tether_sim.train','tether_sim.puffer_train','scripts/train_tether_cuda.py']):active.append(p.parent.name)
            except OSError:pass
        if active:time.sleep(2);continue
        index=len(state['trials'])+1;phase=state['phase'];name=f'cuda_auto_{index:04d}_{phase}'
        run=ROOT/'tmp/cuda-research'/name;plan_path=ROOT/'configs/cuda_research'/f'{name}.json'
        if run.exists() or plan_path.exists():
            state.update(paused=True,error='Existing trial intent requires review: '+name);write(state);continue
        previous=state.get('incumbents',{}).get(phase)
        plan=dict(name=name,direct=True,phase=phase,reward_version=2,steps=65536,budget_seconds=90,
          eval_every_steps=32768,evaluation_steps=250,validation_seeds=[51011,51012,51013],
          learning_rate=[1e-5,3e-6,3e-5][(index-1)%3],initial_throttle=[.61,.60,.62][((index-1)//3)%3],
          seed=1000+index,playback_seconds=0,
          hypothesis='Explicit takeoff/hover reward and restrained PPO updates may prevent ground collapse',
          limitations='Reduced C CUDA particle tether; provisional validation only. No original1mm benchmark or hardware fidelity claim.')
        if previous:plan['resume']=previous['checkpoint']
        plan_path.parent.mkdir(parents=True,exist_ok=True);plan_path.write_text(json.dumps(plan,indent=2))
        trial=dict(name=name,phase=phase,plan=str(plan_path.relative_to(ROOT)),run=str(run.relative_to(ROOT)),status='launching',launched_at=time.time())
        state['trials'].append(trial);state.update(status='running '+name);write(state)
        run.parent.mkdir(parents=True,exist_ok=True)
        sources=run.parent/(name+'.sources');sources.mkdir(exist_ok=False)
        hashes={}
        for source in ['scripts/train_tether_cuda.py','tether_sim/cuda_task.py','tether_sim/cuda_backend.py','tether_sim/cuda/host.c','tether_sim/cuda/physics_v001.cu']:
            data=(ROOT/source).read_bytes();hashes[source]=hashlib.sha256(data).hexdigest();shutil.copyfile(ROOT/source,sources/Path(source).name)
        (sources/'hashes.json').write_text(json.dumps(hashes,indent=2))
        with (run.parent/(name+'.log')).open('wb') as log:
            child=subprocess.Popen([python,'-u','scripts/train_tether_cuda.py','--plan',str(plan_path),'--output',str(run),'--port','8768'],env=env,stdout=log,stderr=log)
            while child.poll() is None:time.sleep(1)
        state=read();trial=state['trials'][-1]
        if child.returncode==0 and (run/'selection.json').exists():
            failures=0;selection=json.loads((run/'selection.json').read_text());score=selection['best']
            rank=lambda v:(v['crashes'],-v['hover_successes'],-v['takeoffs'],v['mean_final_error_mm'])
            incumbent=state.setdefault('incumbents',{}).get(phase)
            promoted=incumbent is None or rank(score)<rank(incumbent['score'])
            trial.update(status='complete',selection=selection,promoted=promoted)
            if promoted:state['incumbents'][phase]=dict(checkpoint=str(run/'best.pt'),score=score)
            if score['crashes']==0 and score['hover_successes']>=18 and score['mean_final_error_mm']<100:
                state['phase']={'takeoff':'hover','hover':'destination','destination':'destination'}[phase]
            state.pop('error',None)
        else:
            failures+=1;trial.update(status='failed',exit_code=child.returncode);state['error']='Trainer failed; inspect '+str(run)+'.log'
            if failures>=3:state['paused']=True
        state['status']='ready for next bounded trial' if not state.get('paused') else 'paused for failure review'
        write(state)
        (ROOT/'outbox/tether_cuda_loop.md').write_text('# Continuous CUDA research\n\n'+json.dumps({k:state.get(k) for k in ['status','phase','paused','incumbents','error']},indent=2)+'\n\nFull rejected trials: state/cuda_research/loop.json. No hardware accuracy claim.\n')
finally:
    server.shutdown();lock.close()
