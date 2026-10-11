"""Windows localhost launcher/monitor. No drone hardware commands."""
import argparse
import http.client
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(r'D:\drone-rl-lab')
NATIVE='/home/n33du/ai/drone-rl-lab'
TRAIN='/home/n33du/ai/crazyflie-tether-venv/bin/python'
SIM='/home/n33du/ai/tether-sim-venv/bin/python'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
TOKEN=secrets.token_urlsafe(32)
VIEWER_PORT=0
cache={'runs':[],'processes':[],'error':'Connecting to WSL…'}
lock=threading.Lock()
action_lock=threading.Lock()
backend=None
launchers=[]


def wsl_args(*args):
    return ['wsl.exe','-d','Ubuntu-Laya','--cd',NATIVE,'--',*args]


def collect():
    global cache
    while True:
        try:
            result=subprocess.run(wsl_args('/usr/bin/python3','/mnt/d/drone-rl-lab/tether_sim/monitor_snapshot.py'),capture_output=True,text=True,timeout=12,creationflags=FLAGS)
            if result.returncode:
                raise RuntimeError(result.stderr[-500:])
            data=json.loads(result.stdout)
            with lock: cache=data
        except Exception as e:
            with lock: cache={**cache,'error':str(e)}
        time.sleep(8)


def launch(args,label):
    logs=ROOT/'tmp'/'command-center'
    logs.mkdir(parents=True,exist_ok=True)
    with (logs/(label+'.log')).open('ab') as log:
        process=subprocess.Popen(wsl_args(*args),stdout=log,stderr=log,creationflags=FLAGS)
    launchers.append((process,label))
    return process.pid


def act(action,run_name=None):
    global backend
    if action not in ('preview','policy','train'):
        raise ValueError('Unknown launch action')
    with action_lock:
        # Refresh processes before mutations to avoid launching duplicate jobs.
        result=subprocess.run(wsl_args('/usr/bin/python3','/mnt/d/drone-rl-lab/tether_sim/monitor_snapshot.py'),capture_output=True,text=True,timeout=12,creationflags=FLAGS)
        if result.returncode: raise RuntimeError(result.stderr[-300:])
        data=json.loads(result.stdout)
        if action=='train':
            if any('tether_sim.train' in p['args'] for p in data['processes']) or any(p.poll() is None and label.startswith('tether_v007_cuda_') for p,label in launchers):
                return 'Training is already running; no duplicate started.'
            name='tether_v007_cuda_'+time.strftime('%Y%m%d_%H%M%S')
            launch([TRAIN,'-u','-m','tether_sim.train','--config','configs/tether_v007_cuda.json','--output','tmp/tether-runs/'+name],name)
            return 'Started V007 CUDA continuation: '+name+'. Existing results are preserved.'
        if action=='policy':
            run_name=run_name or 'tether_v005'
            run=next((r for r in data['runs'] if r['name']==run_name),{})
            if not run.get('playback_ready'):
                raise ValueError('Selected run best/final checkpoint pairs are not ready yet.')
        # The command-center owns port VIEWER_PORT only. Never kill unrelated viewers.
        existing=[p for p in data['processes'] if 'tether_sim.viewer' in p['args'] and str(VIEWER_PORT) in p['args']]
        for p in existing:
            subprocess.run(wsl_args('kill',str(p['pid'])),timeout=5,creationflags=FLAGS,check=True)
        ip=subprocess.run(wsl_args('hostname','-I'),capture_output=True,text=True,timeout=5,creationflags=FLAGS,check=True).stdout.split()[0]
        socket.inet_aton(ip)
        backend=(ip,VIEWER_PORT)
        monitor_run=run_name if action=='policy' else next((r['name'] for r in data['runs'] if r.get('active')),data['runs'][0]['name'] if data['runs'] else 'tether_v005')
        args=[TRAIN if action=='policy' else SIM,'-u','-m','tether_sim.viewer','--host',ip,'--port',str(VIEWER_PORT),'--seed','21001','--training-run','tmp/tether-runs/'+monitor_run]
        if action=='policy':
            run_name=run_name or 'tether_v005'
            run=next((r for r in data['runs'] if r['name']==run_name),{})
            if not {'best.zip','final.zip'}.issubset(set(run.get('checkpoints',[]))):
                raise ValueError('Selected run best/final checkpoint pairs are not ready yet.')
            args += ['--policy-run','tmp/tether-runs/'+run_name,'--checkpoint','best']
        else: args+=['--task-config','configs/tether_v007_cuda.json']
        launch(args,'viewer')
        return 'Starting '+('selected trained policy' if action=='policy' else 'V007 geometry-aware PD obstacle preview')+'. Open Flight Studio in a few seconds.'


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def send(self,body,kind='application/json',status=200):
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.path=='/api/status':
            with lock: payload={**cache,'token':TOKEN,'viewer':bool(backend)}
            self.send(json.dumps(payload).encode());return
        if self.path.startswith('/sim'):
            if not backend:
                self.send(b'Launch a simulator from Command Center first.','text/plain',503);return
            connection=http.client.HTTPConnection(*backend,timeout=3)
            try:
                path=self.path[4:] or '/'
                connection.request('GET',path)
                response=connection.getresponse();body=response.read()
                kind=response.getheader('Content-Type','application/octet-stream')
                if path=='/':
                    body=body.replace(b"'/state'",b"'/sim/state'").replace(b"'/command'",b"'/sim/command'").replace(b"'/frame.jpg'",b"'/sim/frame.jpg'").replace(b"'/camera.jpg'",b"'/sim/camera.jpg'").replace(b'href="/guide"',b'href="/guide"')
                self.send(body,kind,response.status)
            except (OSError,http.client.HTTPException):
                self.send(b'Simulator starting or unavailable. Return to Command Center and launch again.','text/plain',503)
            finally: connection.close()
            return
        file=Path(__file__).with_name('command_center.html' if self.path=='/' else 'guide.html')
        if self.path not in ('/','/guide'):
            self.send(b'Not found','text/plain',404);return
        self.send(file.read_bytes(),'text/html; charset=utf-8')

    def do_POST(self):
        if self.path=='/api/shutdown':
            length=int(self.headers.get('Content-Length','0'))
            data=json.loads(self.rfile.read(min(length,2048)))
            if not secrets.compare_digest(data.get('token',''),TOKEN):
                self.send(b'{}',status=403);return
            self.send(b'{}')
            threading.Thread(target=self.server.shutdown,daemon=True).start()
            return
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8765','http://localhost:8765'):
            self.send(b'{}',status=403);return
        length=int(self.headers.get('Content-Length','0'))
        if length>2048: self.send(b'{}',status=413);return
        body=self.rfile.read(length)
        if self.path=='/sim/command' and backend:
            connection=http.client.HTTPConnection(*backend,timeout=3)
            try:
                connection.request('POST','/command',body,{'Content-Type':'application/json'})
                response=connection.getresponse();self.send(response.read(),status=response.status)
            except OSError: self.send(b'{}',status=503)
            finally: connection.close()
            return
        try:
            data=json.loads(body)
            if self.path!='/api/launch' or not secrets.compare_digest(data.get('token',''),TOKEN):
                self.send(b'{}',status=403);return
            self.send(json.dumps({'message':act(data.get('action'),data.get('run'))}).encode())
        except Exception as error:
            self.send(json.dumps({'error':str(error)}).encode(),status=400)


def main():
    global VIEWER_PORT
    p=argparse.ArgumentParser()
    p.add_argument('--ready',required=True)
    args=p.parse_args()
    # Reserve an independent WSL simulator port; never terminate another viewer.
    result=subprocess.run(wsl_args('/usr/bin/python3','-c','import socket; s=socket.socket(); s.bind(("0.0.0.0",0)); print(s.getsockname()[1]); s.close()'),capture_output=True,text=True,timeout=20,creationflags=FLAGS,check=True)
    VIEWER_PORT=int(result.stdout.strip())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    Path(args.ready).write_text(json.dumps({'port':server.server_port,'token':TOKEN,'pid':os.getpid()}))
    threading.Thread(target=collect,daemon=True).start()
    try: server.serve_forever()
    finally:
        try:
            result=subprocess.run(wsl_args('/usr/bin/python3','/mnt/d/drone-rl-lab/tether_sim/monitor_snapshot.py'),capture_output=True,text=True,timeout=12,creationflags=FLAGS)
            for process in json.loads(result.stdout).get('processes',[]):
                if 'tether_sim.viewer' in process['args'] and str(VIEWER_PORT) in process['args']:
                    subprocess.run(wsl_args('kill',str(process['pid'])),timeout=5,creationflags=FLAGS)
        except Exception: pass
        for process,label in launchers:
            if label=='viewer' and process.poll() is None: process.terminate()

if __name__=='__main__': main()

