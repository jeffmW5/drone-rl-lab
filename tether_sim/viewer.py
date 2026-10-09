"""Run with python -m tether_sim.viewer; local, server-rendered 3D dashboard."""
import argparse
import io
import json
import threading
import time
import math
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mujoco
from PIL import Image
from .env import TetherEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--training-run',default='tmp/tether-runs/tether_v002')
    args = parser.parse_args()
    env = TetherEnv()
    env.model.vis.quality.offsamples=2
    env.model.vis.quality.shadowsize=2048
    prop_bases=[env.model.geom(f'prop{i}').quat.copy() for i in range(4)]
    lock = threading.Lock()
    shared = dict(frame=b'', camera=b'', state={}, commands=[], paused=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ('/stream','/camera-stream'):
                self.send_response(200)
                self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                key='frame' if self.path=='/stream' else 'camera'
                previous=None
                try:
                    while True:
                        with lock:
                            body=shared[key]
                        if body and body is not previous:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body+b'\r\n')
                            self.wfile.flush()
                            previous=body
                        time.sleep(.005)
                except (BrokenPipeError,ConnectionResetError):
                    return
            with lock:
                if self.path == '/frame.jpg':
                    body, mime = shared['frame'], 'image/jpeg'
                elif self.path == '/camera.jpg':
                    body, mime = shared['camera'], 'image/jpeg'
                elif self.path == '/state':
                    body, mime = json.dumps(shared['state']).encode(), 'application/json'
                else:
                    body, mime = Path(__file__).with_name('studio.html').read_bytes(), 'text/html'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            size = int(self.headers.get('Content-Length',0))
            if size > 4096:
                self.send_error(413)
                return
            try:
                command = json.loads(self.rfile.read(size))
                with lock:
                    shared['commands'].append(command)
                self.send_response(204)
                self.end_headers()
            except (ValueError, TypeError):
                self.send_error(400)

    server = ThreadingHTTPServer(('127.0.0.1',args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0,0,.7]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -15
    print(f'Tether Lab http://localhost:{args.port}', flush=True)
    scenario, paused, gust_until = 'lift_carry', False, 0
    env.reset(options={'scenario':scenario})
    mode='overview'
    accumulator=0.
    last=time.monotonic()
    frame_times=[]
    onboard_jpeg=b''
    frame_number=0
    try:
        while True:
            start = time.monotonic()
            elapsed=min(.2,start-last)
            last=start
            with lock:
                commands, shared['commands'] = shared['commands'], []
            for command in commands:
                if command.get('type') == 'reset':
                    scenario = command.get('scenario','hover')
                    if scenario in ('hover','swing','lift','carry','lift_carry'):
                        env.reset(options={'scenario':scenario})
                        paused=False
                        accumulator=0
                elif command.get('type') == 'pause':
                    paused = not paused
                elif command.get('type') == 'gust':
                    gust_until = env.data.time + 1.
                elif command.get('type') == 'view':
                    mode=command.get('mode','overview')
                    camera.distance={'overview':1.65,'follow':.8,'inspect':.22}.get(mode,1.65)
                    if mode=='inspect':
                        paused=True
                elif command.get('type') == 'orbit':
                    camera.azimuth += max(-30,min(30,float(command.get('dx',0))))
                    camera.elevation = max(-85,min(-5,camera.elevation+float(command.get('dy',0))))
                    camera.distance = max(.4,min(8,camera.distance+float(command.get('zoom',0))))
            if not paused:
                accumulator+=elapsed
                env.wind[0] = .025 if env.data.time < gust_until else 0
                for _ in range(min(10,int(accumulator/env.config.control_dt))):
                    _,_,terminated,truncated,_ = env.step(env.baseline())
                    accumulator-=env.config.control_dt
                    if truncated:
                        env.reset(options={'scenario':scenario})
                        accumulator=0
                        break
                    if terminated:
                        paused = True
                        break
            else:
                accumulator=0
            camera.lookat[:]=env.data.xpos[env.drone]+[0,0,(-.27 if mode=='overview' else -.04 if mode=='follow' else .005)]
            for i in range(4):
                angle=env.data.time*(250 if i%2 else -250)
                mujoco.mju_mulQuat(env.model.geom(f'prop{i}').quat,np.array([math.cos(angle/2),0,0,math.sin(angle/2)]),prop_bases[i])
            frame = env.render('onboard' if mode=='onboard' else camera)
            fp = io.BytesIO()
            Image.fromarray(frame).save(fp,format='JPEG',quality=85)
            if frame_number%3==0:
                cp = io.BytesIO()
                Image.fromarray(env.camera_frame()).save(cp,format='JPEG')
                onboard_jpeg=cp.getvalue()
            frame_number+=1
            frame_times.append(start)
            frame_times=frame_times[-60:]
            fps=(len(frame_times)-1)/(frame_times[-1]-frame_times[0]) if len(frame_times)>1 else 0
            state = env.metrics() | dict(paused=paused,scenario=scenario, wind_N=env.wind.tolist(),fps=fps,view=mode,lag_s=accumulator)
            try:
                state['training']=json.loads((Path(args.training_run)/'status.json').read_text())
            except (OSError,ValueError):
                state['training']={'status':'not started'}
            with lock:
                shared.update(frame=fp.getvalue(), camera=onboard_jpeg,state=state)
            time.sleep(max(0,1/30-(time.monotonic()-start)))
    finally:
        server.shutdown()
        env.close()


if __name__ == '__main__':
    main()
