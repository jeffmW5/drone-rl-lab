"""Smoke-check a running local studio and record measured streaming performance."""
import io
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image


def main():
    base='http://localhost:8766'
    command=urllib.request.Request(base+'/command',data=json.dumps({'type':'reset','scenario':'lift_carry'}).encode(),headers={'Content-Type':'application/json'})
    urllib.request.urlopen(command,timeout=5).close()
    deadline=time.monotonic()+3
    while True:
        with urllib.request.urlopen(base+'/state',timeout=5) as r:
            state=json.load(r)
        if state['scenario']=='lift_carry' and state['time']<.5 and not state['paused']:
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Viewer did not acknowledge reset')
        time.sleep(.05)
    samples=[]
    for _ in range(30):
        with urllib.request.urlopen(base+'/state',timeout=5) as r:
            samples.append(json.load(r))
        time.sleep(.1)
    assert samples[-1]['scenario']=='lift_carry' and not samples[-1]['paused']
    assert samples[-1]['time']>samples[0]['time']+2
    with urllib.request.urlopen(base+'/frame.jpg',timeout=5) as r:
        image=Image.open(io.BytesIO(r.read()))
        assert image.size==(960,540)
    with urllib.request.urlopen(base+'/camera.jpg',timeout=5) as r:
        camera=Image.open(io.BytesIO(r.read()))
        assert camera.size==(324,244) and camera.mode=='L'
    with urllib.request.urlopen(base+'/stream',timeout=5) as r:
        assert 'multipart/x-mixed-replace' in r.headers['Content-Type']
        chunk=r.read(262144)
        boundaries=chunk.count(b'--frame\r\n')
        assert boundaries>=2
    report={'sample_seconds':3,'fps_min':min(s['fps'] for s in samples),'fps_mean':sum(s['fps'] for s in samples)/len(samples),'fps_max':max(s['fps'] for s in samples),'max_backlog_s':max(s['lag_s'] for s in samples),'stream_boundaries_in_256KiB':boundaries,'main_resolution':image.size,'onboard_resolution':camera.size,'notes':'Local WSLg run; frame rate varies with host load. Camera updates every third rendered frame.'}
    Path('tether_sim/evidence/viewer.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
