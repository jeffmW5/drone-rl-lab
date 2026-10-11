"""Read local tether artifacts without importing the simulation or Torch."""
import csv
import json
import math
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def snapshot(root=Path('tmp/tether-runs')):
    processes=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().decode().split('\0')
            if 'tether_sim.train' in args or 'tether_sim.viewer' in args:
                processes.append({'pid':int(p.parent.name),'args':args})
        except (OSError, UnicodeError):
            pass
    runs=[]
    for p in sorted(root.glob('*/config.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
        folder=p.parent
        status=read_json(folder/'status.json')
        active=any('tether_sim.train' in q['args'] and str(folder) in q['args'] for q in processes)
        effective=status.get('status','not started')
        if active and effective!='complete':
            effective='training / evaluating'
        elif effective=='training' and not active:
            effective='interrupted / stale'
        curves=[]
        try:
            with (folder/'progress.csv').open() as f:
                for row in csv.DictReader(f):
                    if row.get('time/total_timesteps'):
                        curves.append({k:float(row[k]) for k in ['time/total_timesteps','train/value_loss','train/approx_kl'] if row.get(k) and math.isfinite(float(row[k]))})
        except (OSError, ValueError):
            pass
        evaluation=read_json(folder/'evaluation_final.json')
        if not evaluation:
            choices=sorted(folder.glob('eval_*.json'),key=lambda p:p.stat().st_mtime)
            evaluation=read_json(choices[-1]) if choices else {}
        runs.append(dict(name=folder.name,status=status,effective=effective,active=active,config=read_json(p),
                         checkpoints=sorted(f.name for f in folder.glob('*.zip')),
                         playback_ready=all((folder/f).exists() for f in ('best.zip','final.zip','best_normalize.pkl','final_normalize.pkl')),
                         evaluation=evaluation,curves=curves[-300:]))
    return dict(collected_at=time.time(),processes=processes,runs=runs)


if __name__=='__main__':
    print(json.dumps(snapshot()))
