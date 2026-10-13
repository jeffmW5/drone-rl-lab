from pathlib import Path
import datetime,hashlib,json,subprocess,urllib.request
root=Path.cwd();out=root/'state/cuda_research/methodology/review_0108';now=datetime.datetime.now(datetime.timezone.utc).isoformat();records=[];manifest=[]
for index in (853,854,855):
 d=root/'tmp/cuda-research'/f'cuda_auto_{index:04d}_takeoff'
 r=json.loads((d/'result.json').read_text());selection=json.loads((d/'selection.json').read_text());config=json.loads((d/'config.json').read_text());assert r['status']=='complete'
 periodic=[]
 for p in sorted(d.glob('eval_*.json'),key=lambda p:int(p.stem.split('_')[1])):
  v=json.loads(p.read_text());periodic.append(dict(step=int(p.stem.split('_')[1]),path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**{k:v.get(k) for k in ('crashes','takeoffs','zone_hover_successes','zone_escapes','zone_hover_error_mm')},peak_zone_hold_s=max(h for run in v['runs'] for h in run['zone_hover_hold_s'])))
 for p in (d/'result.json',d/'selection.json',d/'config.json',d/'status.json',d/'thrust_order_checks.json',d/'thrust_order_checks.sources.json'):
  manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 records.append(dict(trial=index,run=str(d),config=config,initial=r['initial'],final=r['final'],best=selection['best'],best_step=selection['best_step'],promoted=selection['promoted'],steps=r['steps'],training_seconds=r['training_seconds'],budget_censored=r['steps']<config['total_timesteps'],periodic=periodic,physics_invariants=json.loads((d/'thrust_order_checks.json').read_text())))
for filename,obj in [('results.json',records),('evaluation_manifest.json',dict(collected_at=now,last_reviewed_trial=855,files=manifest))]:
 with(out/filename).open('x',encoding='utf-8')as f:json.dump(obj,f,indent=2,allow_nan=False)
source=dict(url='https://arxiv.org/html/1712.00378v1',version='v1',sections_read=['3 Eq5-6','4'],retrieved_at=now,author_project='https://sites.google.com/view/time-limits-in-rl',author_code_executed=False)
try:
 with urllib.request.urlopen(urllib.request.Request(source['url'],headers={'User-Agent':'drone-rl-lab-methodology/1.0'}),timeout=20)as response:body=response.read(8000000)
 p=out/'source_arxiv_1712_00378_v1.html';p.write_bytes(body);source.update(file=p.name,sha256=hashlib.sha256(body).hexdigest(),status=200)
except Exception as error:source['error']=str(error)
sources=[dict(url='https://arxiv.org/pdf/1707.06347',version='v2 (28 Aug2017, PDFp0)',sections_read=['2.1','3 Eq7','5 Eq9-12 and Algorithm1'],read_at=now,hf_metadata='source_01.txt',hf_markdown='source_02.txt',author_code_metadata=None,author_code_executed=False),source]
with(out/'original_sources.json').open('x',encoding='utf-8')as f:json.dump(sources,f,indent=2)
with(out/'gpu_process_snapshot.txt').open('x',encoding='utf-8')as f:f.write(now+'\n'+subprocess.run(['pgrep','-af','^/usr/bin/python3 scripts/cuda_autoresearch.py run$|^/home/n33du/ai/puffer-tether-venv/bin/python -u scripts/train_tether_cuda.py '],capture_output=True,text=True).stdout)
print(json.dumps([{'trial':r['trial'],'steps':r['steps'],'censored':r['budget_censored'],'best_step':r['best_step'],'final_escapes':r['final']['zone_escapes'],'final_hover':r['final']['zone_hover_successes'],'final_altitude_mm':r['final']['zone_hover_error_mm'],'physics_check':r['physics_invariants']['status']}for r in records],indent=2))
