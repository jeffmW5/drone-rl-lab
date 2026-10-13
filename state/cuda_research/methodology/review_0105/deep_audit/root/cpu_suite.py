from pathlib import Path
import subprocess,json,hashlib,time,os
root=Path.cwd();out=root/'tmp/deep_audit_root';rows=[]
checks=['check_cuda_logic','check_ground_geometry','check_hover_motor_starts','check_zone_hover','check_collective_response','check_collective_observation','check_landing_penalty','check_optimizer_resume','check_timeout_bootstrap_buffers','check_policy_update_capture','check_advantage_capture','check_advantage_capture_v002']
for name in checks:
 f=root/'scripts'/f'{name}.py';start=time.time();p=subprocess.run(['/home/n33du/ai/puffer-tether-venv/bin/python',str(f)],capture_output=True,text=True,timeout=30,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)))
 rows.append(dict(script=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),exit=p.returncode,elapsed_s=time.time()-start,stdout=p.stdout,stderr=p.stderr));print(name,p.returncode,flush=True)
(out/'cpu_suite_results_with_path.json').write_text(json.dumps(rows,indent=2))
