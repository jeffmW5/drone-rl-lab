import ast,json,hashlib
from pathlib import Path
root=Path.cwd();paths=['scripts/train_tether_cuda.py','tether_sim/cuda_task.py','tether_sim/recovery_reset.py','tether_sim/recovery_probe_v002.py']
for p in paths:ast.parse((root/p).read_text())
base=json.loads((root/'configs/cuda_research/methodology_queue_v000_user_abb_deep_audit.json').read_text())[-1]
queue=[]
for seed in (70501,70502,70503):
 for reset in (5,6):
  p=dict(base);p.update(methodology_trial=f'attitude_recovery_reset{reset}_seed{seed}',methodology_condition=f'attitude_recovery_reset{reset}',methodology_review='state/cuda_research/methodology/review_0109/review.md',seed=seed,training_reset_version=reset,evaluation_reset_version=3,steps=262144,budget_seconds=600,eval_every_steps=16384,recovery_probe_version=2,recovery_models=[{'current_policy':True}],recovery_rollout='/home/n33du/ai/drone-rl-lab/tmp/cuda-research/cuda_auto_0819_takeoff/rollout_65536.pt',evaluation_trace_version=1,learner_valid_mask_version=2,hypothesis='Isolate signed2degree warm-lane orientation exposure (reset6) versus identicalneutralwarmstarts(reset5); test learned recovery/groundtransfer, not reward coefficients',rejection_rule='No promotion without paired replicated recovery improvement, noextra ground failures/lost takeoffs, and genuineground sustainedhover; initialized/reused2degree recovery isnotgroundhover orheldoutproof')
  assert p['physics_version']==5 and p['reward_version']==15 and p['collective_observation_version']==2 and p['control_version']==2 and p['resume'].endswith('cuda_auto_0759_takeoff/best.pt')
  queue.append(p)
filename='methodology_queue_v000_user_abd_attitude_recovery.json';body=json.dumps(queue,indent=2,allow_nan=False)+'\n'
for repo in (root,Path('/mnt/d/drone-rl-lab')):
 target=repo/'configs/cuda_research'/filename
 assert not target.exists(),str(target)
 pending=target.with_suffix('.pending')
 with pending.open('x',encoding='utf-8')as f:f.write(body)
 pending.rename(target)
manifest=dict(queue=filename,queue_sha256=hashlib.sha256(body.encode()).hexdigest(),source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()for p in paths},single_changed_factor='warm-lane initial orientation',tilt_degrees=2,training_seeds=[70501,70502,70503],ground_evaluation_reset=3,mask=2,ground_metrics_unchanged=True,artificial_recovery_cases='2degree cases reuse training states;10degree/rate/velocity cases are fixed transfer diagnostics, not hardware/audit holdout',cpu_tests=['recovery_reset_cpu_test_v002.json','recovery_v002_cpu_checks_root_v003.json'],gpu_results='pending')
with(root/'state/cuda_research/methodology/review_0109/recovery_curriculum/plan.json').open('x',encoding='utf-8')as f:json.dump(manifest,f,indent=2)
print(json.dumps({'frozen_queue':filename,'trials':[p['methodology_trial']for p in queue],'all_syntax_valid':True},indent=2))
