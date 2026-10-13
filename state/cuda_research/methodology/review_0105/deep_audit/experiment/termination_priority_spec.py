import json, hashlib
from pathlib import Path
import torch
OUT=Path('tmp/deep_audit_experiment')

def termination_flags(steps,failed,version=1,training_region_escape=None):
    if version not in (1,2):raise ValueError('Unsupported termination version')
    failure=failed!=0
    truncated=steps>=500
    if training_region_escape is not None:truncated=truncated|training_region_escape
    if version==2:truncated=truncated&~failure
    return failure|truncated,truncated

rows=[]
for step in (0,249,250,499,500,501):
    for code in (0,1,3):
        s=torch.tensor([step]);f=torch.tensor([code])
        d1,t1=termination_flags(s,f,1);d2,t2=termination_flags(s,f,2)
        assert bool(d1)==bool(d2)
        if step<500:assert torch.equal(t1,t2)
        if code==0:assert torch.equal(t1,t2)
        if step>=500 and code!=0:assert bool(t1) and not bool(t2) and bool(d2)
        rows.append(dict(step=step,failed=code,v1=dict(done=bool(d1),truncated=bool(t1),counted_failure=bool(d1&~t1)),v2=dict(done=bool(d2),truncated=bool(t2),counted_failure=bool(d2&~t2))))
# Mixedbatch; legacy artificialregion cutoff remains truncation only absenttruefailure.
s=torch.tensor([500,500,250,250]);f=torch.tensor([0,3,0,1]);region=torch.tensor([False,True,True,True])
d,t=termination_flags(s,f,2,region)
assert d.tolist()==[True,True,True,True] and t.tolist()==[True,False,True,False]
try:termination_flags(s,f,3)
except ValueError:pass
else:raise AssertionError('Version guard failed')
(OUT/'termination_priority_tests.json').write_text(json.dumps(dict(scope='CPU proposed helper specification only, production unchanged. This tests flag logic, not physics.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,all_assertions_passed=True,current_standard250step_evaluation_affected=False),indent=2))
print('18 cases and mixedbatch checks passed; source unchanged; current250step evaluator unaffected.')
