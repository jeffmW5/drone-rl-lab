import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tether_sim.cuda_task import CudaPolicy
from tether_sim.optimizer_resume import restore_optimizer
p=Path('tmp/cuda-research/cuda_auto_0685_takeoff');model=CudaPolicy();model.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=True))
optimizer=torch.optim.Adam(model.parameters(),lr=3e-6,betas=(.9,.999),eps=1e-5)
state=next((p/'puffer').glob('*/trainer_state.pt'))
result=restore_optimizer(optimizer,model,state)
assert result['optimizer_steps']==[192.0] and len(optimizer.state)==9
with torch.no_grad():model.actor.bias[0]+=.01
try:restore_optimizer(optimizer,model,state)
except ValueError as e:assert 'match loaded policy' in str(e)
else:raise AssertionError('Mismatched weights accepted')
print('Actual matched685 Adam state restored192steps/9slots; mismatched policy rejected')
