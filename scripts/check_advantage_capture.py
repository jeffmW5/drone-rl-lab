import sys,hashlib,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,pufferlib.pufferl as module
from tether_sim.advantage_capture import capture_train
original=module.compute_puff_advantage
x=torch.ones(2,3);result=x*2
fake=lambda *args:result
module.compute_puff_advantage=fake
class Trainer:
 global_step=65536
 def train(self):
  out=module.compute_puff_advantage(x,x,x,x,torch.zeros_like(x),.995,.95,1.,1.)
  assert out is result
try:
 with tempfile.TemporaryDirectory(dir='tmp') as name:
  rng=torch.get_rng_state().clone();before=x.clone()
  capture_train(Trainer(),Path(name),hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
  saved=torch.load(next(Path(name).glob('*.pt')),weights_only=True)
  assert torch.equal(saved['advantages'],result) and torch.equal(x,before)
  assert torch.equal(torch.get_rng_state(),rng) and module.compute_puff_advantage is fake
finally:module.compute_puff_advantage=original
print('Capture preserves return identity, input tensors, CPU RNG and restores callable; captures detached actual result')
