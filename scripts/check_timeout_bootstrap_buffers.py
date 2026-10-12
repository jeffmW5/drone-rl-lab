"""CPU buffer integration checks for TimeoutBootstrap, without GPU launches."""
import json
from types import SimpleNamespace
import torch
from tether_sim.timeout_bootstrap import TimeoutBootstrap

class Vec:
    num_agents = 3
    terminal_observer = None
    def recv(self):
        return None, None, None, None, [], list(range(3)), None

def run():
    vec = Vec()
    trainer = SimpleNamespace(ep_indices=torch.tensor([3, 4, 5]), ep_lengths=torch.tensor([31,31,31]), rewards=torch.zeros(6,32), terminals=torch.zeros(6,32))
    hook = TimeoutBootstrap(vec, lambda obs: (None, torch.tensor([[2.], [100.], [-2.]])), .995)
    hook.trainer = trainer
    hook.observe(None, torch.tensor([True,True,True]), torch.tensor([True,True,True]), torch.tensor([0,1,0]), torch.tensor([5.,-5.,-5.]))
    vec.recv()
    trainer.rewards[3,31]=1; trainer.rewards[4,31]=-1; trainer.rewards[5,31]=-1
    trainer.terminals[3:6,31]=1
    # Mimic evaluate resetting indexing before apply; staged coordinates must survive.
    trainer.ep_indices=torch.tensor([0,1,2]);trainer.ep_lengths.zero_()
    hook.apply()
    assert torch.allclose(trainer.rewards[3:6,31],torch.tensor([2.99,-1.,-2.99]))
    assert torch.equal(trainer.terminals[3:6,31],torch.ones(3))
    assert hook.applied==2 and not hook.staged and hook.pending is None
    # Terminal from last send remains pending until next evaluate's first recv.
    hook.observe(None,torch.tensor([True,False,False]),torch.tensor([True,False,False]),torch.zeros(3,dtype=torch.int),torch.tensor([.25,0.,0.]))
    hook.apply(); assert hook.pending is not None
    vec.recv();trainer.rewards[0,0]=.25;trainer.terminals[0,0]=1;hook.apply()
    assert abs(float(trainer.rewards[0,0])-2.24)<1e-6
    # Fail closed when learner clipping/index semantics change.
    hook.observe(None,torch.tensor([True,False,False]),torch.tensor([True,False,False]),torch.zeros(3,dtype=torch.int),torch.tensor([.25,0.,0.]))
    vec.recv();trainer.rewards[0,0]=.5
    try:hook.apply()
    except RuntimeError:pass
    else:raise AssertionError('Misaligned buffer accepted')
    return dict(status='pass',scope='CPU real hook and simulated Puffer buffer lifecycle; GPU learner end-to-end still pending',applied_before_failure=hook.applied)
if __name__=='__main__':print(json.dumps(run(),indent=2))
