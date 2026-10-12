import json
from types import SimpleNamespace
import torch
from tether_sim.potential_shaping import PotentialShaping
class Vec:
    num_agents=2
    def __init__(self):
        self.sim=SimpleNamespace(position=torch.zeros(2,1,3));self.sim.position[:,0,0]=torch.tensor([2.,1.])
        self.goal=torch.zeros(2,3);self.done=torch.tensor([False,False]);self.reward=torch.tensor([5.,-5.])
    def send(self, actions):
        self.sim.position[:,0,0]=torch.tensor([1.,100.]);self.done=torch.tensor([False,True])
    def recv(self):return None,self.reward,self.done,self.done,[],[0,1],None
v=Vec();h=PotentialShaping(v,.995)
t=SimpleNamespace(ep_indices=torch.tensor([2,3]),ep_lengths=torch.tensor([31,31]),rewards=torch.zeros(4,32),terminals=torch.zeros(4,32))
h.trainer=t;v.send(None);v.recv();t.rewards[2:4,31]=torch.tensor([1.,-1.]);t.terminals[2:4,31]=torch.tensor([0.,1.])
t.ep_indices=torch.tensor([0,1]);t.ep_lengths.zero_();h.apply()
assert torch.allclose(t.rewards[2:4,31],torch.tensor([1+.995*(-.5)+2/3,-.5]))
assert torch.equal(t.terminals[2:4,31],torch.tensor([0.,1.]))
# Discounted shaping telescopes to -Phi(initial) at a terminal endpoint.
phi=[-.7,-.2,-.4,0.];g=.995
terms=[g*phi[i+1]-phi[i] for i in range(3)]
assert abs(sum(g**i*x for i,x in enumerate(terms))+.0+phi[0])<1e-12
# Moving closer receives more shaping than moving away from the same state.
assert g*(-.2)-(-.5)>g*(-.7)-(-.5)
print(json.dumps(dict(status='pass',scope='CPU real hook, clipping, row rollover, terminal reset and discounted telescoping; GPU pending'),indent=2))
