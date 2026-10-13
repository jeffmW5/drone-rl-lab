import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,json,hashlib
from pathlib import Path
from pufferlib import _C
root=Path('/home/n33du/ai/drone-rl-lab');out=root/'tmp/deep_audit_learner';torch.set_num_threads(1)
def kernel(values,rewards,terminals,ratio=None):
 if ratio is None:ratio=torch.ones_like(values)
 adv=torch.zeros_like(values);torch.ops.pufferlib.compute_puff_advantage(values,rewards,terminals,ratio,adv,.995,.95,1.,1.);return adv
# Arriving reward: action0 receives reward1, action1 receives reward2, finalaction needs nextobservation outside32cells.
v=torch.zeros(2,4);r=torch.tensor([[0.,1.,0.,-1.],[123.,1.,0.,-1.]]);d=torch.zeros_like(v);d[:,-1]=1
actual=kernel(v,r,d);rho=1.;lam=.995*.95
expected=torch.tensor([[1.-lam*lam,-lam,-1.,0.]]).expand_as(v)
assert torch.allclose(actual,expected,atol=2e-7)
assert torch.equal(actual[0],actual[1])
# Endcell actorgradient is nonzero despite missingreward because mean-centering includes it.
raw=torch.tensor([[-3.,-2.,-1.,0.]])
oldmean=torch.zeros(1,4);actions=torch.tensor([[1.,-1.,1.,1.]])
means=torch.zeros_like(actions,requires_grad=True)
oldlogp=torch.distributions.Normal(oldmean,torch.ones_like(means)).log_prob(actions).detach()
logp=torch.distributions.Normal(means,torch.ones_like(means)).log_prob(actions)
norm=(raw-raw.mean())/(raw.std()+1e-8)
loss=-(norm*(logp-oldlogp).exp()).mean();loss.backward();grad_original=means.grad.clone()
means2=torch.zeros_like(actions,requires_grad=True);valid=torch.tensor([[1.,1.,1.,0.]])
x=raw[valid.bool()];masked=(raw-x.mean())/(x.std()+1e-8)
logp2=torch.distributions.Normal(means2,torch.ones_like(means2)).log_prob(actions)
loss2=(-(masked*(logp2-oldlogp).exp())*valid).sum()/valid.sum();loss2.backward()
assert grad_original[0,-1]!=0 and means2.grad[0,-1]==0
# Real839 buffers: count clipping ofNormal actions, boundarycredit and inheritedAdam steps.
run=root/'tmp/cuda-research/cuda_auto_0839_takeoff';b=torch.load(run/'rollout_65536.pt',map_location='cpu',weights_only=True)
state=torch.load(run/'optimizer_final.pt',map_location='cpu',weights_only=True)
summary={'scope':'CPU-only installed C advantage operator and independent actor-loss gradient invariant, plus839existingartifact reads. No simulator/trainer/GPU launched. Boundaryissue is definite missingcredit/normalizationartifact; 3.125%scope does not establish hovercausality.','c_kernel_alignment_pass':True,'terminal_bootstrap_stop_pass':True,'first_reward_ignored_as_arriving_previous_action_pass':True,'expected_advantages':expected.tolist(),'actual_advantages':actual.tolist(),'uncredited_final_cell_normalized_credit':float(norm[0,-1]),'uncredited_final_cell_actor_gradient':float(grad_original[0,-1]),'mask_final_cell_actor_gradient':float(means2.grad[0,-1]),'839_actions_outside_unit_bounds_fraction':float((b['actions'].abs()>1).float().mean()),'839_actions_std_per_channel':b['actions'].reshape(-1,4).std(0).tolist(),'839_optimizer_steps':sorted({float(x['step']) for x in state['optimizer_state_dict']['state'].values()}),'c_source_sha256':hashlib.sha256(Path('/home/n33du/ai/puffer-tether-venv/lib/python3.12/site-packages/pufferlib/extensions/pufferlib.cpp').read_bytes()).hexdigest()}
(out/'boundary_invariant.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
