import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,torch,types,contextlib
from pathlib import Path
from tether_sim.learner_valid_mask import make_train,enable
import pufferlib.pufferl as module
out=Path('tmp/deep_audit_root/valid_mask_test');out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
class Profile:
 def __call__(self,*a,**kw):pass
 def end(self):pass
 def clear(self):pass
class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.means=torch.nn.Parameter(torch.zeros(8));self.values=torch.nn.Parameter(torch.zeros(8))
 def forward(self,obs,state=None):
  ids=obs[:,0].long();return torch.distributions.Normal(self.means[ids,None],torch.ones(len(ids),1)),self.values[ids,None]
def trainer():
 p=Policy();v=torch.zeros(2,4);r=torch.tensor([[0.,1.,0.,-1.],[0.,1.,0.,-1.]]);d=torch.zeros_like(v);d[:,-1]=1
 obs=torch.arange(8).view(2,4,1).float();actions=torch.ones(2,4,1)
 logp=torch.distributions.Normal(torch.zeros_like(actions),torch.ones_like(actions)).log_prob(actions).sum(-1)
 cfg=dict(prio_beta0=0.,prio_alpha=0.,clip_coef=.2,vf_clip_coef=.2,gamma=.995,gae_lambda=.95,vtrace_rho_clip=1.,vtrace_c_clip=1.,use_rnn=False,vf_coef=.5,ent_coef=0.,max_grad_norm=1e9,anneal_lr=False,checkpoint_interval=100,total_timesteps=8,device='cpu',bptt_horizon=4)
 t=types.SimpleNamespace(profile=Profile(),epoch=0,config=cfg,ratio=torch.ones_like(v),values=v,rewards=r,terminals=d,truncations=torch.zeros_like(v),observations=obs,actions=actions,logprobs=logp,segments=2,minibatch_segments=2,total_minibatches=1,total_epochs=1,amp_context=contextlib.nullcontext(),vecenv=types.SimpleNamespace(single_observation_space=types.SimpleNamespace(shape=(1,))),policy=p,optimizer=torch.optim.Adam(p.parameters(),lr=.01,eps=1e-5),accumulate_minibatches=1,global_step=8,last_log_time=0.,mean_and_log=lambda:None,print_dashboard=lambda:None,save_checkpoint=lambda:None)
 return t
base=trainer();masked=trainer();rng=torch.get_rng_state();basefn,base_source=make_train(False);basefn(base);torch.set_rng_state(rng);manifest=enable(masked,2);masked.train()
last=torch.tensor([3,7]);valid=torch.tensor([0,1,2,4,5,6])
assert (base.policy.means[last].abs()>0).all()
assert torch.equal(masked.policy.means[last],torch.zeros(2))
assert torch.equal(masked.policy.values[last],torch.zeros(2))
# Exact candidate gradient versus independent valid-cell equations, same probability and initial predictions.
ref=trainer();adv=torch.zeros_like(ref.values);torch.ops.pufferlib.compute_puff_advantage(ref.values,ref.rewards,ref.terminals,ref.ratio,adv,.995,.95,1.,1.)
dist,values=ref.policy(ref.observations.reshape(-1,1));newlogp=dist.log_prob(ref.actions.reshape(-1,1)).sum(-1).reshape(2,4);ratio=(newlogp-ref.logprobs).exp();valid_adv=adv[:,:-1];norm=(valid_adv-valid_adv.mean())/(valid_adv.std()+1e-8)
pg=-(norm*ratio[:,:-1]).mean();vl=.5*(values.reshape(2,4)[:,:-1]-valid_adv).square().mean();loss=pg+.5*vl;loss.backward();ref.optimizer.step();ref.optimizer.zero_grad()
assert torch.allclose(ref.policy.means,masked.policy.means,atol=1e-7)
assert torch.allclose(ref.policy.values,masked.policy.values,atol=1e-7)
assert abs(masked.losses['policy_loss']-float(pg.detach()))<1e-7
assert abs(masked.losses['value_loss']-float(vl.detach()))<1e-7
(out/'train_valid_mask_candidate.py').write_text(make_train(True)[1])
result={'scope':'Independent CPU-only actual copied source guarded by exact installed hash, one analytic-table loss/update. This is a learner invariant test, not simulated flight or production training. No package/source change.','mask_manifest':manifest,'baseline_last_actor_weights':base.policy.means[last].tolist(),'masked_last_actor_weights':masked.policy.means[last].tolist(),'masked_last_critic_weights':masked.policy.values[last].tolist(),'reference_valid_cell_actor_maxdiff':float((ref.policy.means-masked.policy.means).abs().max()),'reference_valid_cell_critic_maxdiff':float((ref.policy.values-masked.policy.values).abs().max()),'baseline_losses':dict(base.losses),'masked_losses':dict(masked.losses),'independent_equations_match_pass':True,'baseline_synthetic_no_credit_cells_changed':True,'candidate_synthetic_no_credit_cells_unchanged':True}
(out/'valid_mask_cpu_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
