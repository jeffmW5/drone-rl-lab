import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,json,hashlib
from pathlib import Path
from pufferlib import _C
out=Path(__file__).parent;torch.set_num_threads(1)
values=torch.zeros(2,2);rewards=torch.tensor([[0.,1.],[0.,-1.]]);terminals=torch.tensor([[0.,1.],[0.,1.]]);ratio=torch.ones_like(values);adv=torch.zeros_like(values)
torch.ops.pufferlib.compute_puff_advantage(values,rewards,terminals,ratio,adv,.995,.95,1.,1.)
assert torch.equal(adv,torch.tensor([[1.,0.],[-1.,0.]]))
mean=torch.tensor(0.,requires_grad=True);actions=torch.tensor([1.,-1.]);dist=torch.distributions.Normal(mean.expand(2),torch.ones(2));logp=dist.log_prob(actions);oldlogp=logp.detach();normalized=adv[:,0]/adv[:,0].std();loss=-(normalized*(logp-oldlogp).exp()).mean();loss.backward();gradient=float(mean.grad)
assert gradient<0
# Aligned creditedGaussian gradient ascent shouldincreasepositivecreditedaction anddecreasenegative.
next_mean=mean.detach()-.01*mean.grad;newdist=torch.distributions.Normal(next_mean.expand(2),torch.ones(2));change=newdist.log_prob(actions)-oldlogp
assert change[0]>0 and change[1]<0
# Exact sampledmultinomial hasno replacement within each minibatch, revisitsacross12; expectedcoverage96.832%; not3completepermutedepochs.
summary={'scope':'CPU installedC recurrence and exactGaussian PPO likelihood-ratio actor-loss equation. Non-simulation analytic gradient invariant, nothoverresult.','arriving_transition_advantages':adv.tolist(),'actor_loss_mean_gradient':gradient,'positive_credit_action_logprob_change':float(change[0]),'negative_credit_action_logprob_change':float(change[1]),'passed':True,'actual_configuration_gamma_lambda':.995*.95,'credit_exponential_time_constant_seconds':-.02/float(torch.tensor(.995*.95).log()),'credit_weight_at_31steps':(.995*.95)**31,'sampling_expected_unique_rows_128_12draws32':128*(1-(1-32/128)**12)}
(out/'signed_credit_invariant.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
