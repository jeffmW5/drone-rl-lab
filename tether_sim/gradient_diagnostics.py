"""Observe combined-loss gradients before and after global clipping; no mutation."""
import torch


class GradientDiagnostics:
    groups=('actor','value_head','shared_body')

    def __init__(self,policy,optimizer):
        self.policy=policy;self.raw={};self.steps=0
        device=next(policy.parameters()).device
        self.sums=torch.zeros(3,2,device=device);self.maxima=torch.zeros_like(self.sums)
        self.handles=[]
        for name,parameter in policy.named_parameters():
            def observe(gradient,name=name):
                self.raw[name]=gradient.detach().square().sum()
                return None
            self.handles.append(parameter.register_hook(observe))
        self.handles.append(optimizer.register_step_pre_hook(self.observe_step))

    def observe_step(self,optimizer,args,kwargs):
        parameters=dict(self.policy.named_parameters());rows=[]
        for group in self.groups:
            names=[name for name in parameters if
                   (group=='actor' and (name.startswith('actor.') or name=='log_std')) or
                   (group=='value_head' and name.startswith('value.')) or
                   (group=='shared_body' and name.startswith('body.'))]
            raw=torch.stack([self.raw[name] for name in names if name in self.raw]).sum().sqrt()
            post=torch.stack([parameters[name].grad.detach().square().sum() for name in names if parameters[name].grad is not None]).sum().sqrt()
            rows.append(torch.stack([raw,post]))
        observed=torch.stack(rows);self.sums+=observed;self.maxima=torch.maximum(self.maxima,observed)
        self.steps+=1;self.raw.clear()

    def summary(self):
        averages=(self.sums/max(1,self.steps)).cpu().tolist();maxima=self.maxima.cpu().tolist()
        return dict(optimizer_steps=self.steps,groups={group:dict(mean_before_clip=averages[i][0],mean_after_clip=averages[i][1],max_before_clip=maxima[i][0],max_after_clip=maxima[i][1]) for i,group in enumerate(self.groups)},
                    scope='Combined loss gradients; shared body cannot be attributed separately to actor or critic. Norms are not causal proof.')

    def close(self):
        for handle in self.handles:handle.remove()
