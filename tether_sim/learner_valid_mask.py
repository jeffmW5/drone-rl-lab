"""Opt-in final-column loss mask for the exact audited PuffeRL trainer.
"""
import ast,hashlib,textwrap,types
from pathlib import Path
EXPECTED_SOURCE='efc5ea7537670bf89ac18347bdc97126dd6f282b9ede5598424eaeace6f8f5f4'
REPLACEMENTS=(
 ('old_approx_kl = (-logratio).mean()', 'old_approx_kl = (-logratio[..., :-1]).mean()'),
 ('approx_kl = ((ratio - 1) - logratio).mean()', 'approx_kl = ((ratio[..., :-1] - 1) - logratio[..., :-1]).mean()'),
 ("clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()", "clipfrac = ((ratio[..., :-1] - 1.0).abs() > config['clip_coef']).float().mean()"),
 ('adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)', 'valid_adv = adv[..., :-1]\n            adv = mb_prio * (adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)'),
 ('pg_loss = torch.max(pg_loss1, pg_loss2).mean()', 'pg_loss = torch.max(pg_loss1, pg_loss2)[..., :-1].mean()'),
 ('v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()', 'v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped)[..., :-1].mean()'),
 ('entropy_loss = entropy.mean()', 'entropy_loss = entropy.reshape(mb_returns.shape)[..., :-1].mean()'),
 ("losses['importance'] += ratio.mean().item() / self.total_minibatches", "losses['importance'] += ratio[..., :-1].mean().item() / self.total_minibatches"),
 ('y_pred = self.values.flatten()', 'y_pred = self.values[..., :-1].flatten()'),
 ('y_true = advantages.flatten() + self.values.flatten()', 'y_true = advantages[..., :-1].flatten() + self.values[..., :-1].flatten()'),
)
def make_train(mask=True):
 import pufferlib.pufferl as module
 source=Path(module.__file__).read_bytes()
 if hashlib.sha256(source).hexdigest()!=EXPECTED_SOURCE:raise RuntimeError('Masked learner requires audited PufferLib source')
 text=source.decode();tree=ast.parse(text)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PuffeRL')
 node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 # Function starts at def, intentionally omitting external exception decorator.
 fn='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
 if mask:
  for old,new in REPLACEMENTS:
   if fn.count(old)!=1:raise RuntimeError('Audited train fragment mismatch: '+old)
   fn=fn.replace(old,new)
 fn=textwrap.dedent(fn)
 scope=module.__dict__;exec(compile(fn,'<audited_train_valid_mask_v001>' if mask else '<audited_train_baseline>','exec'),scope)
 return module.record(scope['train']),fn

def enable(trainer,version):
 if version not in (1,2):raise ValueError('Unsupported learner mask version')
 if version==1:return {'version':1,'enabled':False,'source_sha256':EXPECTED_SOURCE}
 if trainer.config['bptt_horizon']<2:raise ValueError('Need one valid action and bootstrap cell')
 fn,source=make_train();trainer.train=types.MethodType(__import__('pufferlib.pufferl',fromlist=['record']).record(fn),trainer)
 return {'version':2,'enabled':True,'source_sha256':EXPECTED_SOURCE,'patch_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'Finalcolumn retained for nextvalue bootstrap but excluded from normalization and policy/value/entropy loss and loss diagnostics. Sampler/priority/returns/value refresh/optimizer unchanged.'}

