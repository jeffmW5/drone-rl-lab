"""Read-only capture of actual Puffer advantage calls in one bounded train update."""
import hashlib,json
from pathlib import Path
import torch

def capture_train(trainer,out,expected_hash):
    import pufferlib.pufferl as module
    digest=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest!=expected_hash:raise RuntimeError('Advantage capture requires audited learner source')
    original=module.compute_puff_advantage;original_sample=torch.multinomial;calls=[];samples=[]
    def observed(*args,**kwargs):
        before={name:args[i].detach().cpu().clone() for i,name in enumerate(('values','rewards','terminals','ratio'))}
        result=original(*args,**kwargs)
        before['advantages']=result.detach().cpu().clone()
        index=len(calls);path=out/f'advantages_{trainer.global_step}_{index:02d}.pt'
        torch.save(before,path);calls.append(str(path.name))
        return result
    def sampled(*args,**kwargs):
        result=original_sample(*args,**kwargs)
        index=len(samples);path=out/f'minibatch_indices_{trainer.global_step}_{index:02d}.pt'
        torch.save({'indices':result.detach().cpu().clone(),'probabilities':args[0].detach().cpu().clone(),'preceding_advantage_call':len(calls)-1},path)
        samples.append(path.name)
        return result
    module.compute_puff_advantage=observed
    torch.multinomial=sampled
    try:trainer.train()
    finally:
        module.compute_puff_advantage=original
        torch.multinomial=original_sample
    (out/f'advantages_{trainer.global_step}.json').write_text(json.dumps({'version':2,'source_sha256':digest,'calls':calls,'samples':samples,'scope':'Actual raw recomputed advantages and input values/rewards/terminals/ratios before minibatch selection/normalization. Detached copies; no modified tensors or RNG.'},indent=2))
