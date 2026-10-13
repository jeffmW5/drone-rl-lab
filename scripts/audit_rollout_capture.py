"""CPU audit of immutable rollout captures; no trainer or CUDA initialization."""
import argparse,json,hashlib
from pathlib import Path
import torch
from tether_sim.cuda_task import navigation_priority_reward

def audit(path):
    x=torch.load(path,map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in x.values())
    o=x['observations']; v=x['values']; r=x['rewards']; d=x['terminals']
    assert o.shape==(128,32,98) and r.shape==(128,32)
    goal=o[:,:,-3:]; pos=o[:,:,:42].reshape(128,32,14,3)+goal[:,:,None,:]
    vel=o[:,:,42:84].reshape(128,32,14,3)
    q=o[:,:,84:88]
    expected=navigation_priority_reward(pos.flatten(0,1),vel.flatten(0,1),q.flatten(0,1),goal.flatten(0,1),torch.zeros(4096),12).reshape(128,32).clamp(-1,1)
    # Landing latch overrides reward to -1; terminal observation is reset.
    eligible=(d==0)&(r!=-1); eligible[:,0]=False
    diff=(expected-r).abs()[eligible]
    adv=torch.zeros_like(v); last=torch.zeros(128)
    # Exact scalar recurrence in installed CUDA source, initial ratios=1.
    # Reconstructed only for first minibatch; actual later advantages differ.
    for t in range(30,-1,-1):
        alive=1-d[:,t+1]
        delta=r[:,t+1]+.995*v[:,t+1]*alive-v[:,t]
        last=delta+.995*.95*last*alive; adv[:,t]=last
    dist=torch.linalg.vector_norm(o[:,:,39:42],dim=-1)
    progress=dist[:,:-1]-dist[:,1:]
    valid=d[:,1:]==0
    def group(mask):
        z=adv[:,:-1][mask&valid]
        return dict(count=z.numel(),mean_advantage=z.mean().item() if z.numel() else None,positive_fraction=(z>0).float().mean().item() if z.numel() else None)
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),finite=True,
        eligible_reward_cells=int(eligible.sum()),reward_formula_max_abs_error=diff.max().item() if diff.numel() else None,
        reward_min=r.min().item(),reward_max=r.max().item(),reward_mean=r.mean().item(),negative_one_cells=int((r==-1).sum()),terminal_cells=int(d.sum()),
        approaching=group(progress>1e-5),receding=group(progress < -1e-5),
        reconstructed_first_minibatch_advantage_mean=adv[:,:-1].mean().item(),
        scope='Incoming reward matches current observation; outgoing action at t uses reward t+1 in installed recurrence. CPU reconstruction is initial-ratio first-minibatch only, not saved actual advantages; observational groups are not causal action rankings.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=[audit(f) for f in sorted(a.run.glob('rollout_*.pt'))]
    if len(reports)!=2:raise RuntimeError('Expected initial/final captures')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
