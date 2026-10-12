"""Version1 potential shaping after learner reward clipping; no controller helper."""
import torch

class PotentialShaping:
    def __init__(self, vec, gamma):
        self.vec, self.gamma = vec, gamma
        self.trainer=None;self.pending=None;self.staged=[];self.applied=0
        self.original_send, self.original_recv = vec.send, vec.recv
        vec.send, vec.recv = self.send, self.recv

    def potential(self):
        error=torch.linalg.vector_norm(self.vec.sim.position[:,-1]-self.vec.goal,dim=1)
        return -error/(1+error)

    def send(self, actions):
        if self.pending is not None:raise RuntimeError('Unconsumed shaping transition')
        before=self.potential().clone()
        self.original_send(actions)
        after=torch.where(self.vec.done,0,self.potential())
        self.pending=(self.gamma*after-before).detach()

    def recv(self):
        result=self.original_recv()
        if self.pending is not None:
            if self.trainer is None:raise RuntimeError('Shaping trainer not attached')
            ids=torch.as_tensor(result[5],device=self.pending.device).long()
            if not torch.equal(ids,torch.arange(self.vec.num_agents,device=ids.device)):
                raise RuntimeError('Shaping requires synchronous full-agent batch')
            self.staged.append((self.trainer.ep_indices[ids].long().clone(),
                self.trainer.ep_lengths[ids].long().clone(),self.pending.clone(),
                result[1].clamp(-1,1).clone(),result[2].float().clone()))
            self.pending=None
        return result

    def apply(self):
        with torch.no_grad():
            for rows,cols,shaping,clipped,done in self.staged:
                if not torch.allclose(self.trainer.rewards[rows,cols],clipped) or not torch.equal(self.trainer.terminals[rows,cols],done):
                    raise RuntimeError('Shaping reward/terminal alignment mismatch')
                if not bool(torch.isfinite(shaping).all()):raise RuntimeError('Nonfinite shaping')
                self.trainer.rewards[rows,cols]+=shaping.to(self.trainer.rewards.dtype)
                self.applied+=len(rows)
        self.staged.clear()
