"""Version1: post-clipping timeout bootstrap for synchronous PufferLib3.
Does not change terminal masks or installed PufferLib. Fail closed on alignment.
"""
import torch

class TimeoutBootstrap:
    def __init__(self, vec, policy, gamma):
        self.vec, self.policy, self.gamma = vec, policy, gamma
        self.trainer = None
        self.pending = None
        self.staged = []
        self.applied = 0
        self.original_recv = vec.recv
        self.original_observer = vec.terminal_observer
        vec.terminal_observer = self.observe
        vec.recv = self.recv

    def observe(self, obs, done, truncated, failed, reward):
        if self.original_observer is not None:
            self.original_observer(obs, done, truncated, failed, reward)
        if self.pending is not None:
            raise RuntimeError('Unconsumed terminal bootstrap')
        ids = torch.nonzero(truncated & (failed == 0)).flatten()
        if ids.numel():
            with torch.no_grad():
                _, value = self.policy(obs)
                self.pending = (ids.clone(), value.flatten()[ids].detach().clone(), reward[ids].clamp(-1, 1).clone())

    def recv(self):
        result = self.original_recv()
        if self.pending is not None:
            if self.trainer is None:
                raise RuntimeError('Bootstrap trainer not attached')
            ids, values, clipped = self.pending
            env_ids = torch.as_tensor(result[5], device=ids.device)
            if not torch.equal(env_ids, torch.arange(self.vec.num_agents, device=ids.device)):
                raise RuntimeError('Bootstrap requires synchronous contiguous full-agent batch')
            rows = self.trainer.ep_indices[ids].long().clone()
            columns = self.trainer.ep_lengths[ids].long().clone()
            self.staged.append((rows, columns, values, clipped))
            self.pending = None
        return result

    def apply(self):
        # evaluate has finished copying/clipping rewards, but train has not run.
        with torch.no_grad():
            for rows, columns, values, clipped in self.staged:
                rewards = self.trainer.rewards[rows, columns]
                terminals = self.trainer.terminals[rows, columns]
                if not bool((terminals == 1).all()) or not torch.allclose(rewards, clipped):
                    raise RuntimeError('Timeout reward/terminal buffer alignment mismatch')
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError('Nonfinite terminal critic value')
                self.trainer.rewards[rows, columns] = rewards + self.gamma * values
                self.applied += len(rows)
        self.staged.clear()
