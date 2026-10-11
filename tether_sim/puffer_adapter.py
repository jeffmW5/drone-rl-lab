"""PufferLib 3 bridge retaining the existing MuJoCo navigation task."""
import pickle
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from .generalize import make_task


class PufferTask(gym.Wrapper):
    def __init__(self, config, normalizer_path):
        super().__init__(make_task(config))
        with open(normalizer_path, 'rb') as file:
            norm = pickle.load(file)
        self.reward_std = float(np.sqrt(norm.ret_rms.var + norm.epsilon))
        self.reward_clip = norm.clip_reward
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               self.env.observation_space.shape, np.float32)
        self.episode_return = 0.
        self.episode_steps = 0
        self.failure_dir = config.get('numerical_failure_dir')

    def reset(self, *, seed=None, options=None):
        self.reset_seed = None if seed is None else int(seed)
        obs, _ = self.env.reset(seed=self.reset_seed, options=options)
        self.episode_return = 0.
        self.episode_steps = 0
        return obs.astype(np.float32), {}

    def step(self, action):
        if self.failure_dir:
            sim = self.env.env
            before = dict(time=float(sim.data.time), qpos=sim.data.qpos.tolist(),
                          qvel=sim.data.qvel.tolist(), motors=sim.motors.tolist(),
                          wind=sim.wind.tolist(), target=sim.target.tolist(),
                          action=np.asarray(action).tolist(), domain=self.env.domain,
                          physics=asdict(sim.config), reset_seed=self.reset_seed,
                          episode_steps=self.episode_steps)
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except FloatingPointError as error:
            if self.failure_dir:
                folder = Path(self.failure_dir)
                folder.mkdir(parents=True, exist_ok=True)
                before['error'] = str(error)
                (folder/f'{os.getpid()}_{time.time_ns()}.json').write_text(json.dumps(before, indent=2))
            raise  # Numerical failures remain invalid, never a successful/reset episode.
        self.episode_return += reward
        self.episode_steps += 1
        metrics = {}
        if terminated or truncated:
            metrics = dict(episode_return=self.episode_return, episode_steps=self.episode_steps,
                           collision=float(info.get('obstacle_collision', False)),
                           goal_distance_m=info['goal_distance_m'],
                           success=float(not terminated and info['goal_reached']
                                         and info['payload_goal_error_m'] < .08 and info['swing_deg'] < 10))
            if 'precision_success' in info:
                metrics.update(success=float(not terminated and info['precision_success']),
                               payload_goal_error_mm=info['payload_goal_error_mm'],
                               precision_hold_seconds=info['precision_hold_seconds'])
        scaled = float(np.clip(reward / self.reward_std, -self.reward_clip, self.reward_clip))
        # PufferLib 3's advantage implementation does not separately consume
        # truncations. Treat finite task duration as terminal to prevent returns
        # leaking across auto-reset episodes. This differs from SB3 time-limit
        # bootstrapping and is recorded in each experiment's provenance.
        return obs.astype(np.float32), scaled, bool(terminated or truncated), bool(truncated), metrics


def make_puffer_env(config, normalizer_path, buf=None, seed=0):
    import pufferlib.emulation
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=PufferTask(config, normalizer_path), buf=buf, seed=seed)


class TransferredPolicy(torch.nn.Module):
    """Expose an SB3 Gaussian policy through PufferLib's policy interface."""
    def __init__(self, policy, normalizer):
        super().__init__()
        self.policy = policy
        self.register_buffer('obs_mean', torch.as_tensor(normalizer.obs_rms.mean, dtype=torch.float32))
        self.register_buffer('obs_var', torch.as_tensor(normalizer.obs_rms.var, dtype=torch.float32))
        self.epsilon = normalizer.epsilon
        self.clip_obs = normalizer.clip_obs

    def normalize(self, observations):
        return torch.clamp((observations.float() - self.obs_mean) /
                           torch.sqrt(self.obs_var + self.epsilon), -self.clip_obs, self.clip_obs)

    def forward(self, observations, state=None):
        obs = self.normalize(observations)
        latent_pi, latent_vf = self.policy.mlp_extractor(self.policy.extract_features(obs))
        distribution = self.policy._get_action_dist_from_latent(latent_pi).distribution
        return distribution, self.policy.value_net(latent_vf)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)
