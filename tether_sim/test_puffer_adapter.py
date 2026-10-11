"""Physics/normalization/transfer checks; run in the PufferLib environment."""
import json
import pickle
import pytest
pytest.importorskip('pufferlib')
import numpy as np
import torch
from stable_baselines3 import PPO
from .generalize import make_task
from .puffer_adapter import PufferTask, TransferredPolicy, make_puffer_env


def test_adapter_retains_rollout_and_terminal_contacts():
    config = json.load(open('configs/tether_puffer_v001.json'))
    source = config['resume_run'] + '/interrupted_normalize.pkl'
    raw = make_task(config)
    wrapped = PufferTask(config, source)
    a, _ = raw.reset(seed=51001)
    b, _ = wrapped.reset(seed=51001)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    for i in range(100):
        action = np.sin(i + np.arange(4)).astype(np.float32) * .2
        a, r, d, t, info = raw.step(action)
        b, q, e, u, _ = wrapped.step(action)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        assert np.isclose(q, np.clip(r/wrapped.reward_std, -wrapped.reward_clip, wrapped.reward_clip))
        assert e == (d or t) and u == t
    # Place a physical rotor inside an obstacle; both paths must latch contact.
    box = raw.env.config.obstacles[0]
    for task in (raw, wrapped.env):
        task.env.data.qpos[:3] = box[:3]
        import mujoco
        mujoco.mj_forward(task.env.model, task.env.data)
    _, _, d, _, info = raw.step(np.zeros(4, dtype=np.float32))
    _, _, e, _, metrics = wrapped.step(np.zeros(4, dtype=np.float32))
    assert d and e and info['obstacle_collision'] and metrics['collision'] == 1
    raw.close(); wrapped.close()


def test_policy_transfer_matches_sb3_means_and_values():
    config = json.load(open('configs/tether_puffer_v001.json'))
    source = config['resume_run']
    model = PPO.load(source+'/interrupted.zip', device='cpu')
    with open(source+'/interrupted_normalize.pkl', 'rb') as file:
        normalizer = pickle.load(file)
    policy = TransferredPolicy(model.policy, normalizer)
    env = make_task(config)
    observations = np.stack([env.reset(seed=seed)[0] for seed in (51001,51002,51003)])
    expected, _ = model.predict(normalizer.normalize_obs(observations), deterministic=True)
    with torch.no_grad():
        distribution, value = policy(torch.as_tensor(observations, dtype=torch.float32))
        expected_value = model.policy.predict_values(torch.as_tensor(normalizer.normalize_obs(observations)))
    np.testing.assert_allclose(np.clip(distribution.mean.numpy(), -1, 1), expected, atol=1e-6)
    np.testing.assert_allclose(value.numpy(), expected_value.numpy(), atol=1e-5)
    env.close()


def test_puffer_buffer_and_duration_boundary():
    config = json.load(open('configs/tether_puffer_v001.json'))
    config['episode_seconds'] = .04
    env = make_puffer_env(config, config['resume_run']+'/interrupted_normalize.pkl')
    obs, _ = env.reset(seed=51001)
    assert obs.dtype == np.float32 and obs.shape == (1, env.single_observation_space.shape[0])
    env.step(np.zeros((1,4), dtype=np.float32))
    _, _, done, truncated, metrics = env.step(np.zeros((1,4), dtype=np.float32))
    assert done and truncated and metrics['episode_steps'] == 2
    assert env.done
    env.reset()
    assert not env.done and not env.terminals[0] and not env.truncations[0]
    env.close()


def test_compiled_cuda_advantages_match_cpu():
    import pufferlib.pufferl  # Register the upstream CPU and CUDA operators.
    if not torch.cuda.is_available() or not torch._C._dispatch_has_kernel_for_dispatch_key('pufferlib::compute_puff_advantage', 'CUDA'):
        pytest.skip('Run the CUDA kernel parity check in the compiled PufferLib runtime')
    torch.manual_seed(73)
    values = torch.randn(4, 64)
    rewards = torch.randn(4, 64)
    dones = torch.zeros(4, 64)
    dones[:, 20] = 1
    ratio = torch.ones(4, 64)
    cpu = torch.zeros_like(values)
    gpu = cpu.cuda()
    torch.ops.pufferlib.compute_puff_advantage(values, rewards, dones, ratio, cpu, .995, .95, 1., 1.)
    torch.ops.pufferlib.compute_puff_advantage(values.cuda(), rewards.cuda(), dones.cuda(), ratio.cuda(), gpu, .995, .95, 1., 1.)
    np.testing.assert_allclose(gpu.cpu().numpy(), cpu.numpy(), rtol=1e-5, atol=1e-5)
