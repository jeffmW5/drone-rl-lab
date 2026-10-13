"""CPU audit of production adapter and CUDA equation signs; no GPU trainer."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tether_sim.cuda_task import CudaVector, approach_stop_reward

source = Path('tether_sim/cuda/physics_v001.cu').read_text()
torque_expr = re.search(r'float torque\[3\]=\{(.*?)\};', source, re.S).group(1).split(',')
axis_expr = re.search(r'float axis\[3\]=\{(.*?)\};', source, re.S).group(1).split(',')

def expression(expr, variables):
    return eval(re.sub(r'(\d)f\b', r'\1', expr).strip(), {'__builtins__': {}}, variables)

def torque(motors):
    return np.array([expression(e, {'motors': motors}) for e in torque_expr])

def thrust_axis(q):
    return np.array([expression(e, dict(zip('wxyz', q))) for e in axis_expr])

class FakePhysics:
    def __init__(self):
        self.position = torch.zeros(1, 14, 3)
        self.position[:, 0, 2] = .8
        self.velocity = torch.zeros_like(self.position)
        self.orientation = torch.zeros(1, 7)
        self.orientation[:, 0] = 1
        self.motors = torch.zeros(1, 4)
        self.failed = torch.zeros(1, dtype=torch.int32)
        self.crash = False
    def step(self, command, dt=.001, substeps=20):
        self.command = command.clone()
        if self.crash:
            self.failed[:] = 1

def vector():
    vec = object.__new__(CudaVector)
    vec.sim = FakePhysics()
    vec.integration_version = 1
    vec.diagnostic_zone_version = 0
    vec.reset_motor_forces = torch.zeros(1,4)
    vec.direct = True
    vec.control_version = 2
    vec.reward_version = 8
    vec.goal = torch.zeros(1, 3)
    vec.steps = torch.zeros(1, dtype=torch.int32)
    vec.previous_error = torch.zeros(1)
    vec.start = vec.sim.position.clone()
    vec.training_region = None
    vec.terminal_observer = None
    return vec

real_tensor, real_as_tensor = torch.tensor, torch.as_tensor
def cpu_call(fn):
    def wrapped(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return fn(*args, **kwargs)
    return wrapped

checks = []
with patch('torch.tensor', cpu_call(real_tensor)), patch('torch.as_tensor', cpu_call(real_as_tensor)):
    vec = vector()
    vec.send(np.zeros((1, 4)))
    assert np.allclose(vec.sim.command.numpy(), .5)
    assert np.allclose(torque(vec.sim.command.numpy()[0]), 0)
    checks.append('equal motors give positive collective and zero roll/pitch/yaw torque')
    for channel in range(1, 4):
        a = np.zeros((1, 4)); a[0, channel] = 1
        vec.send(a)
        positive = torque(vec.sim.command.numpy()[0])
        a[0, channel] = -1
        vec.send(a)
        negative = torque(vec.sim.command.numpy()[0])
        assert positive[channel-1] > 0
        assert np.allclose(negative, -positive, atol=1e-9)
        assert np.allclose(np.delete(positive, channel-1), 0, atol=1e-9)
    checks.append('production mixer and parsed CUDA torques agree for both signs of all three axes')
    for collective, expected in [(-1, 0), (1, 1)]:
        vec.send(np.array([[collective, 0, 0, 0]]))
        assert np.allclose(vec.sim.command.numpy(), expected)
    checks.append('collective -1 means motors off; +1 means full thrust')
    vec = vector()
    assert vec.obs().shape == (1, 98)
    vec.goal[:] = torch.tensor([[1, 2, 3]])
    assert torch.allclose(vec.obs()[:, :3], vec.sim.position[:, 0]-vec.goal)
    checks.append('98 observation entries; position observation is position minus goal')
    for axis in range(3):
        for direction in [-1, 1]:
            vec.goal.zero_(); vec.goal[0, axis] = direction
            vec.sim.position[:, -1] = 0
            vec.sim.velocity.zero_(); vec.sim.velocity[:, -1, axis] = .4*direction
            toward = approach_stop_reward(vec.sim.position, vec.sim.velocity, vec.sim.orientation, vec.goal, vec.sim.failed)
            vec.sim.velocity[:, -1, axis] *= -1
            away = approach_stop_reward(vec.sim.position, vec.sim.velocity, vec.sim.orientation, vec.goal, vec.sim.failed)
            assert toward > away
    checks.append('toward target beats away on all +/- world axes; reward uses goal minus position')
    vec = vector(); vec.sim.crash = True
    vec.sim.position[:, -1, 0] = 4
    captured = []
    vec.terminal_observer = lambda obs, done, truncated, failed, reward: captured.append((obs.clone(), reward.clone()))
    vec.send(np.zeros((1, 4)))
    assert vec.done.item() and not vec.truncated.item()
    assert vec.reward.item() < -1 and captured[0][1].item() == vec.reward.item()
    assert captured[0][0][0, 39].item() == 4
    assert torch.equal(vec.sim.position, vec.start) and vec.sim.failed.item() == 0
    checks.append('crash keeps negative terminal reward and pre-reset observation, then clears physics state')
    vec = vector(); vec.steps[:] = 499
    vec.send(np.zeros((1, 4)))
    assert vec.done.item() and vec.truncated.item() and vec.steps.item() == 0
    checks.append('500-step timeout ends and resets episode')

assert np.allclose(thrust_axis([1, 0, 0, 0]), [0, 0, 1])
angle = .1
assert thrust_axis([np.cos(angle/2), np.sin(angle/2), 0, 0])[1] < 0
assert thrust_axis([np.cos(angle/2), 0, np.sin(angle/2), 0])[0] > 0
assert np.allclose(thrust_axis([0, 1, 0, 0]), [0, 0, -1])
checks.append('identity thrust points up; positive roll tilts toward -Y, positive pitch toward +X; inverted thrust points down')
assert '-(d==2?9.81f:0)' in source
assert '(length-params[0]/(B-1))/length' in source
for length, expected_direction in [(1., 1), (.01, -1)]:
    correction = (length-.6/13)/length
    assert np.sign(correction) == expected_direction
assert 0 < (.0434+.001+.005)*9.81/(4*.2) < 1
checks.append('gravity subtracts Z; stretched-link projection pulls endpoints together, compressed links push apart')
print(json.dumps(dict(status='pass', checks=checks, scope='CPU production-adapter tests plus source-equation audit; not a compiled CUDA dynamics or hardware test'), indent=2))
