"""CPU contract tests for recovery2; no CUDA context or physics execution."""
import hashlib
import argparse
import json
import math
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from types import SimpleNamespace

import torch

from tether_sim.cuda_task import CudaPolicy
from tether_sim.recovery_probe_v002 import _validate_plan, probe


class FakeCudaRng:
    def __init__(self):
        self.state = torch.tensor([17, 29, 43], dtype=torch.uint8)
        self.get_calls = 0
        self.set_calls = 0

    def get_rng_state(self):
        self.get_calls += 1
        return self.state.clone()

    def set_rng_state(self, state):
        self.set_calls += 1
        self.state = state.clone()


class FakeVector:
    """Exercise actual probe orchestration with deterministic toy state changes.

    The ground lane fails at step3 and is reset to ideal hover afterwards; an
    unrelated lane truncates at step5. Neither may count reset states as recovery.
    This tests bookkeeping, not simulator stability.
    """
    def __init__(self, count, rng, **kwargs):
        expected = dict(direct=True, reward_version=15, phase='takeoff',
                        control_version=2, physics_version=5,
                        training_reset_version=3, diagnostic_zone_version=1)
        assert kwargs == expected
        self.count = count
        self.rng = rng
        self.sim = SimpleNamespace(
            position=torch.zeros(count, 14, 3, dtype=torch.float64),
            velocity=torch.zeros(count, 14, 3, dtype=torch.float64),
            orientation=torch.zeros(count, 7), motors=torch.zeros(count, 4),
            mass=torch.tensor([.0434] + [.001 / 12] * 12 + [.005]).repeat(count, 1))
        self.start = torch.zeros_like(self.sim.position)
        fraction = torch.arange(14).double() / 13
        self.start[:, :, 0] = fraction * math.sqrt(.6 ** 2 - .025 ** 2)
        self.start[:, :, 2] = .05 - fraction * .025
        self.goal = torch.zeros(count, 3)
        self.goal[:, 2] = .2
        self.steps = torch.zeros(count, dtype=torch.int32)
        self.terminal_observer = None
        self.send_calls = 0
        self.initial_q = None

    def async_reset(self, seed):
        torch.manual_seed(seed)
        torch.rand(4)  # Deliberately consume CPU RNG to test restoration.
        self.rng.state.add_(7)  # Same contract check for injected CUDA RNG.

    def obs(self):
        s = self.sim
        return torch.cat(((s.position - self.goal[:, None]).flatten(1),
                          s.velocity.flatten(1), s.orientation, s.motors,
                          self.goal), dim=1).float()

    def send(self, actions):
        assert tuple(actions.shape) == (self.count, 4)
        assert bool((self.steps == 0).all()), 'Diagnostic timeouts must remain disabled'
        self.send_calls += 1
        if self.initial_q is None:
            self.initial_q = self.sim.orientation.clone()
        self.steps += 1
        done = torch.zeros(self.count, dtype=torch.bool)
        truncated = done.clone()
        failed = torch.zeros(self.count, dtype=torch.int32)
        reward = torch.zeros(self.count)
        if self.send_calls == 3:
            self.sim.position[0, 0] = torch.tensor([2.01, 0, .82])
            self.sim.velocity[0, 0] = torch.tensor([1.5, 0, 0])
            done[0] = True
            failed[0] = 3
            reward[0] = -1
        if self.send_calls == 5:
            done[4] = truncated[4] = True
        if bool(done.any()):
            self.terminal_observer(self.obs(), done, truncated, failed, reward)
            self.sim.position[done, :, :2] = 0
            self.sim.position[done, :, 2] = .8 - torch.arange(14).double() * .6 / 13
            self.sim.velocity[done] = 0
            self.sim.orientation[done] = 0
            self.sim.orientation[done, 0] = 1
            self.steps[done] = 0


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('tmp/deep_audit_physics/recovery_v002_cpu_checks.json'))
    output=parser.parse_args().output
    if output.exists():
        raise RuntimeError('Refuse overwrite existing CPU evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    policy = CudaPolicy()
    policy.configure_collective_observation(2)
    original = {key: value.clone() for key, value in policy.state_dict().items()}
    plan = dict(physics_version=5, control_version=2, reward_version=15,
                diagnostic_zone_version=1)
    rejected = []
    for key, value in (('physics_version', 6), ('control_version', 1),
                       ('reward_version', 14), ('diagnostic_zone_version', 0)):
        bad = dict(plan, **{key: value})
        try:
            _validate_plan(bad)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError('Unsupported diagnostic context accepted')
    for entry in ({'checkpoint': 'missing.pt'},
                  {'current_policy': True, 'checkpoint': 'missing.pt'}):
        try:
            _validate_plan(dict(plan, recovery_models=[entry]))
        except ValueError:
            pass
        else:
            raise AssertionError('Ambiguous/missing model metadata accepted')

    rng = FakeCudaRng()
    instances = []

    def factory(count, **kwargs):
        vec = FakeVector(count, rng, **kwargs)
        instances.append(vec)
        return vec

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        observations = torch.zeros(4, 98)
        positions = torch.zeros(14, 3)
        positions[:, 2] = .6 - torch.arange(14) * .6 / 13
        observations[:, :42] = positions.flatten()
        observations[:, 84] = 1
        observations[:, 91:95] = .0494 * 9.81 / 4
        observations[:, 97] = .2
        observations[0, 42] = .1
        observations[1, 42] = 1.5
        observations[2, 42] = .05  # Excluded by terminal flag.
        observations[3, 42] = float('nan')  # Excluded by finite-state guard.
        rollout = temporary / 'visited.pt'
        torch.save(dict(observations=observations,
                        terminals=torch.tensor([0., 0., 1., 0.])), rollout)
        checkpoint = temporary / 'policy.pt'
        torch.save(original, checkpoint)
        configured = dict(plan, recovery_rollout=str(rollout),
                          recovery_models=[{'current_policy': True},
                                           {'checkpoint': str(checkpoint),
                                            'collective_observation_version': 1}])
        cpu_before = torch.get_rng_state().clone()
        gpu_before = rng.state.clone()
        result = probe(policy, configured, vector_factory=factory, cuda_rng=rng)
        assert torch.equal(torch.get_rng_state(), cpu_before)
        assert torch.equal(rng.state, gpu_before)
        assert rng.get_calls == rng.set_calls == 1
        assert result['visited_evidence']['selected_indices'] == dict(visited_slow=0, visited_fast=1)
        assert result['visited_evidence']['eligible_counts'] == dict(visited_slow=1, visited_fast=1)
        assert result['models'][0]['current_policy'] is True
        assert result['models'][0]['collective_observation_version'] == 2
        assert result['models'][1]['current_policy'] is False
        assert result['models'][1]['collective_observation_version'] == 1
        assert len(result['models']) == 2
        for model, vec in zip(result['models'], instances):
            cases = {case['name']: case for case in model['cases']}
            assert len(cases) == 16
            failed = cases['ground']
            assert failed['first_failure']['time_s'] == .06
            assert failed['first_failure']['failed'] == 3
            assert failed['first_failure']['observation'][0] > 2
            assert failed['peak_hold_s'] == 0 and not failed['hover_success']
            assert not failed['survived_20s'] and failed['first_landing_s'] is None
            assert cases['pitch_plus10']['first_failure'] is None
            assert cases['pitch_plus10']['first_terminal']['truncated']
            assert not cases['pitch_plus10']['survived_20s']
            assert cases['ideal_hover']['survived_20s']
            assert model['elapsed_simulated_s'] == 20 and vec.send_calls == 1000
            for name, axis, angle in [('roll_plus2', 1, 2), ('roll_minus2', 1, -2),
                                      ('pitch_plus2', 2, 2), ('pitch_minus2', 2, -2)]:
                row = next(i for i, case in enumerate(model['cases']) if case['name'] == name)
                q = vec.initial_q[row]
                assert abs(float(q[0]) - math.cos(math.radians(angle) / 2)) < 1e-7
                assert abs(float(q[axis]) - math.sin(math.radians(angle) / 2)) < 1e-7
            assert all(not frame['active'][0] for frame in model['frames'] if frame['time_s'] >= .06)
        assert all(torch.equal(policy.state_dict()[key], value) for key, value in original.items())
        assert policy.collective_observation_version == 2 and policy.training

        # Errors after RNG capture must restore both streams as well.
        cpu_before = torch.get_rng_state().clone()
        gpu_before = rng.state.clone()

        def broken_factory(count, **kwargs):
            torch.rand(3)
            rng.state.add_(11)
            raise RuntimeError('Deliberate factory failure')

        try:
            probe(policy, plan, vector_factory=broken_factory, cuda_rng=rng)
        except RuntimeError as error:
            assert str(error) == 'Deliberate factory failure'
        else:
            raise AssertionError('Expected factory error')
        assert torch.equal(torch.get_rng_state(), cpu_before)
        assert torch.equal(rng.state, gpu_before)
        assert rng.get_calls == rng.set_calls == 2

    paths = ['tether_sim/recovery_probe_v002.py', 'scripts/check_recovery_probe_v002.py']
    evidence = dict(status='passed', checks=['Strict diagnostic context and model metadata',
                    'Current-policy architecture/weights preserved; checkpoint version explicit',
                    'Signed2degree quaternion cases and all v1 cases',
                    'Visited slow/fast filtering excludes terminal/nonfinite states',
                    'Pre-reset failure retained; inactive reset lanes cannot earn recovery',
                    'Truncation distinct from true failure', 'Diagnostic20s bound/timeouts',
                    'CPU and injected CUDA RNG restored on success and exception'],
                    scope='Actual recovery2 orchestration with CPU toy-vector and CUDA RNG doubles. '
                          'No CUDA context, physics execution, learned stability or hardware claim.',
                    source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                   for path in paths})
    output.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
