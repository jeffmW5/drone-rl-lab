"""Versioned, read-only recovery diagnostic; caller owns the serial GPU lease.

No helper controller, learning, action sampling or benchmark modification. A
current-policy entry retains its configured architecture and weights. Explicit
checkpoints require architecture metadata. Factories support CPU invariance
checks without creating a CUDA context.
"""
import copy
import hashlib
import math
from pathlib import Path

import torch

from .cuda_task import CudaVector


def _validate_plan(plan):
    required = dict(physics_version=5, control_version=2, reward_version=15,
                    diagnostic_zone_version=1)
    for key, value in required.items():
        if plan.get(key) != value:
            raise ValueError(f'Recovery2 requires {key}={value}')
    models = plan.get('recovery_models', [{'current_policy': True}])
    if not isinstance(models, list) or not models:
        raise ValueError('recovery_models must be a nonempty list')
    for item in models:
        if not isinstance(item, dict):
            raise ValueError('Recovery model entry must be a dictionary')
        if item.get('current_policy') is True:
            if 'checkpoint' in item:
                raise ValueError('Current policy and checkpoint are exclusive')
        elif 'checkpoint' not in item or item.get('current_policy', False):
            raise ValueError('Recovery model requires current_policy=true or checkpoint')
        elif item.get('collective_observation_version') not in (1, 2):
            raise ValueError('Checkpoint recovery requires explicit observation version1/2')
    return models


def _visited_cases(plan):
    path_value = plan.get('recovery_rollout')
    evidence = dict(available=False, reason='recovery_rollout not supplied',
                    selected_indices={}, eligible_counts={})
    if path_value is None:
        return [], evidence
    path = Path(path_value)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    observations = saved['observations'].reshape(-1, 98)
    terminals = saved['terminals'].reshape(-1)
    if terminals.numel() != observations.shape[0]:
        raise ValueError('Visited terminal and observation lengths differ')
    height = observations[:, 2] + observations[:, 97]
    speed = observations[:, 42:45].norm(dim=1)
    eligible = ((height - .8).abs() < .1) & (terminals == 0)
    eligible &= torch.isfinite(observations).all(dim=1)
    evidence.update(available=True, reason=None, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    scope='Actual captured buffer states; reconstructed FP32 observations, '
                          'not a full-FP64 solver snapshot or a credit attribution.')
    cases = []
    for name, mask in (('visited_slow', eligible & (speed < .2)),
                       ('visited_fast', eligible & (speed > 1))):
        indices = mask.nonzero().flatten()
        evidence['eligible_counts'][name] = int(indices.numel())
        if indices.numel():
            index = int(indices[len(indices) // 2])
            evidence['selected_indices'][name] = index
            cases.append(dict(name=name, kind='visited', observation=observations[index],
                              source_index=index))
    return cases, evidence


def _case_specs(visited):
    cases = [dict(name='ground', kind='ground'),
             dict(name='ideal_hover', kind='ideal')]
    for axis, coordinate in (('roll', 1), ('pitch', 2)):
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            cases.append(dict(name=f'{axis}_{label}10', kind='angle',
                              coordinate=coordinate, angle_deg=10 * sign))
    cases.extend([dict(name='vx_plus', kind='velocity', value=.5),
                  dict(name='vx_minus', kind='velocity', value=-.5),
                  dict(name='wx_plus', kind='rate', value=1.),
                  dict(name='wx_minus', kind='rate', value=-1.)])
    cases.extend(visited)
    for axis, coordinate in (('roll', 1), ('pitch', 2)):
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            cases.append(dict(name=f'{axis}_{label}2', kind='angle',
                              coordinate=coordinate, angle_deg=2 * sign))
    return cases


def _initialize_cases(vec, cases):
    sim = vec.sim
    device = sim.position.device
    sim.position[:, :, :2] = 0
    sim.position[:, :, 2] = .8 - torch.arange(14, device=device).double() * .6 / 13
    sim.velocity.zero_()
    sim.orientation.zero_()
    sim.orientation[:, 0] = 1
    sim.motors[:] = sim.mass.sum(1, keepdim=True) * 9.81 / 4
    for row, case in enumerate(cases):
        if case['kind'] == 'ground':
            sim.position[row] = vec.start[row]
            sim.motors[row] = 0
        elif case['kind'] == 'angle':
            half = math.radians(case['angle_deg']) / 2
            sim.orientation[row, 0] = math.cos(half)
            sim.orientation[row, case['coordinate']] = math.sin(half)
        elif case['kind'] == 'velocity':
            sim.velocity[row, :, 0] = case['value']
        elif case['kind'] == 'rate':
            sim.orientation[row, 4] = case['value']
        elif case['kind'] == 'visited':
            obs = case['observation'].to(device)
            vec.goal[row] = obs[95:98]
            sim.position[row] = obs[:42].reshape(14, 3).double() + obs[95:98].double()
            sim.velocity[row] = obs[42:84].reshape(14, 3).double()
            sim.orientation[row] = obs[84:91]
            sim.motors[row] = obs[91:95]


def _clone_model(policy, item):
    model = copy.deepcopy(policy)
    if item.get('current_policy') is True:
        version = getattr(model, 'collective_observation_version', 1)
        if version not in (1, 2):
            raise ValueError('Unsupported current policy observation version')
        if ('collective_observation_version' in item and
                item['collective_observation_version'] != version):
            raise ValueError('Current policy observation version must remain unchanged')
        source = dict(current_policy=True, collective_observation_version=version,
                      scope='Deep copy of current configured policy; weights unchanged')
    else:
        path = Path(item['checkpoint'])
        version = item['collective_observation_version']
        model.configure_collective_observation(version)
        weights = torch.load(path, map_location='cpu', weights_only=True)
        weights.setdefault('obs_scale', model.obs_scale)
        model.load_state_dict(weights)
        source = dict(current_policy=False, checkpoint=str(path),
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      collective_observation_version=version)
    model.eval()
    return model, source


def probe(policy, plan, *, vector_factory=None, cuda_rng=None):
    """Query fixed policies for <=20s per case; first true failure ends its case.

    Normal callers use probe(policy, plan). Optional factories are intended only
    for CPU contract tests; production defaults create the established CUDA
    vector and preserve the calling thread's current CUDA RNG state.
    """
    models = _validate_plan(plan)
    vector_factory = CudaVector if vector_factory is None else vector_factory
    cuda_rng = torch.cuda if cuda_rng is None else cuda_rng
    cpu_state = torch.get_rng_state()
    gpu_state = cuda_rng.get_rng_state()
    try:
        visited, visited_evidence = _visited_cases(plan)
        cases = _case_specs(visited)
        result = dict(version=2, control_dt_s=.02, maximum_simulated_s=20,
                      scope='Recovery from artificial and captured visited states. '
                            'No helper controller, learning, action sampling or reset '
                            'counted as recovery; standard benchmark unchanged.',
                      visited_evidence=visited_evidence, models=[])
        for item in models:
            model, source = _clone_model(policy, item)
            vec = vector_factory(len(cases), direct=True, reward_version=15,
                                 phase='takeoff', control_version=2, physics_version=5,
                                 training_reset_version=3, diagnostic_zone_version=1)
            vec.async_reset(51011)
            _initialize_cases(vec, cases)
            sim = vec.sim
            device = sim.position.device
            active = torch.ones(len(cases), device=device, dtype=torch.bool)
            hold = torch.zeros(len(cases), device=device)
            peak = hold.clone()
            tookoff = sim.position[:, 0, 2] > .3
            landing = [None] * len(cases)
            failures = [None] * len(cases)
            terminals = [None] * len(cases)
            frames = []
            initial_observations = vec.obs().detach().cpu().tolist()

            def terminal(obs, done, truncated, failed, reward):
                for lane in (done & active).nonzero().flatten().cpu().tolist():
                    event = dict(time_s=(step + 1) * .02,
                                 failed=int(failed[lane]),
                                 truncated=bool(truncated[lane]),
                                 reward=float(reward[lane]),
                                 observation=obs[lane].detach().cpu().tolist(),
                                 scope='Pre-reset policy observation')
                    terminals[lane] = event
                    if event['failed'] != 0:
                        failures[lane] = event
                active[done] = False

            vec.terminal_observer = terminal
            for step in range(1000):
                vec.steps.zero_()  # Disable500-step timeout only in this diagnostic.
                with torch.no_grad():
                    dist, _ = model(vec.obs())
                    mean = dist.mean
                vec.send(mean.detach().cpu().numpy())
                position = sim.position[:, 0]
                velocity = sim.velocity[:, 0]
                extent = position[:, :2].abs().amax(1)
                speed = velocity.norm(dim=1)
                tookoff |= (position[:, 2] > .3) & active
                landed = (position[:, 2] <= .051) & tookoff & active
                for lane in landed.nonzero().flatten().cpu().tolist():
                    if landing[lane] is None:
                        landing[lane] = (step + 1) * .02
                stable = (active & (extent <= 1) &
                          ((position[:, 2] - .8).abs() < .1) & (speed < .2))
                hold = torch.where(stable, hold + .02, 0)
                peak = torch.maximum(peak, hold)
                if step % 10 == 0 or not bool(active.any()) or step == 999:
                    frames.append(dict(time_s=(step + 1) * .02,
                                       active=active.cpu().tolist(),
                                       carrier=position.cpu().tolist(),
                                       velocity=velocity.cpu().tolist(),
                                       orientation=sim.orientation.cpu().tolist(),
                                       mean_action=mean.cpu().tolist(),
                                       scope='Post-step state; inactive lanes may have '
                                             'auto-reset and never count as recovery'))
                if not bool(active.any()):
                    break
            source.update(elapsed_simulated_s=(step + 1) * .02,
                          initial_observations=initial_observations,
                          cases=[dict(name=case['name'], kind=case['kind'],
                                      peak_hold_s=float(peak[i]),
                                      hover_success=bool(peak[i] >= 1),
                                      first_landing_s=landing[i], first_failure=failures[i],
                                      first_terminal=terminals[i],
                                      survived_20s=bool(active[i]) and step == 999)
                                 for i, case in enumerate(cases)], frames=frames)
            result['models'].append(source)
        return result
    finally:
        torch.set_rng_state(cpu_state)
        cuda_rng.set_rng_state(gpu_state)
