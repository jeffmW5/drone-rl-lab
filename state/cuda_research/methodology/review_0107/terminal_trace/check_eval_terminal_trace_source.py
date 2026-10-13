"""CPU-only checks for the opt-in pre-reset evaluation terminal recorder."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tether_sim.eval_terminal_trace import EvalTerminalTrace


def inputs(count=4):
    obs = torch.zeros(count, 98)
    obs[:, 95:98] = torch.tensor([.2, -.1, .2])
    positions = torch.arange(count * 42, dtype=torch.float32).reshape(count, 14, 3) / 100
    positions[:, 0] = torch.tensor([2.25, -.35, .78])
    obs[:, :42] = (positions - obs[:, None, 95:98]).flatten(1)
    obs[:, 42:84] = torch.arange(count * 42, dtype=torch.float32).reshape(count, 42) / 200
    obs[:, 84] = 1
    obs[:, 88:91] = torch.tensor([.1, .2, .3])
    obs[:, 91:95] = torch.tensor([.11, .12, .13, .14])
    return (obs, torch.zeros(count, dtype=torch.bool), torch.zeros(count, dtype=torch.bool),
            torch.zeros(count, dtype=torch.int32), torch.full((count,), -.75),
            torch.full((count,), 7, dtype=torch.int32)), positions


def invoke(trace, tensors, absolute_step):
    saved = [tensor.clone() for tensor in tensors]
    rng = torch.get_rng_state().clone()
    obs, done, truncated, failed, reward, steps = tensors
    trace.record(obs, done, truncated, failed, reward,
                 absolute_step=absolute_step, episode_steps=steps)
    assert torch.equal(rng, torch.get_rng_state()), 'Recorder consumed RNG'
    for current, original in zip(tensors, saved):
        torch.testing.assert_close(current, original, rtol=0, atol=0, equal_nan=True)
    assert all(tensor.device.type == 'cpu' for tensor in tensors)
    json.dumps(trace.as_dict(), allow_nan=False)


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected validation rejection')


trace = EvalTerminalTrace(4, 250)
tensors, positions = inputs()
invoke(trace, tensors, 1)
assert trace.as_dict()['terminal_count'] == 0
obs, done, truncated, failed, reward, steps = tensors
done[:] = torch.tensor([True, True, False, True])
truncated[:] = torch.tensor([False, True, False, True])
failed[:] = torch.tensor([3, 0, 3, 1])
steps[:] = torch.tensor([7, 500, 7, 500])
reward[:] = torch.tensor([-1., .45, -.5, -1.])
invoke(trace, tensors, 7)
snapshot = trace.as_dict()
assert snapshot['terminal_count'] == 3
assert [event['lane'] for event in snapshot['terminals']] == [0, 1, 3]
assert [event['reason'] for event in snapshot['terminals']] == [
    'failure', 'truncation', 'failure_with_truncation']
assert snapshot['first_terminal_by_lane'][2] is None
for event in snapshot['terminals']:
    lane = event['lane']
    torch.testing.assert_close(torch.tensor(event['positions']), positions[lane], rtol=0, atol=2e-7)
    torch.testing.assert_close(torch.tensor(event['velocities']), obs[lane, 42:84].reshape(14, 3))
    assert event['orientation'] == obs[lane, 84:91].tolist()
    assert event['motors'] == obs[lane, 91:95].tolist()
    assert event['goal'] == obs[lane, 95:98].tolist()
    assert event['episode_step'] == int(steps[lane])
    assert event['absolute_step'] == 7 and event['absolute_time_s'] == .14
    assert event['done'] and event['truncated'] == bool(truncated[lane])
    assert event['failed'] == int(failed[lane]) and event['reward'] == float(reward[lane])
    assert not event['has_nonfinite'] and event['nonfinite_output_value_count'] == 0
# Simulate the auto-reset after the observer. Sparse post-send snapshots show the
# ground state, while the recorder must retain the prior 2.25 m escape state.
obs[:,:42] = -obs[:, None, 95:98].expand(-1,14,-1).flatten(1)
obs[:,2] = .05 - obs[:,97]
obs[:,42:84] = 0
assert trace.as_dict()['terminals'][0]['positions'][0][0] > 2
torch.testing.assert_close(obs[0,:3] + obs[0,95:98], torch.tensor([0.,0.,.05]), rtol=0, atol=1e-8)
# First events stay fixed while all-event chronology grows; as_dict owns copies.
done[:] = torch.tensor([True, False, True, False])
truncated.zero_(); failed[:] = torch.tensor([1, 0, 0, 0]); steps[:] = 2
invoke(trace, tensors, 9)
assert trace.as_dict()['terminal_count'] == 5
assert trace.as_dict()['first_terminal_by_lane'][0]['absolute_step'] == 7
assert trace.as_dict()['first_terminal_by_lane'][2]['absolute_step'] == 9
external = trace.as_dict(); external['terminals'][0]['positions'][0][0] = -999
assert trace.as_dict()['terminals'][0]['positions'][0][0] > 2
# Malformed floating evidence survives as explicit markers and JSON nulls.
nf, _ = inputs()
nf[1][0] = True; nf[3][0] = 2; nf[0][0,0] = float('nan')
nf[0][0,42] = float('inf'); nf[0][0,84] = -float('inf')
nf[0][0,91] = float('nan'); nf[4][0] = float('nan')
invoke(trace, nf, 11)
event = trace.as_dict()['terminals'][-1]
assert event['has_nonfinite'] and event['nonfinite_observation_indices'] == [0,42,84,91]
assert event['nonfinite_reward'] and event['nonfinite_output_value_count'] == 5
assert event['positions'][0][0] is None and event['velocities'][0][0] is None
assert event['orientation'][0] is None and event['motors'][0] is None and event['reward'] is None
# Full bound: one event per lane per absolute step, no silent overwrites.
bounded = EvalTerminalTrace(2, 3)
bound_tensors, _ = inputs(2); bound_tensors[1].fill_(True)
for step in range(1,4):
    bound_tensors[-1].fill_(step)
    invoke(bounded, bound_tensors, step)
assert bounded.as_dict()['terminal_count'] == bounded.max_events == 6
before = json.dumps(bounded.as_dict(), sort_keys=True, allow_nan=False)
rejected(lambda: invoke(bounded, bound_tensors, 3))
rejected(lambda: invoke(bounded, bound_tensors, 4))
rejected(lambda: invoke(bounded, bound_tensors, 2))
assert before == json.dumps(bounded.as_dict(), sort_keys=True, allow_nan=False)
rejected(lambda: EvalTerminalTrace(0,250))
rejected(lambda: EvalTerminalTrace(4,True))
rejected(lambda: EvalTerminalTrace(4,250, float('nan')))
shape_trace = EvalTerminalTrace(4,250)
rejected(lambda: shape_trace.record(obs[:,:97], done, truncated, failed, reward,
                                  absolute_step=1, episode_steps=steps))
assert shape_trace.as_dict()['terminal_count'] == 0
# Failed row validation must leave an entire batch uncaptured, not half-written.
atomic, _ = inputs(); atomic[1][0:2] = True; atomic[-1][1] = 0
rejected(lambda: invoke(shape_trace, atomic, 1))
assert shape_trace.as_dict()['terminal_count'] == 0
assert not torch.cuda.is_initialized(), 'Test initialized CUDA'

# Compile only the current Windows evaluator function, not its module-level
# argument parsing, server, CUDA setup or trainer. Replace device factories with
# CPU factories and simulate exact callback-before-reset chronology.
import ast
from types import SimpleNamespace
trainer_path = Path(__file__).resolve().with_name('train_tether_cuda.py')
trainer_bytes = trainer_path.read_bytes()
trainer_ast = ast.parse(trainer_bytes.decode('utf-8-sig'))
evaluate_node = next(node for node in trainer_ast.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')


class CpuTorchProxy:
    def __getattr__(self, name):
        value = getattr(torch, name)
        if name in ('zeros','empty','ones','full','tensor','arange','as_tensor','rand','randn'):
            def create(*args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return value(*args, **kwargs)
            return create
        return value


class FakeVector:
    def __init__(self, count, **settings):
        assert count == 8
        self.count = count
        self.diagnostic_zone_version = settings.get('diagnostic_zone_version', 0)
        p = torch.zeros(count,14,3,dtype=torch.float64)
        p[:,:,2] = .8 - torch.arange(14,dtype=torch.float64) * .6/13
        self.sim = SimpleNamespace(position=p, velocity=torch.zeros_like(p),
                                   orientation=torch.zeros(count,7),
                                   motors=torch.full((count,4),.12),
                                   failed=torch.zeros(count,dtype=torch.int32))
        self.sim.orientation[:,0] = 1
        self.start = p.clone()
        self.goal = torch.tensor([.2,-.1,.2]).repeat(count,1)
        self.steps = torch.zeros(count,dtype=torch.int32)
        self.done = torch.zeros(count,dtype=torch.bool)
        self.truncated = self.done.clone()
        self.reward = torch.zeros(count)
        self.zone_counts = dict(band_steps=0,escapes=0)
        self.terminal_observer = None
        self.absolute_step = 0

    def async_reset(self, seed):
        torch.manual_seed(seed)
        # Seeded RNG consumption exposes any accidental recorder draw.
        torch.rand(self.count)

    def obs(self):
        return torch.cat(((self.sim.position-self.goal[:,None]).flatten(1),
                          self.sim.velocity.flatten(1),self.sim.orientation,
                          self.sim.motors,self.goal),dim=1).float()

    def send(self, action):
        assert action.shape == (self.count,4)
        self.absolute_step += 1
        self.steps += 1
        self.done.zero_(); self.truncated.zero_(); self.sim.failed.zero_()
        self.sim.position[:,0,0] += .01
        self.sim.velocity[:,0,0] = .1
        self.reward.fill_(.4)
        scenario = {3:(0,3,False),4:(1,0,True),6:(2,1,True),26:(0,3,False)}
        if self.absolute_step not in scenario:
            return
        lane, failure, timeout = scenario[self.absolute_step]
        self.sim.position[lane,0] = torch.tensor([2.25,-.35,.78],dtype=torch.float64)
        self.sim.velocity[lane,0] = torch.tensor([.9,.1,-.3],dtype=torch.float64)
        self.done[lane] = True; self.truncated[lane] = timeout
        self.sim.failed[lane] = failure; self.reward[lane] = -1
        if timeout:
            self.steps[lane] = 500
        if failure == 3:
            self.zone_counts['escapes'] += 1
        if self.terminal_observer is not None:
            self.terminal_observer(self.obs(), self.done, self.truncated,
                                   self.sim.failed, self.reward)
        # Match the production order: observer first, then auto-reset; done and
        # truncated flags persist for the evaluator's existing metric logic.
        self.sim.position[lane] = self.start[lane]
        self.sim.position[lane,0,2] = .05
        self.sim.velocity[lane] = 0
        self.sim.orientation[lane] = 0; self.sim.orientation[lane,0] = 1
        self.sim.motors[lane] = 0; self.sim.failed[lane] = 0; self.steps[lane] = 0


class FakePolicy:
    def __call__(self, observations):
        return SimpleNamespace(mean=torch.zeros(len(observations),4)), torch.zeros(len(observations),1)


namespace = dict(torch=CpuTorchProxy(), CudaVector=FakeVector,
                 args=SimpleNamespace(direct=True), publish=lambda *args: None)
exec(compile(ast.Module(body=[evaluate_node],type_ignores=[]),str(trainer_path),'exec'),namespace)
base_plan = dict(evaluation_steps=28,diagnostic_zone_version=1,training_reset_version=3,
                 reward_version=15,collective_observation_version=2,physics_version=5)
namespace['plan'] = dict(base_plan,evaluation_trace_version=0)
baseline = namespace['evaluate'](FakePolicy(),51011)
baseline_rng = torch.get_rng_state().clone()
assert 'evaluation_terminal_trace' not in baseline
namespace['plan'] = dict(base_plan,evaluation_trace_version=1)
instrumented = namespace['evaluate'](FakePolicy(),51011)
assert torch.equal(baseline_rng,torch.get_rng_state()), 'Evaluator hook changed RNG'
actual_trace = instrumented.pop('evaluation_terminal_trace')
assert baseline.keys() == instrumented.keys()
assert json.dumps(baseline,sort_keys=True,allow_nan=False) == json.dumps(instrumented,sort_keys=True,allow_nan=False), 'Existing result field changed'
assert [event['absolute_step'] for event in actual_trace['terminals']] == [3,4,6,26]
assert [event['lane'] for event in actual_trace['terminals']] == [0,1,2,0]
assert [event['reason'] for event in actual_trace['terminals']] == [
    'failure','truncation','failure_with_truncation','failure']
assert actual_trace['first_terminal_by_lane'][0]['absolute_step'] == 3
assert actual_trace['terminals'][-1]['episode_step'] == 23
assert abs(actual_trace['terminals'][-1]['positions'][0][0]-2.25) < 2e-7
assert baseline['trajectory'][1]['step'] == 25 and baseline['trajectory'][1]['terminal']
assert baseline['trajectory'][1]['positions'][0][0] == 0
assert baseline['trajectory'][1]['positions'][0][2] == .05
namespace['plan'] = dict(base_plan,evaluation_trace_version=2)
rejected(lambda: namespace['evaluate'](FakePolicy(),51011))
assert not torch.cuda.is_initialized()
integration = dict(trainer_path=str(trainer_path),
                   trainer_sha256=hashlib.sha256(trainer_bytes).hexdigest(),
                   evaluation_function_sha256=hashlib.sha256(ast.dump(evaluate_node).encode()).hexdigest(),
                   all_existing_result_fields_equal=True,rng_equal=True,
                   unchanged_fields=sorted(baseline),
                   baseline_post_reset_trajectory=baseline['trajectory'],
                   captured_pre_reset_trace=actual_trace,
                   scope='Actual evaluator AST with CPU factories and fake pre-reset vector; no dynamics or GPU equivalence proof.')

# Exercise the actual queued-reference branch: correct source, mismatching old
# score, and tampered immutable source. This does not execute the trainer.
import copy
import tempfile
reference_node = next(node for node in ast.walk(trainer_ast)
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "plan.get('evaluation_trace_reference')")
reference_code = compile(ast.Module(body=[reference_node],type_ignores=[]),str(trainer_path),'exec')
with tempfile.TemporaryDirectory(prefix='eval-terminal-reference-') as temporary:
    folder = Path(temporary)
    expected = dict(crashes=baseline['crashes'],runs=[copy.deepcopy(baseline)])
    initial = copy.deepcopy(expected)
    initial['runs'][0]['evaluation_terminal_trace'] = actual_trace
    reference = folder/'reference.json'
    reference.write_text(json.dumps(dict(final=expected),allow_nan=False))
    reference_plan = dict(evaluation_trace_version=1,evaluation_trace_reference=str(reference),
                          evaluation_trace_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    reference_namespace = dict(plan=reference_plan,Path=Path,json=json,initial=initial,out=folder)
    before = copy.deepcopy(initial)
    exec(reference_code,reference_namespace)
    comparison = json.loads((folder/'eval_terminal_trace_comparison.json').read_text())
    assert comparison['legacy_fields_exactly_equal'] and initial == before
    assert comparison['terminal_count'] == 4 and comparison['recorded_physical_failures'] == 3
    assert comparison['expected_crashes'] == comparison['observed_crashes'] == baseline['crashes']
    initial['runs'][0]['zone_hover_error_mm'] += 1
    exec(reference_code,reference_namespace)
    assert not json.loads((folder/'eval_terminal_trace_comparison.json').read_text())['legacy_fields_exactly_equal']
    reference.write_text('tampered reference')
    try:
        exec(reference_code,reference_namespace)
    except RuntimeError as error:
        assert str(error) == 'Trace reference source changed'
    else:
        raise AssertionError('Tampered immutable reference was accepted')
    reference_plan['evaluation_trace_version'] = 0
    rejected(lambda: exec(reference_code,reference_namespace))
integration['reference_contract'] = dict(correct_source_matches=True,metric_mismatch_visible=True,
                                         tampered_source_rejected=True,baseline_without_trace_rejected=True,
                                         input_not_mutated=True)

out = Path('tmp/deep_audit_experiment/eval_terminal_trace_cpu_test.json')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    for version in range(2,1000):
        candidate = out.with_name(f'eval_terminal_trace_cpu_test_v{version:03d}.json')
        if not candidate.exists():
            out = candidate
            break
    else:
        raise FileExistsError('Terminal trace evidence suffix range exhausted')
result = dict(scope='CPU-only fixture tests; no dynamics, training, CUDA or real evaluation equality proof.',
              cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
              source_sha256=hashlib.sha256(Path('tether_sim/eval_terminal_trace.py').read_bytes()).hexdigest(),
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              checks=['nonterminal exclusion', 'mixed failure and timeout flags retained',
                      '98-observation reconstruction', 'pre-reset evidence retained after mutation',
                      'first and all terminal chronology', 'bounded unique capture',
                      'strict JSON with nonfinite nulls and explicit markers',
                      'no input mutation or RNG draws', 'atomic malformed-batch rejection'],
              passed=True, sample_trace=snapshot, nonfinite_event=event, evaluator_integration=integration,
              missing_evidence='Actual GPU observer hook and unchanged evaluation metrics require separate queued diagnostic.')
out.write_text(json.dumps(result, indent=2, allow_nan=False))
print('Evaluation terminal trace CPU tests passed; evidence saved to ' + str(out))


