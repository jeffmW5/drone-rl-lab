import copy
import json
import pytest
from .research_loop import validation_summary, improves, candidate_config, tick


def result(errors, seeds=(1, 2), crashes=(False, False), successes=(False, False)):
    return {'PPO_residual': {'runs': [dict(seed=s, final_payload_goal_error_mm=e,
             crashed=c, success=p, route_remaining_m=0.)
             for s, e, c, p in zip(seeds, errors, crashes, successes)]}}


def test_promotion_requires_accuracy_and_no_safety_regression():
    old = validation_summary(result([100, 100]), [1, 2])
    assert improves(validation_summary(result([90, 90]), [1, 2]), old)
    assert not improves(validation_summary(result([99, 99]), [1, 2]), old)
    assert not improves(validation_summary(result([1, 1], crashes=(True, False)), [1, 2]), old)
    with pytest.raises(ValueError):
        validation_summary(result([1, 1]), [7, 8])


def test_candidate_cannot_relax_precision_or_gpu_limits():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    base = json.loads((root/'configs/tether_puffer_precision_v001.json').read_text())
    plan = json.loads((root/'configs/tether_research_1070ti_v001.json').read_text())
    trial = copy.deepcopy(plan['trials'][0])
    parent = dict(run='tmp/parent', checkpoint='best')
    cfg = candidate_config(base, plan, trial, parent)
    assert cfg['navigation']['payload_tolerance_m'] == .001
    assert cfg['test_seeds'] == cfg['eval_seeds']
    trial['changes']['episode_seconds'] = 200
    with pytest.raises(ValueError):
        candidate_config(base, plan, trial, parent)


def test_active_job_prevents_launch_and_pausing_prevents_launch(tmp_path, monkeypatch):
    import tether_sim.research_loop as module
    monkeypatch.setattr(module, 'snapshot', lambda root: dict(processes=[
        dict(pid=12, args=['python', '-m', 'tether_sim.puffer_train'])]))
    state = tick(tmp_path, {}, dict(trials=[]))
    assert state['active_pids'] == [12]
    monkeypatch.setattr(module, 'snapshot', lambda root: dict(processes=[]))
    assert tick(tmp_path, {}, dict(trials=[], paused=True))['status'] == 'paused'


def test_training_lease_excludes_duplicates_and_releases(tmp_path):
    from .training_lock import acquire_training_lease
    path = tmp_path/'gpu.lock'
    first = acquire_training_lease(path)
    with pytest.raises(RuntimeError):
        acquire_training_lease(path)
    first.close()
    acquire_training_lease(path).close()


def test_checkpoint_selection_includes_initial_and_numeric_ties(tmp_path):
    from .research_loop import select_checkpoint
    for name in ('initial.zip','initial_normalize.pkl','best.zip','best_normalize.pkl'):
        (tmp_path/name).write_bytes(b'test pair')
    (tmp_path/'evaluation_initial.json').write_text(json.dumps(result([100,100])))
    for step, errors in [(40960,[90,90]), (200704,[95,95]), (81920,[90,90])]:
        data = result(errors)
        data['timesteps'] = step
        (tmp_path/f'eval_{step}.json').write_text(json.dumps(data))
    chosen = select_checkpoint(tmp_path, [1,2])
    assert chosen['checkpoint'] == 'best'
    assert chosen['evidence'].endswith('eval_40960.json')
    (tmp_path/'evaluation_initial.json').write_text(json.dumps(result([1,1])))
    assert select_checkpoint(tmp_path, [1,2])['checkpoint'] == 'initial'


def test_daily_allowance_counts_recovery_and_deduplicates_launches(tmp_path):
    from .research_loop import launch_allowance
    folder = tmp_path/'state/tether_research'
    folder.mkdir(parents=True)
    rows = [dict(run='one',launched_at=99990),dict(run='two',launched_at=99991)]
    (folder/'v001.json').write_text(json.dumps(dict(trials=rows)))
    (folder/'duplicate.json').write_text(json.dumps(dict(trials=rows)))
    (folder/'recovery.json').write_text(json.dumps(dict(run='three',launched_at=99992)))
    assert launch_allowance(tmp_path,100000)['remaining'] == 0
    assert launch_allowance(tmp_path,100000)['next_allowed_at'] == 186390
    assert launch_allowance(tmp_path,186390)['remaining'] == 1
