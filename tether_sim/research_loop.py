"""Resumable, one-job-at-a-time local Puffer research campaign.

Run with system Python: this orchestrator deliberately does not import Torch.
Codex's thread heartbeat reviews the ledger and invokes one tick at a time.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
if __package__:
    from .monitor_snapshot import snapshot, read_json
else:
    from monitor_snapshot import snapshot, read_json


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def validation_summary(result, seeds):
    runs = result.get('PPO_residual', {}).get('runs', [])
    if sorted(r.get('seed') for r in runs) != sorted(seeds):
        raise ValueError('Validation seed mismatch; cannot compare')
    errors = [float(r['final_payload_goal_error_mm']) for r in runs]
    if not all(math.isfinite(x) and x >= 0 for x in errors):
        raise ValueError('Invalid precision measurements')
    return dict(crashes=sum(bool(r['crashed']) for r in runs),
                successes=sum(bool(r['success']) for r in runs),
                mean_final_error_mm=sum(errors)/len(errors),
                completed_routes=sum(r['route_remaining_m'] <= 1e-6 for r in runs),
                errors_mm=errors, seeds=list(seeds))


def rank(summary):
    return (summary['crashes'], -summary['successes'], summary['mean_final_error_mm'])


def improves(candidate, incumbent, fraction=.05):
    """Small validation sets select provisional candidates, not scientific proof."""
    if candidate['crashes'] > incumbent['crashes']:
        return False
    if candidate['completed_routes'] < incumbent['completed_routes']:
        return False
    if candidate['crashes'] < incumbent['crashes']:
        return True
    if candidate['successes'] != incumbent['successes']:
        return candidate['successes'] > incumbent['successes']
    return candidate['mean_final_error_mm'] < incumbent['mean_final_error_mm']*(1-fraction)


def select_checkpoint(folder, seeds):
    choices = []
    initial = folder/'evaluation_initial.json'
    if initial.exists() and (folder/'initial.zip').exists():
        choices.append(('initial', validation_summary(read_json(initial), seeds), str(initial)))
    for path in folder.glob('eval_*.json'):
        data = read_json(path)
        summary = validation_summary(data, seeds)
        # "best" refers to the trainer's best periodic validation rank. For
        # ties, the trainer retains the first checkpoint.
        choices.append(('best', summary, str(path)))
    if not choices:
        raise ValueError('No valid checkpoint evaluation')
    choices.sort(key=lambda row: (rank(row[1]), read_json(Path(row[2])).get('timesteps', 0)))
    chosen = choices[0]
    if not (folder/(chosen[0]+'.zip')).exists() or not (folder/(chosen[0]+'_normalize.pkl')).exists():
        raise ValueError('Checkpoint pair unavailable')
    return dict(run=str(folder), checkpoint=chosen[0], validation=chosen[1], evidence=chosen[2])


def candidate_config(base, plan, trial, incumbent, parent_config=None):
    config = copy.deepcopy(base)
    for key in ('learning_rate', 'gamma', 'n_epochs'):
        if parent_config and key in parent_config:
            config[key] = parent_config[key]
    config.update(name=trial['name'], hypothesis=trial['hypothesis'],
                  resume_run=incumbent['run'], resume_checkpoint=incumbent['checkpoint'],
                  eval_seeds=plan['validation_seeds'], test_seeds=plan['validation_seeds'],
                  eval_every_steps=100000, checkpoint_every_steps=50000)
    config.update(trial['changes'])
    # Evaluation seeds, environment and accuracy contract cannot be altered by
    # a parameter trial. Test-named evaluations here are validation, not holdout.
    for key in ('task', 'episode_seconds', 'navigation', 'physics', 'residual_scale'):
        if config[key] != base[key]:
            raise ValueError('Trial changes the benchmark contract: '+key)
    if config['device'] != 'cuda' or config['n_envs'] > 8 or config['num_workers'] > 4:
        raise ValueError('1070 Ti resource contract violated')
    if config['total_timesteps'] > 400000 or config['budget_seconds'] > 1800:
        raise ValueError('Per-experiment limit exceeded')
    config['notes'] = ('Local adaptive research trial: '+trial['hypothesis']+
                       '. Actual parent is resume_run/resume_checkpoint. Fixed 90-second '
                       'precision benchmark, 3 validation seeds, frozen normalization. '
                       'evaluation_final uses validation seeds; no held-out claim. '
                       'Uncalibrated privileged MuJoCo simulation. CUDA learning and '
                       'advantages; four CPU workers. Provisional best selection only.')
    return config


def report(path, state):
    lines = ['# Local 1070 Ti tether research loop', '',
             'Status: '+state['status'], '',
             'Validation selects provisional candidates; training reward does not select winners.',
             'No hardware accuracy claim. Fresh audit seeds are reserved for a final selected policy.', '',
             '| Trial | State | Crashes | 1 mm successes | Mean final error (mm) | Decision |',
             '|---|---|---:|---:|---:|---|']
    for trial in state['trials']:
        val = trial.get('candidate', {}).get('validation', {})
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            trial['name'], trial['status'], val.get('crashes',''), val.get('successes',''),
            round(val['mean_final_error_mm'], 3) if val else '', trial.get('decision','')))
    if state.get('incumbent'):
        lines += ['', 'Provisional incumbent: '+json.dumps(state['incumbent'], sort_keys=True)]
    if state.get('error'):
        lines += ['', 'Error: '+state['error']]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n')


def tick(root, plan, state, launch=True):
    active = [p for p in snapshot(root/'tmp/tether-runs')['processes']
              if any(m in p['args'] for m in ('tether_sim.train','tether_sim.puffer_train'))]
    if active:
        state['status'] = 'waiting for active training/evaluation'
        state['active_pids'] = [p['pid'] for p in active]
        return state
    if state.get('paused'):
        state['status'] = 'paused'
        return state
    state['active_pids'] = []
    if any(t['status'] == 'failed' for t in state['trials']):
        state['status'] = 'needs review: failed trial; no automatic continuation'
        return state
    if not state.get('incumbent'):
        source = root/plan['initial_run']
        if read_json(source/'status.json').get('status') != 'complete':
            state['status'] = 'needs review: initial run not complete'
            return state
        state['incumbent'] = select_checkpoint(source, plan['validation_seeds'])
    for trial in state['trials']:
        if trial['status'] == 'running':
            folder = root/trial['run']
            status = read_json(folder/'status.json').get('status')
            if status != 'complete':
                state['status'] = 'needs review: trial stopped without completion'
                state['error'] = trial['name']+': '+str(status)
                return state
            candidate = select_checkpoint(folder, plan['validation_seeds'])
            promoted = improves(candidate['validation'], state['incumbent']['validation'])
            trial.update(status='complete', candidate=candidate,
                         decision='provisional promotion' if promoted else 'retain incumbent')
            if promoted:
                state['incumbent'] = candidate
        elif trial['status'] == 'queued':
            if not launch:
                state['status'] = 'ready to launch '+trial['name']
                return state
            # Config files are created once, before execution. Never overwrite
            # an existing frozen config or run folder on recovery.
            config = candidate_config(read_json(root/plan['base_config']), plan, trial, state['incumbent'],
                                      read_json(Path(state['incumbent']['run'])/'config.json'))
            config_path = root/'configs'/('research_'+trial['name']+'.json')
            folder = root/'tmp/tether-runs'/trial['name']
            if config_path.exists() or folder.exists():
                raise ValueError('Existing artifacts require review: '+trial['name'])
            write_json(config_path, config)
            trial.update(status='running', run=str(folder.relative_to(root)),
                         config=str(config_path.relative_to(root)),
                         config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                         parent=copy.deepcopy(state['incumbent']), launched_at=time.time())
            state['status'] = 'launching '+trial['name']
            # Persist intent before launching; an interrupted launch is reviewed
            # rather than silently rerun or overwriting experiment artifacts.
            write_json(root/plan['state_path'], state)
            log = root/'tmp/tether-research'/('log_'+trial['name']+'.txt')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('ab') as output:
                proc = subprocess.Popen(['bash', 'scripts/run_tether_puffer.sh', '--config',
                                         str(config_path), '--output', str(folder.relative_to(root))], cwd=root,
                                        stdout=output, stderr=output, start_new_session=True)
            trial['pid'] = proc.pid
            return state
    state['status'] = 'batch complete: research review and fresh audit required'
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['tick', 'status', 'pause', 'resume', 'dry-run'])
    parser.add_argument('--plan', default='configs/tether_research_1070ti_v001.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    plan = read_json(root/args.plan)
    state_path = root/plan['state_path']
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with (state_path.parent/'research.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='another research tick owns the lock')))
            return
        state = read_json(state_path) or dict(status='queued', trials=copy.deepcopy(plan['trials']))
        if args.action in ('pause', 'resume'):
            state['paused'] = args.action == 'pause'
            state['status'] = 'paused' if state['paused'] else 'ready'
        elif args.action in ('tick', 'dry-run'):
            try:
                state = tick(root, plan, state, launch=args.action=='tick')
            except (ValueError, OSError) as error:
                state.update(status='needs review', error=str(error))
        if args.action not in ('status', 'dry-run'):
            state['updated_at'] = time.time()
            write_json(state_path, state)
            report(root/plan['report_path'], state)
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
