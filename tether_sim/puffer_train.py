"""Bounded PufferLib 3 CUDA training, with SB3-compatible policy exports."""
import argparse
import atexit
import hashlib
import json
import os
import pickle
import signal
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from .puffer_adapter import TransferredPolicy, make_puffer_env
from .train import evaluate


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tether_puffer_v001.json')
    parser.add_argument('--output', default='tmp/tether-runs/tether_puffer_v001')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config.update(total_timesteps=4096, n_envs=2, num_workers=2,
                      rollout_batch_size=1024, minibatch_size=256, bptt_horizon=64,
                      eval_seeds=[51001], test_seeds=[61001], budget_seconds=120)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', config)
    write_json(out/'status.json', dict(status='initializing', timesteps=0, device='cuda', updated_at=time.time()))
    import pufferlib
    import pufferlib.vector
    from pufferlib.pufferl import PuffeRL, ADVANTAGE_CUDA
    if config.get('require_cuda_advantage', False) and not ADVANTAGE_CUDA:
        raise RuntimeError('This frozen experiment requires the compiled CUDA advantage kernel; use run_tether_puffer.sh')
    # Upstream installs an immediate SIGINT exit handler. Replace it so this
    # runner can export its latest controller and stop workers cleanly.
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    source = Path(config['resume_run'])
    checkpoint = config['resume_checkpoint']
    model_file = source/(checkpoint+'.zip')
    norm_file = source/(checkpoint+'_normalize.pkl')
    # Fork CPU-only workers before any CUDA context is initialized. This shares
    # imported Python/Torch memory instead of duplicating it in every worker.
    # The workers never execute CUDA operations.
    import multiprocessing
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU workers must be created before CUDA initialization')
    multiprocessing.set_start_method('fork', force=True)
    vec = pufferlib.vector.make(make_puffer_env, env_kwargs=dict(config=config, normalizer_path=str(norm_file)),
                                backend='Multiprocessing', num_envs=config['n_envs'],
                                num_workers=config['num_workers'], batch_size=config['n_envs'],
                                zero_copy=True, seed=config['seed'])
    def terminate_workers():
        for process in vec.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
    atexit.register(terminate_workers)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    model = PPO.load(model_file, device='cuda')
    with norm_file.open('rb') as file:
        normalizer = pickle.load(file)
    normalizer.training = False
    provenance = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                      dirty=bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
                      pid=os.getpid(), trainer='PufferLib 3 PuffeRL', pufferlib=str(pufferlib.__version__),
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(0), advantage_cuda=ADVANTAGE_CUDA,
                      source_timesteps=model.num_timesteps, source_run=str(source),
                      policy_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                      normalizer_sha256=hashlib.sha256(norm_file.read_bytes()).hexdigest(),
                      limitations=config['notes'])
    write_json(out/'provenance.json', provenance)
    print('PROVENANCE '+json.dumps(provenance), flush=True)
    write_json(out/'status.json', dict(status='evaluating initial transfer', timesteps=0, device='cuda', updated_at=time.time()))
    initial = evaluate(model, normalizer, config)
    write_json(out/'evaluation_initial.json', initial)
    policy = TransferredPolicy(model.policy, normalizer).to('cuda')
    train_config = dict(env='tether_navigation_v2', seed=config['seed'], torch_deterministic=True, cpu_offload=False,
                        device='cuda', optimizer='adam', anneal_lr=False, precision='float32', amp=False,
                        total_timesteps=config['total_timesteps'], learning_rate=config['learning_rate'],
                        gamma=config['gamma'], gae_lambda=.95, update_epochs=config['n_epochs'],
                        clip_coef=.2, vf_coef=.5, vf_clip_coef=.2, max_grad_norm=.5, ent_coef=0.,
                        adam_beta1=.9, adam_beta2=.999, adam_eps=1e-5, min_lr_ratio=1.,
                        data_dir=str(out/'puffer_checkpoints'), checkpoint_interval=10,
                        batch_size=config['rollout_batch_size'], minibatch_size=config['minibatch_size'],
                        max_minibatch_size=config['minibatch_size'], bptt_horizon=config['bptt_horizon'],
                        compile=False, compile_mode='default', compile_fullgraph=False, use_rnn=False,
                        vtrace_rho_clip=1., vtrace_c_clip=1., prio_alpha=0., prio_beta0=0.)
    trainer = None
    started = time.monotonic()
    best_rank = None
    next_eval = config['eval_every_steps']
    next_save = config['checkpoint_every_steps']
    failed = False
    def export(name):
        # Same parameter order: normalization statistics are buffers, not parameters.
        model.policy.optimizer = trainer.optimizer
        model.num_timesteps = trainer.global_step
        model.save(out/name)
        # Frozen statistics are exported byte-for-byte. A detached VecNormalize
        # lacks the live wrapper attributes needed by its save() method.
        shutil.copyfile(norm_file, out/(name+'_normalize.pkl'))
    try:
        trainer = PuffeRL(train_config, vec, policy)
        trainer.optimizer.load_state_dict(model.policy.optimizer.state_dict())
        for group in trainer.optimizer.param_groups:
            group['lr'] = config['learning_rate']
        export('initial')
        while trainer.global_step < config['total_timesteps'] and not stopping[0]:
            trainer.evaluate()  # PufferLib calls rollout collection "evaluate".
            trainer.train()
            elapsed = time.monotonic()-started
            state = dict(status='training', timesteps=trainer.global_step, elapsed_seconds=elapsed,
                         steps_per_second=trainer.global_step/max(elapsed,1), device='cuda',
                         gpu=torch.cuda.get_device_name(0), updated_at=time.time(), losses=trainer.losses)
            write_json(out/'status.json', state)
            if trainer.global_step >= next_save:
                export('checkpoint_'+str(trainer.global_step))
                next_save += config['checkpoint_every_steps']
            if trainer.global_step >= next_eval:
                export('latest')
                result = evaluate(model, normalizer, config)
                result.update(timesteps=trainer.global_step, elapsed_seconds=time.monotonic()-started)
                write_json(out/('eval_'+str(trainer.global_step)+'.json'), result)
                ppo = result['PPO_residual']
                rank = (ppo['crashes'], -ppo['successes'], ppo['mean_payload_rmse_m'])
                if config.get('task') == 'random_navigation_precision_v1':
                    rank = (ppo['crashes'], -ppo['successes'],
                            float(np.mean([r['final_payload_goal_error_mm'] for r in ppo['runs']])))
                if best_rank is None or rank < best_rank:
                    export('best'); best_rank = rank
                next_eval += config['eval_every_steps']
            if elapsed >= config['budget_seconds']:
                break
        export('final')
        if not (out/'best.zip').exists():
            export('best')
        result = evaluate(model, normalizer, config, seeds=config['test_seeds'])
        write_json(out/'evaluation_final.json', result)
        write_json(out/'status.json', dict(status='interrupted' if stopping[0] else 'complete',
                   timesteps=trainer.global_step, elapsed_seconds=time.monotonic()-started,
                   device='cuda', evaluation=result, updated_at=time.time()))
    except BaseException as error:
        failed = True
        if trainer is not None:
            export('interrupted')
        write_json(out/'status.json', dict(status='failed', error=repr(error),
                   timesteps=trainer.global_step if trainer else 0, device='cuda', updated_at=time.time()))
        raise
    finally:
        if failed:
            # Upstream close waits for pipe replies; failed startup workers may
            # never reply. Stop only this vector's own children in that case.
            terminate_workers()
            if trainer is not None:
                trainer.utilization.stop()
        elif trainer is not None:
            # Upstream vector close may wait indefinitely for a worker pipe.
            # Checkpoints are saved before stopping our CPU-only children.
            trainer.save_checkpoint()
            trainer.utilization.stop()
            terminate_workers()
        else:
            terminate_workers()


if __name__ == '__main__':
    main()
