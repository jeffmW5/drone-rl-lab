"""
Drone RL Lab — Racing Backend (lsy_drone_racing)
==================================================
Config-driven training for drone gate racing using CleanRL PPO.
Wraps lsy_drone_racing's training infrastructure.

Usage:
    python train.py configs/exp_010_racing_baseline.yaml   # via dispatcher
    python train_racing.py configs/exp_010_racing_baseline.yaml  # direct

Requirements:
    pip install -e /path/to/lsy_drone_racing[sim,rl]
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _to_np(x):
    """Convert JAX/torch/numpy arrays to numpy (handles GPU tensors)."""
    if hasattr(x, '__jax_array__') or type(x).__module__.startswith('jax'):
        return np.asarray(x)
    if hasattr(x, 'cpu'):  # PyTorch tensor
        return x.detach().cpu().numpy()
    return np.array(x)


def _action_for_env(action_tensor, use_jax_gpu=False):
    """Convert PyTorch action tensor to format the env expects.

    When JAX runs on GPU, the env expects JAX GPU arrays, not numpy.
    """
    if use_jax_gpu:
        import jax.numpy as jnp
        return jnp.array(action_tensor.detach().cpu().numpy())
    return action_tensor.cpu().numpy()


# PyYAML
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

# Lazy-load lsy_drone_racing to give a clear error message
try:
    import torch
    from lsy_drone_racing.control.train_rl import Args, Agent, make_envs, train_ppo
except ImportError as e:
    print(f"ERROR: lsy_drone_racing not installed. {e}")
    print("Install with: cd /path/to/lsy_drone_racing && pip install -e '.[sim,rl]'")
    sys.exit(1)


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load experiment config from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required = ["name", "hypothesis", "racing"]
    for field in required:
        if field not in config:
            raise ValueError(f"Config missing required field: {field}")

    config.setdefault("budget_seconds", 600)
    config.setdefault("backend", "racing")
    return config


def build_args(racing_cfg: dict) -> Args:
    """Build an Args dataclass from the YAML racing: section."""
    args = Args()

    # Map YAML keys to Args fields
    field_names = {f.name for f in Args.__dataclass_fields__.values()}
    for key, value in racing_cfg.items():
        if key in field_names:
            setattr(args, key, value)

    # CPU-friendly defaults for VM
    if not racing_cfg.get("cuda", False):
        args.cuda = False
        args.jax_device = "cpu"

    # Compute derived fields
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    return args


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_racing(agent: Agent, envs_fn, device, n_episodes: int = 10, jax_on_gpu: bool = False) -> dict:
    """Run evaluation episodes and return stats."""
    eval_envs = envs_fn()
    rewards_all = []
    lengths_all = []

    obs, info = eval_envs.reset()
    if isinstance(obs, np.ndarray):
        obs_tensor = torch.Tensor(obs).to(device)
    else:
        obs_tensor = torch.Tensor(_to_np(obs)).to(device)

    episode_rewards = np.zeros(eval_envs.num_envs)
    episode_lengths = np.zeros(eval_envs.num_envs)
    completed = 0

    while completed < n_episodes:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True)

        obs, reward, terminated, truncated, info = eval_envs.step(_action_for_env(action, jax_on_gpu))
        obs_tensor = torch.Tensor(_to_np(obs)).to(device)

        reward_np = _to_np(reward)
        episode_rewards += reward_np
        episode_lengths += 1

        dones = np.logical_or(_to_np(terminated), _to_np(truncated))
        for i in np.where(dones)[0]:
            if completed < n_episodes:
                rewards_all.append(float(episode_rewards[i]))
                lengths_all.append(int(episode_lengths[i]))
                completed += 1
            episode_rewards[i] = 0
            episode_lengths[i] = 0

    eval_envs.close()

    return {
        "mean_reward": float(np.mean(rewards_all)),
        "std_reward": float(np.std(rewards_all)),
        "mean_episode_length": float(np.mean(lengths_all)),
        "n_episodes": len(rewards_all),
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run(config_path: str):
    config = load_config(config_path)

    name = config["name"]
    hypothesis = config["hypothesis"]
    budget = config["budget_seconds"]
    racing_cfg = config["racing"]

    # ── Paths ─────────────────────────────────────────────────────────────────
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results", name)
    os.makedirs(results_dir, exist_ok=True)

    # Copy config for reproducibility
    import shutil
    shutil.copy2(config_path, os.path.join(results_dir, "config.yaml"))

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {name}")
    print(f"  BACKEND:    racing (lsy_drone_racing)")
    print(f"  HYPOTHESIS: {hypothesis}")
    print(f"  BUDGET:     {budget}s")
    print(f"  CONFIG:     {config_path}")
    print(f"  OUTPUT:     {results_dir}")
    print(f"{'='*65}\n")

    # ── Build Args from config ────────────────────────────────────────────────
    args = build_args(racing_cfg)
    level = racing_cfg.get("level", "level0")

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    jax_device = args.jax_device if args.cuda else "cpu"
    jax_on_gpu = jax_device == "gpu"

    n_obs = racing_cfg.get("n_obs", 2)
    print(f"[INFO] PyTorch device: {device}")
    print(f"[INFO] JAX device:     {jax_device}")
    print(f"[INFO] Num envs:       {args.num_envs}")
    print(f"[INFO] Total steps:    {args.total_timesteps:,}")
    print(f"[INFO] Level:          {level}")
    print(f"[INFO] n_obs:          {n_obs} (observation stacking)")

    # ── Create environments ───────────────────────────────────────────────────
    env_type = racing_cfg.get("env_type", "trajectory")  # "trajectory" or "race"
    reward_coefs = {
        "n_obs": racing_cfg.get("n_obs", 2),  # observation stacking (default matches Args)
        "rpy_coef": racing_cfg.get("rpy_coef", 0.06),
        "d_act_th_coef": racing_cfg.get("d_act_th_coef", 0.4),
        "d_act_xy_coef": racing_cfg.get("d_act_xy_coef", 1.0),
        "act_coef": racing_cfg.get("act_coef", 0.02),
        "gate_aware": racing_cfg.get("gate_aware", False),
    }

    if env_type == "race":
        # RaceCoreEnv pipeline — train directly on gate-racing env
        from lsy_drone_racing.control.train_race import make_race_envs
        race_coefs = {
            **reward_coefs,
            "gate_bonus": racing_cfg.get("gate_bonus", 5.0),
            "proximity_coef": racing_cfg.get("proximity_coef", 2.0),
            "speed_coef": racing_cfg.get("speed_coef", 0.1),
            "max_episode_steps": racing_cfg.get("max_episode_steps", 1500),
        }
        race_config = racing_cfg.get("race_config", f"{level}_attitude.toml")
        envs = make_race_envs(
            config=race_config,
            num_envs=args.num_envs,
            jax_device=jax_device,
            torch_device=device,
            coefs=race_coefs,
        )
        print(f"[INFO] env_type:       race (VecDroneRaceEnv)")
        print(f"[INFO] race_config:    {race_config}")
    else:
        # Original trajectory-following pipeline
        envs = make_envs(
            config=f"{level}.toml",
            num_envs=args.num_envs,
            jax_device=jax_device,
            torch_device=device,
            coefs=reward_coefs,
        )
        print(f"[INFO] env_type:       trajectory (RandTrajEnv)")

    print(f"[INFO] Obs space:      {envs.single_observation_space}")
    print(f"[INFO] Action space:   {envs.single_action_space}\n")

    # ── Create agent ──────────────────────────────────────────────────────────
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    agent = Agent(obs_shape, act_shape).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ── Training storage ──────────────────────────────────────────────────────
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + act_shape).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # For evaluations.npz compatibility
    eval_timesteps = []
    eval_results = []

    # ── Early stopping config ──────────────────────────────────────────────────
    es_cfg = racing_cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", False)
    es_window = es_cfg.get("window", 50)
    es_patience = es_cfg.get("patience", 200)
    es_min_delta = es_cfg.get("min_delta", 0.05)
    es_reward_history = []
    es_best_avg = float('-inf')
    es_best_iteration = 0
    early_stopped = False
    plateau_iteration = None

    if es_enabled:
        print(f"[INFO] Early stopping: window={es_window}, patience={es_patience}, min_delta={es_min_delta}")

    # ── Training loop ─────────────────────────────────────────────────────────
    wall_start = time.time()
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(_to_np(next_obs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("[Training] Starting CleanRL PPO loop...")

    for iteration in range(1, args.num_iterations + 1):
        # Check time budget
        elapsed = time.time() - wall_start
        if elapsed > budget:
            print(f"\n[TimeBudget] {budget}s exceeded — stopping at iteration {iteration}.")
            break

        # Anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # ── Rollout phase ─────────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(_action_for_env(action, jax_on_gpu))
            reward = torch.tensor(_to_np(reward), dtype=torch.float32).to(device)
            rewards_buf[step] = reward
            next_obs = torch.Tensor(_to_np(next_obs)).to(device)
            next_done = torch.Tensor(
                np.logical_or(_to_np(terminated), _to_np(truncated)).astype(float)
            ).to(device)

        # ── Compute advantages (GAE) ──────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # ── Flatten batches ───────────────────────────────────────────────
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + act_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ── PPO update ────────────────────────────────────────────────────
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ── Compute iteration reward (every iteration for early stopping) ──
        iter_reward = rewards_buf.sum(0).mean().item()

        # ── Early stopping check ─────────────────────────────────────────
        if es_enabled:
            es_reward_history.append(iter_reward)
            if len(es_reward_history) >= es_window:
                current_avg = np.mean(es_reward_history[-es_window:])
                if current_avg > es_best_avg + es_min_delta:
                    es_best_avg = current_avg
                    es_best_iteration = iteration
                elif iteration - es_best_iteration >= es_patience:
                    print(f"\n[EarlyStopping] Reward plateau detected at iteration {iteration}.")
                    print(f"  Best rolling avg: {es_best_avg:.4f} at iteration {es_best_iteration}")
                    print(f"  Current rolling avg: {current_avg:.4f}")
                    print(f"  No improvement for {iteration - es_best_iteration} iterations (patience={es_patience})")
                    early_stopped = True
                    plateau_iteration = iteration
                    break

        # ── Periodic logging ──────────────────────────────────────────────
        if iteration % 10 == 0 or iteration == 1:
            mean_reward = iter_reward
            elapsed = time.time() - wall_start
            print(
                f"  iter {iteration:>5}/{args.num_iterations} | "
                f"step {global_step:>10,} | "
                f"reward {mean_reward:>8.2f} | "
                f"pg_loss {pg_loss.item():>8.4f} | "
                f"v_loss {v_loss.item():>8.4f} | "
                f"time {elapsed:>6.0f}s"
            )

            # Save for evaluations.npz
            eval_timesteps.append(global_step)
            eval_results.append(mean_reward)

    elapsed = time.time() - wall_start
    envs.close()

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_path = os.path.join(results_dir, "model.ckpt")
    torch.save(agent.state_dict(), ckpt_path)
    print(f"[Saved] {ckpt_path}")

    # ── Save evaluations.npz for plot.py compatibility ────────────────────────
    if eval_timesteps:
        np.savez(
            os.path.join(results_dir, "evaluations.npz"),
            timesteps=np.array(eval_timesteps),
            results=np.array(eval_results).reshape(-1, 1),
        )

    # ── Final eval stats (from training data, not separate eval) ──────────────
    mean_reward = float(np.mean(eval_results[-5:])) if eval_results else 0.0
    std_reward = float(np.std(eval_results[-5:])) if eval_results else 0.0

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "experiment": name,
        "backend": "racing",
        "hypothesis": hypothesis,
        "config_file": os.path.basename(config_path),
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "timesteps_trained": global_step,
        "racing_kwargs": racing_cfg,
        "level": level,
        "early_stopped": early_stopped,
        "plateau_iteration": plateau_iteration,
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {metrics_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS — {name}")
    print(f"  mean_reward  = {mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"  timesteps    = {global_step:,}")
    print(f"  wall time    = {elapsed:.1f}s")
    print(f"  level        = {level}")
    if early_stopped:
        print(f"  early_stop   = iteration {plateau_iteration} (plateau)")
    print(f"{'='*65}\n")

    # ── Write outbox file ─────────────────────────────────────────────────────
    outbox_dir = os.path.join(lab_dir, "outbox")
    os.makedirs(outbox_dir, exist_ok=True)
    outbox_path = os.path.join(outbox_dir, f"{name}.md")
    with open(outbox_path, "w") as f:
        f.write(f"# {name} — Results\n\n")
        f.write(f"**Backend:** racing (lsy_drone_racing)\n")
        f.write(f"**Hypothesis:** {hypothesis}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| mean_reward | {mean_reward:.3f} +/- {std_reward:.3f} |\n")
        f.write(f"| timesteps_trained | {global_step:,} |\n")
        f.write(f"| wall_time | {elapsed:.1f}s |\n")
        f.write(f"| level | {level} |\n\n")
        f.write(f"*(Linux Claude: write full analysis to "
                f"`results/{name}/EXPERIMENT.md`, then update this file.)*\n")
    print(f"[Updated] {outbox_path}")

    # ── Auto-benchmark (optional) ─────────────────────────────────────────────
    import shutil as _shutil
    bench_cfg = config.get("benchmark", {})
    if bench_cfg.get("enabled", False):
        benchmark_script = os.path.join(lab_dir, "scripts", "benchmark.py")
        if os.path.isfile(benchmark_script) and _shutil.which("pixi") and os.path.isdir("/media/lsy_drone_racing"):
            print(f"\n[AutoBenchmark] Starting structured benchmark...")
            levels = bench_cfg.get("levels", [level])
            if "level2" not in levels and level != "level2" and bench_cfg.get("include_level2", True):
                levels.append("level2")
            n_bench_runs = bench_cfg.get("n_runs", 5)
            controller = bench_cfg.get("controller", "attitude_rl_generic.py")

            env = os.environ.copy()
            env["DRONE_RL_CKPT_PATH"] = ckpt_path
            env["DRONE_RL_GATE_AWARE"] = str(racing_cfg.get("gate_aware", False)).lower()
            env["SCIPY_ARRAY_API"] = "1"

            import subprocess
            try:
                level_args = []
                for lv in levels:
                    level_args.extend(["--level", lv])
                result = subprocess.run(
                    [sys.executable, benchmark_script,
                     "--experiment", name,
                     "--n_runs", str(n_bench_runs),
                     "--controller", controller,
                     ] + level_args,
                    env=env, capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    print(result.stdout)
                    bench_path = os.path.join(results_dir, "benchmark.json")
                    if os.path.isfile(bench_path):
                        with open(bench_path) as f:
                            metrics["benchmark"] = json.load(f)
                        with open(metrics_path, "w") as f:
                            json.dump(metrics, f, indent=2)
                        print(f"[Updated] {metrics_path} (with benchmark results)")
                else:
                    print(f"[AutoBenchmark] FAILED (exit {result.returncode})")
                    if result.stderr:
                        print(result.stderr[-500:])
            except subprocess.TimeoutExpired:
                print("[AutoBenchmark] TIMEOUT (600s) — skipping")
            except Exception as e:
                print(f"[AutoBenchmark] ERROR: {e}")
        else:
            if bench_cfg.get("enabled"):
                print("[AutoBenchmark] Skipping — pixi or lsy_drone_racing not available")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_racing.py configs/exp_NNN.yaml")
        sys.exit(1)

    run(sys.argv[1])
