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


# =============================================================================
# OBSERVATION NORMALIZATION (exp_071+)
# =============================================================================

class RunningObsNormalizer:
    """Running mean/std observation normalizer (Welford's online algorithm).

    Tracks per-dimension running mean and variance across all envs.
    Normalizes: obs_norm = clip((obs - mean) / sqrt(var + eps), -clip_val, clip_val)

    Compatible with checkpoint save/load: call state_dict() / load_state_dict().
    """

    def __init__(self, obs_dim: int, eps: float = 1e-8, clip_val: float = 10.0):
        self.obs_dim = obs_dim
        self.eps = eps
        self.clip_val = clip_val
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self._M2 = np.zeros(obs_dim, dtype=np.float64)

    def update(self, obs_batch: np.ndarray):
        """Update running stats with a batch of observations. obs_batch: (N, obs_dim)."""
        batch = obs_batch.reshape(-1, self.obs_dim).astype(np.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        # Parallel Welford merge
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / max(total_count, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self._M2 = m_a + m_b + delta ** 2 * self.count * batch_count / max(total_count, 1)
        self.mean = new_mean
        self.count = total_count
        self.var = self._M2 / max(self.count, 1)

    def normalize(self, obs):
        """Normalize observation tensor (torch or numpy). Returns same type."""
        import torch
        if isinstance(obs, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=obs.device)
            std = torch.sqrt(torch.tensor(self.var + self.eps, dtype=torch.float32, device=obs.device))
            return torch.clamp((obs - mean) / std, -self.clip_val, self.clip_val)
        else:
            std = np.sqrt(self.var + self.eps).astype(np.float32)
            return np.clip((obs - self.mean.astype(np.float32)) / std, -self.clip_val, self.clip_val)

    def state_dict(self) -> dict:
        return {
            "obs_norm_mean": self.mean.copy(),
            "obs_norm_var": self.var.copy(),
            "obs_norm_count": self.count,
        }

    def load_state_dict(self, d: dict):
        self.mean = d["obs_norm_mean"]
        self.var = d["obs_norm_var"]
        self.count = d["obs_norm_count"]


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


def _startup_log(start_time: float, message: str) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[Startup +{elapsed:6.1f}s] {message}", flush=True)


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

def evaluate_racing(agent: Agent, envs_fn, device, n_episodes: int = 10, jax_on_gpu: bool = False,
                     obs_normalizer: RunningObsNormalizer = None) -> dict:
    """Run evaluation episodes and return stats."""
    eval_envs = envs_fn()
    rewards_all = []
    lengths_all = []

    obs, info = eval_envs.reset()
    obs_np = _to_np(obs) if not isinstance(obs, np.ndarray) else obs
    if obs_normalizer is not None:
        obs_tensor = torch.Tensor(obs_normalizer.normalize(obs_np)).to(device)
    else:
        obs_tensor = torch.Tensor(obs_np).to(device)

    episode_rewards = np.zeros(eval_envs.num_envs)
    episode_lengths = np.zeros(eval_envs.num_envs)
    completed = 0

    while completed < n_episodes:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True)

        obs, reward, terminated, truncated, info = eval_envs.step(_action_for_env(action, jax_on_gpu))
        obs_np = _to_np(obs)
        if obs_normalizer is not None:
            obs_tensor = torch.Tensor(obs_normalizer.normalize(obs_np)).to(device)
        else:
            obs_tensor = torch.Tensor(obs_np).to(device)

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
    startup_start = time.perf_counter()
    _startup_log(startup_start, f"Loading config from {config_path}")
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
    _startup_log(startup_start, "Built training args and selected devices")

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
            "oob_coef": racing_cfg.get("oob_coef", 0.0),
            "z_low": racing_cfg.get("z_low", 0.0),
            "z_high": racing_cfg.get("z_high", 2.0),
            "alt_coef": racing_cfg.get("alt_coef", 0.0),
            "survive_coef": racing_cfg.get("survive_coef", 0.0),
            "vz_coef": racing_cfg.get("vz_coef", 0.0),
            "vz_threshold": racing_cfg.get("vz_threshold", 0.5),
            "random_gate_start": racing_cfg.get("random_gate_start", False),
            "random_gate_ratio": racing_cfg.get("random_gate_ratio", 1.0),
            "spawn_offset": racing_cfg.get("spawn_offset", 0.75),
            "spawn_pos_noise": racing_cfg.get("spawn_pos_noise", 0.15),
            "spawn_vel_noise": racing_cfg.get("spawn_vel_noise", 0.3),
            # exp_056: bilateral progress
            "bilateral_progress": racing_cfg.get("bilateral_progress", False),
            # exp_057: body-frame gate observations
            "body_frame_obs": racing_cfg.get("body_frame_obs", False),
            # exp_058: soft-collision curriculum
            "soft_collision": racing_cfg.get("soft_collision", False),
            "soft_collision_penalty": racing_cfg.get("soft_collision_penalty", 5.0),
            "soft_collision_steps": racing_cfg.get("soft_collision_steps", 5000000),
            # exp_059: asymmetric actor-critic
            "asymmetric_critic": racing_cfg.get("asymmetric_critic", False),
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
    _startup_log(startup_start, "Environment construction complete")

    # ── Observation normalization (exp_071+) ──────────────────────────────────
    obs_normalize = racing_cfg.get("obs_normalize", False)
    obs_normalizer = None
    if obs_normalize:
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        obs_norm_clip = racing_cfg.get("obs_norm_clip", 10.0)
        obs_normalizer = RunningObsNormalizer(obs_dim, clip_val=obs_norm_clip)
        print(f"[INFO] Observation normalization: ON (dim={obs_dim}, clip={obs_norm_clip})")

    # ── Create agent ──────────────────────────────────────────────────────────
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    hidden_size = args.hidden_size
    if racing_cfg.get("asymmetric_critic", False):
        from lsy_drone_racing.control.train_rl import AsymmetricAgent
        actor_obs_dim = getattr(envs, 'actor_obs_dim', obs_shape[0])
        agent = AsymmetricAgent(obs_shape, act_shape, actor_obs_dim, hidden_size=hidden_size).to(device)
        print(f"[INFO] AsymmetricAgent: actor={actor_obs_dim}D, total={obs_shape[0]}D, hidden={hidden_size}")
    else:
        agent = Agent(obs_shape, act_shape, hidden_size=hidden_size).to(device)
    if hidden_size != 64:
        print(f"[INFO] Network hidden_size={hidden_size} (params: {sum(p.numel() for p in agent.parameters()):,})")

    # Load pretrained checkpoint if specified (for fine-tuning)
    pretrained_ckpt = racing_cfg.get("pretrained_ckpt", None)
    if pretrained_ckpt:
        agent.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        print(f"[INFO] Loaded pretrained checkpoint: {pretrained_ckpt}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    _startup_log(startup_start, "Model and optimizer initialized")

    # ── Training storage ──────────────────────────────────────────────────────
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + act_shape).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminated_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # For evaluations.npz compatibility
    eval_timesteps = []
    eval_results = []

    # Optional periodic deterministic evaluation on a matched training env.
    det_eval_cfg = racing_cfg.get("periodic_deterministic_eval", {})
    det_eval_enabled = det_eval_cfg.get("enabled", False)
    det_eval_every = int(det_eval_cfg.get("every_iterations", 50))
    det_eval_episodes = int(det_eval_cfg.get("n_episodes", 8))
    det_eval_num_envs = int(det_eval_cfg.get("num_envs", min(args.num_envs, 8)))
    det_eval_save_best = det_eval_cfg.get("save_best_checkpoint", True)
    det_eval_timesteps = []
    det_eval_results = []
    det_eval_lengths = []
    best_det_eval = float("-inf")
    best_det_iteration = None
    best_det_global_step = None
    best_det_ckpt_path = os.path.join(results_dir, "best_det.ckpt")
    final_det_eval = None

    if det_eval_enabled:
        print(
            "[INFO] Periodic deterministic eval: "
            f"every={det_eval_every} iter, episodes={det_eval_episodes}, "
            f"num_envs={det_eval_num_envs}, save_best={det_eval_save_best}"
        )

        if env_type == "race":
            from lsy_drone_racing.control.train_race import make_race_envs

            def make_det_eval_envs():
                return make_race_envs(
                    config=race_config,
                    num_envs=det_eval_num_envs,
                    jax_device=jax_device,
                    torch_device=device,
                    coefs=dict(race_coefs),
                )
        else:
            def make_det_eval_envs():
                return make_envs(
                    config=f"{level}.toml",
                    num_envs=det_eval_num_envs,
                    jax_device=jax_device,
                    torch_device=device,
                    coefs=dict(reward_coefs),
                )

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
    _startup_log(startup_start, "Resetting environments (may trigger initial JAX compile)")
    next_obs, _ = envs.reset()
    next_obs_np = _to_np(next_obs)
    if obs_normalizer is not None:
        obs_normalizer.update(next_obs_np)
        next_obs = torch.Tensor(obs_normalizer.normalize(next_obs_np)).to(device)
    else:
        next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    _startup_log(startup_start, "Environment reset complete")

    print("[Training] Starting CleanRL PPO loop...")
    _startup_log(startup_start, "Entering PPO rollout loop")
    first_step_logged = False
    first_iteration_logged = False

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
            if not first_step_logged:
                _startup_log(startup_start, "Starting first rollout step (compile/warmup likely here)")
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(_action_for_env(action, jax_on_gpu))
            if not first_step_logged:
                _startup_log(startup_start, "First rollout env.step complete")
                first_step_logged = True
            reward = torch.tensor(_to_np(reward), dtype=torch.float32).to(device)
            rewards_buf[step] = reward
            terminated_buf[step] = torch.Tensor(_to_np(terminated).astype(float)).to(device)
            next_obs_np = _to_np(next_obs)
            if obs_normalizer is not None:
                obs_normalizer.update(next_obs_np)
                next_obs = torch.Tensor(obs_normalizer.normalize(next_obs_np)).to(device)
            else:
                next_obs = torch.Tensor(next_obs_np).to(device)
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
                    nextnonterminal = 1.0 - terminated_buf[t + 1]
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

        if det_eval_enabled and iteration % det_eval_every == 0:
            det_stats = evaluate_racing(
                agent,
                make_det_eval_envs,
                device,
                n_episodes=det_eval_episodes,
                jax_on_gpu=jax_on_gpu,
                obs_normalizer=obs_normalizer,
            )
            det_eval_timesteps.append(global_step)
            det_eval_results.append(det_stats["mean_reward"])
            det_eval_lengths.append(det_stats["mean_episode_length"])
            print(
                "  [DetEval] "
                f"iter {iteration:>5} | step {global_step:>10,} | "
                f"reward {det_stats['mean_reward']:>8.2f} +/- {det_stats['std_reward']:<6.2f} | "
                f"ep_len {det_stats['mean_episode_length']:>7.1f}"
            )

            if det_eval_save_best and det_stats["mean_reward"] > best_det_eval:
                best_det_eval = det_stats["mean_reward"]
                best_det_iteration = iteration
                best_det_global_step = global_step
                best_ckpt_data = agent.state_dict()
                if obs_normalizer is not None:
                    best_ckpt_data.update(obs_normalizer.state_dict())
                torch.save(best_ckpt_data, best_det_ckpt_path)
                print(
                    f"  [DetEval] New best checkpoint: {best_det_ckpt_path} "
                    f"(reward={best_det_eval:.2f})"
                )

        if not first_iteration_logged:
            _startup_log(startup_start, f"First training iteration complete at global_step={global_step:,}")
            first_iteration_logged = True

    elapsed = time.time() - wall_start
    envs.close()

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_path = os.path.join(results_dir, "model.ckpt")
    ckpt_data = agent.state_dict()
    if obs_normalizer is not None:
        ckpt_data.update(obs_normalizer.state_dict())
    torch.save(ckpt_data, ckpt_path)
    print(f"[Saved] {ckpt_path}")

    if det_eval_enabled:
        final_det_eval = evaluate_racing(
            agent,
            make_det_eval_envs,
            device,
            n_episodes=det_eval_episodes,
            jax_on_gpu=jax_on_gpu,
            obs_normalizer=obs_normalizer,
        )
        if not det_eval_timesteps or det_eval_timesteps[-1] != global_step:
            det_eval_timesteps.append(global_step)
            det_eval_results.append(final_det_eval["mean_reward"])
            det_eval_lengths.append(final_det_eval["mean_episode_length"])
        if final_det_eval["mean_reward"] > best_det_eval:
            best_det_eval = final_det_eval["mean_reward"]
            best_det_iteration = iteration
            best_det_global_step = global_step
            if det_eval_save_best:
                final_ckpt_data = agent.state_dict()
                if obs_normalizer is not None:
                    final_ckpt_data.update(obs_normalizer.state_dict())
                torch.save(final_ckpt_data, best_det_ckpt_path)
                print(
                    f"[FinalDetEval] Updated best checkpoint: {best_det_ckpt_path} "
                    f"(reward={best_det_eval:.2f})"
                )
        print(
            "[FinalDetEval] "
            f"reward {final_det_eval['mean_reward']:.2f} +/- {final_det_eval['std_reward']:.2f} | "
            f"ep_len {final_det_eval['mean_episode_length']:.1f}"
        )

    # ── Save evaluations.npz for plot.py compatibility ────────────────────────
    if eval_timesteps:
        np.savez(
            os.path.join(results_dir, "evaluations.npz"),
            timesteps=np.array(eval_timesteps),
            results=np.array(eval_results).reshape(-1, 1),
        )

    if det_eval_timesteps:
        np.savez(
            os.path.join(results_dir, "deterministic_evaluations.npz"),
            timesteps=np.array(det_eval_timesteps),
            rewards=np.array(det_eval_results),
            mean_episode_lengths=np.array(det_eval_lengths),
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
        "periodic_deterministic_eval_enabled": det_eval_enabled,
        "best_det_eval_mean_reward": round(best_det_eval, 3) if best_det_iteration is not None else None,
        "best_det_eval_iteration": best_det_iteration,
        "best_det_eval_global_step": best_det_global_step,
        "best_det_ckpt_path": (
            os.path.basename(best_det_ckpt_path)
            if det_eval_save_best and best_det_iteration is not None
            else None
        ),
        "final_det_eval_mean_reward": (
            round(final_det_eval["mean_reward"], 3) if final_det_eval else None
        ),
        "final_det_eval_std_reward": (
            round(final_det_eval["std_reward"], 3) if final_det_eval else None
        ),
        "final_det_eval_mean_episode_length": (
            round(final_det_eval["mean_episode_length"], 3) if final_det_eval else None
        ),
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {metrics_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS — {name}")
    print(f"  mean_reward  = {mean_reward:.3f} +/- {std_reward:.3f}")
    if best_det_iteration is not None:
        print(f"  best_det_eval = {best_det_eval:.3f} at iter {best_det_iteration}")
    if final_det_eval:
        print(
            "  final_det_eval = "
            f"{final_det_eval['mean_reward']:.3f} +/- {final_det_eval['std_reward']:.3f}"
        )
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
