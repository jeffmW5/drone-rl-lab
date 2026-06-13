"""GPU-vectorized PPO trainer for the AI Grand Prix surrogate environment."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ai_gp_rl.contract import ACTION_NAMES
from ai_gp_rl.env import AIGPEnvConfig, AIGPVectorEnv
from ai_gp_rl.model import ActorCritic


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for field in ("name", "hypothesis", "ai_gp"):
        if field not in config:
            raise ValueError(f"config missing required field: {field}")
    config.setdefault("backend", "ai_gp")
    config.setdefault("budget_seconds", 3600)
    return config


def _build_env_config(section: dict[str, Any], *, evaluation: bool = False) -> AIGPEnvConfig:
    env_values = copy.deepcopy(section.get("env", {}))
    valid_fields = AIGPEnvConfig.__dataclass_fields__
    unknown = sorted(set(env_values) - set(valid_fields))
    if unknown:
        raise ValueError(f"unknown AI-GP environment config fields: {unknown}")
    if evaluation:
        env_values["num_envs"] = int(section.get("eval_num_envs", 128))
        env_values["seed"] = int(section.get("seed", 42)) + 10_000
        env_values["randomization"] = bool(section.get("eval_randomization", False))
        env_values["near_gate_spawn_ratio_start"] = 0.0
        env_values["near_gate_spawn_ratio_end"] = 0.0
        if not env_values["randomization"]:
            env_values["vision_noise_std"] = 0.0
            env_values["vision_dropout_probability"] = 0.0
    else:
        env_values.setdefault("num_envs", int(section.get("num_envs", 4096)))
        env_values.setdefault("seed", int(section.get("seed", 42)))
    env_values.setdefault("device", section.get("device", "cuda"))
    return AIGPEnvConfig(**env_values)


def _save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    global_step: int,
    evaluation: dict[str, float] | None,
    actor_features: tuple[str, ...],
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "schema_version": 1,
        "observation_dim": model.observation_dim,
        "actor_observation_dim": model.actor_observation_dim,
        "action_dim": model.action_dim,
        "hidden_sizes": list(model.hidden_sizes),
        "actor_features": list(actor_features),
        "action_names": list(ACTION_NAMES),
        "action_semantics": (
            "tanh-normalized [collective offset, roll rate, pitch rate, yaw rate]; "
            "negative pitch is forward"
        ),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "evaluation": evaluation,
            "config": config,
            "metadata": metadata,
        },
        path,
    )


def _load_initial_actor(
    model: ActorCritic,
    checkpoint_path: Path,
    actor_features: tuple[str, ...],
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location=next(model.parameters()).device, weights_only=False
    )
    metadata = checkpoint.get("metadata", {})
    if metadata.get("policy_role") != "distilled_live_student":
        raise ValueError("initial actor checkpoint must be a distilled live student")
    if tuple(metadata.get("actor_features", ())) != actor_features:
        raise ValueError("initial actor checkpoint feature contract does not match")
    if str(metadata.get("policy_architecture", "mlp")) != "mlp":
        raise ValueError("PPO fine-tuning currently requires an MLP student")

    source_state = checkpoint["model_state_dict"]
    actor_state = {
        key.removeprefix("actor_mean."): value
        for key, value in source_state.items()
        if key.startswith("actor_mean.")
    }
    expected_keys = set(model.actor_mean.state_dict())
    if set(actor_state) != expected_keys:
        raise ValueError("initial actor checkpoint network shape does not match")
    model.actor_mean.load_state_dict(actor_state)
    return metadata


@torch.no_grad()
def evaluate(
    model: ActorCritic,
    env: AIGPVectorEnv,
    episode_count: int,
) -> dict[str, float]:
    observation, _ = env.reset()
    returns: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    successes: list[torch.Tensor] = []
    collisions: list[torch.Tensor] = []
    out_of_bounds: list[torch.Tensor] = []
    vertical_runaways: list[torch.Tensor] = []
    distance_reduction: list[torch.Tensor] = []
    max_altitude = env.position[:, 2].clone()
    upward_run = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    longest_upward_run = torch.zeros_like(upward_run)
    completed = 0
    while completed < episode_count:
        action, _, _, _ = model.get_action_and_value(
            observation, deterministic=True
        )
        observation, _, _, _, info = env.step(action)
        max_altitude = torch.maximum(max_altitude, info["position"][:, 2])
        sustained_upward = (
            (info["velocity"][:, 2] > 2.0)
            & (info["position"][:, 2] > 1.5)
        )
        upward_run = torch.where(
            sustained_upward, upward_run + 1, torch.zeros_like(upward_run)
        )
        longest_upward_run = torch.maximum(longest_upward_run, upward_run)
        done_ids = torch.where(info["done"])[0]
        if done_ids.numel() == 0:
            continue
        remaining = episode_count - completed
        done_ids = done_ids[:remaining]
        returns.append(info["episode_return"][done_ids].detach().cpu())
        gates.append(info["gates_passed"][done_ids].float().detach().cpu())
        successes.append(info["success"][done_ids].float().detach().cpu())
        collisions.append(info["collision"][done_ids].float().detach().cpu())
        out_of_bounds.append(
            info["out_of_bounds"][done_ids].float().detach().cpu()
        )
        vertical_runaways.append(
            (
                (max_altitude[done_ids] >= 0.95 * env.config.max_altitude_m)
                | (
                    longest_upward_run[done_ids].float() * env.config.dt
                    >= 1.0
                )
            )
            .float()
            .detach()
            .cpu()
        )
        distance_reduction.append(
            info["distance_reduction"][done_ids].detach().cpu()
        )
        completed += done_ids.numel()
        all_done_ids = torch.where(info["done"])[0]
        max_altitude[all_done_ids] = env.position[all_done_ids, 2]
        upward_run[all_done_ids] = 0
        longest_upward_run[all_done_ids] = 0

    return {
        "mean_return": float(torch.cat(returns).mean()),
        "mean_gates": float(torch.cat(gates).mean()),
        "success_rate": float(torch.cat(successes).mean()),
        "collision_rate": float(torch.cat(collisions).mean()),
        "out_of_bounds_rate": float(torch.cat(out_of_bounds).mean()),
        "vertical_runaway_rate": float(torch.cat(vertical_runaways).mean()),
        "mean_distance_reduction_m": float(torch.cat(distance_reduction).mean()),
        "episodes": completed,
    }


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    section = config["ai_gp"]
    budget_seconds = int(config["budget_seconds"])
    env_config = _build_env_config(section)
    device = torch.device(env_config.device)
    allow_cpu = bool(section.get("allow_cpu", False))
    if device.type == "cuda" and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError(
                "AI-GP training requested CUDA but no CUDA device is available. "
                "Set ai_gp.allow_cpu=true only for smoke tests."
            )
        device = torch.device("cpu")
        env_config.device = "cpu"

    seed = int(section.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root = Path(__file__).resolve().parent
    results_dir = root / "results" / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, results_dir / "config.yaml")

    print(f"\n{'=' * 68}")
    print(f"  EXPERIMENT: {config['name']}")
    print("  BACKEND:    ai_gp (Torch GPU vector env + PPO)")
    print(f"  DEVICE:     {device}")
    print(f"  ENVS:       {env_config.num_envs:,}")
    print(f"  BUDGET:     {budget_seconds}s")
    print(f"  OUTPUT:     {results_dir}")
    print(f"{'=' * 68}\n")

    env = AIGPVectorEnv(env_config)
    eval_config = _build_env_config(section, evaluation=True)
    eval_config.device = str(device)
    eval_env = AIGPVectorEnv(eval_config)
    eval_env.set_curriculum(1.0)

    hidden_sizes = tuple(int(value) for value in section.get("hidden_sizes", [128, 128]))
    model = ActorCritic(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        actor_observation_dim=env.actor_observation_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)
    initial_actor_checkpoint = section.get("initial_actor_checkpoint")
    initial_actor_metadata: dict[str, Any] | None = None
    if initial_actor_checkpoint:
        checkpoint_path = Path(initial_actor_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        initial_actor_metadata = _load_initial_actor(
            model, checkpoint_path, env.actor_feature_names
        )
        print(f"  INITIAL:    {checkpoint_path}")
    model.actor_logstd.data.fill_(float(section.get("initial_logstd", -0.5)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(section.get("learning_rate", 3e-4)),
        eps=1e-5,
    )

    num_steps = int(section.get("num_steps", 32))
    total_timesteps = int(section.get("total_timesteps", 100_000_000))
    batch_size = env.num_envs * num_steps
    num_minibatches = int(section.get("num_minibatches", 16))
    if batch_size % num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    minibatch_size = batch_size // num_minibatches
    total_iterations = max(total_timesteps // batch_size, 1)

    gamma = float(section.get("gamma", 0.99))
    gae_lambda = float(section.get("gae_lambda", 0.95))
    update_epochs = int(section.get("update_epochs", 4))
    clip_coef = float(section.get("clip_coef", 0.2))
    ent_coef = float(section.get("ent_coef", 0.005))
    vf_coef = float(section.get("vf_coef", 0.5))
    max_grad_norm = float(section.get("max_grad_norm", 0.5))
    anneal_lr = bool(section.get("anneal_lr", True))
    base_learning_rate = float(section.get("learning_rate", 3e-4))
    target_kl = section.get("target_kl")
    eval_interval = int(section.get("eval_interval_iterations", 25))
    eval_episodes = int(section.get("eval_episodes", 256))

    observations = torch.zeros(
        (num_steps, env.num_envs, env.observation_dim), device=device
    )
    actions = torch.zeros(
        (num_steps, env.num_envs, env.action_dim), device=device
    )
    logprobs = torch.zeros((num_steps, env.num_envs), device=device)
    rewards = torch.zeros((num_steps, env.num_envs), device=device)
    dones = torch.zeros((num_steps, env.num_envs), device=device)
    values = torch.zeros((num_steps, env.num_envs), device=device)

    next_observation, _ = env.reset()
    next_done = torch.zeros(env.num_envs, device=device)
    global_step = 0
    wall_start = time.time()
    best_score = (float("-inf"), -1.0, -1.0, float("-inf"))
    best_evaluation: dict[str, float] | None = None
    history: list[dict[str, float]] = []

    for iteration in range(1, total_iterations + 1):
        elapsed = time.time() - wall_start
        if elapsed >= budget_seconds:
            print(f"[TimeBudget] stopped after {elapsed:.1f}s")
            break

        curriculum_steps = int(section.get("curriculum_steps", 30_000_000))
        curriculum_progress = min(global_step / max(curriculum_steps, 1), 1.0)
        env.set_curriculum(curriculum_progress)
        if anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / total_iterations
            optimizer.param_groups[0]["lr"] = fraction * base_learning_rate

        completed_returns: list[torch.Tensor] = []
        completed_gates: list[torch.Tensor] = []
        completed_successes: list[torch.Tensor] = []
        for step in range(num_steps):
            global_step += env.num_envs
            observations[step] = next_observation
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_observation)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_done = (terminated | truncated).float()
            rewards[step] = reward

            done_ids = torch.where(info["done"])[0]
            if done_ids.numel():
                completed_returns.append(info["episode_return"][done_ids].detach())
                completed_gates.append(info["gates_passed"][done_ids].float().detach())
                completed_successes.append(info["success"][done_ids].float().detach())

        with torch.no_grad():
            next_value = model.get_value(next_observation).reshape(-1)
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(env.num_envs, device=device)
            for step in reversed(range(num_steps)):
                if step == num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[step + 1]
                    next_values = values[step + 1]
                delta = (
                    rewards[step]
                    + gamma * next_values * next_nonterminal
                    - values[step]
                )
                last_gae = (
                    delta
                    + gamma * gae_lambda * next_nonterminal * last_gae
                )
                advantages[step] = last_gae
            returns = advantages + values

        batch_observations = observations.reshape(-1, env.observation_dim)
        batch_actions = actions.reshape(-1, env.action_dim)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        clip_fractions: list[float] = []
        approx_kl = torch.tensor(0.0, device=device)
        for _ in range(update_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    batch_observations[indices], batch_actions[indices]
                )
                log_ratio = new_logprob - batch_logprobs[indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        float(((ratio - 1.0).abs() > clip_coef).float().mean())
                    )

                minibatch_advantage = batch_advantages[indices]
                minibatch_advantage = (
                    minibatch_advantage - minibatch_advantage.mean()
                ) / (minibatch_advantage.std() + 1e-8)
                policy_loss = torch.maximum(
                    -minibatch_advantage * ratio,
                    -minibatch_advantage
                    * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
                ).mean()

                new_value = new_value.view(-1)
                value_unclipped = (new_value - batch_returns[indices]).square()
                value_clipped = batch_values[indices] + torch.clamp(
                    new_value - batch_values[indices], -clip_coef, clip_coef
                )
                value_loss = 0.5 * torch.maximum(
                    value_unclipped,
                    (value_clipped - batch_returns[indices]).square(),
                ).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            if target_kl is not None and approx_kl > float(target_kl):
                break

        elapsed = time.time() - wall_start
        steps_per_second = global_step / max(elapsed, 1e-6)
        train_return = (
            float(torch.cat(completed_returns).mean())
            if completed_returns
            else float("nan")
        )
        train_gates = (
            float(torch.cat(completed_gates).mean())
            if completed_gates
            else float("nan")
        )
        train_success = (
            float(torch.cat(completed_successes).mean())
            if completed_successes
            else float("nan")
        )
        record = {
            "iteration": iteration,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "train_return": train_return,
            "train_mean_gates": train_gates,
            "train_success_rate": train_success,
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy_loss),
            "approx_kl": float(approx_kl),
            "clip_fraction": sum(clip_fractions) / max(len(clip_fractions), 1),
            "curriculum_progress": curriculum_progress,
            "near_gate_spawn_ratio": env.near_gate_spawn_ratio,
            "soft_collisions": float(env.soft_collisions),
        }
        history.append(record)

        if iteration == 1 or iteration % 10 == 0:
            print(
                f"iter {iteration:>5}/{total_iterations} | "
                f"step {global_step:>11,} | "
                f"SPS {steps_per_second:>9,.0f} | "
                f"return {train_return:>8.2f} | gates {train_gates:>5.2f} | "
                f"success {train_success:>6.1%}"
            )

        if iteration % eval_interval == 0:
            evaluation = evaluate(model, eval_env, eval_episodes)
            bounded_progress = (
                evaluation["mean_gates"]
                * (1.0 - evaluation["collision_rate"])
                * (1.0 - evaluation["out_of_bounds_rate"])
                * (1.0 - evaluation["vertical_runaway_rate"])
            )
            score = (
                bounded_progress + evaluation["success_rate"],
                evaluation["success_rate"],
                evaluation["mean_gates"],
                evaluation["mean_distance_reduction_m"],
            )
            print(
                "  [Eval] "
                f"success={evaluation['success_rate']:.1%} "
                f"gates={evaluation['mean_gates']:.2f} "
                f"collision={evaluation['collision_rate']:.1%} "
                f"progress={evaluation['mean_distance_reduction_m']:.2f}m"
            )
            if score > best_score:
                best_score = score
                best_evaluation = evaluation
                _save_checkpoint(
                    results_dir / "best_policy.pt",
                    model,
                    optimizer,
                    config,
                    global_step,
                    evaluation,
                    env.actor_feature_names,
                    {
                        "policy_role": "distilled_live_student",
                        "policy_architecture": "mlp",
                        "observation_contract": (
                            initial_actor_metadata or {}
                        ).get("observation_contract"),
                        "base_actor_features": (
                            initial_actor_metadata or {}
                        ).get("base_actor_features"),
                        "history_length": (
                            initial_actor_metadata or {}
                        ).get("history_length", 1),
                        "source_initial_actor_checkpoint": (
                            str(initial_actor_checkpoint)
                            if initial_actor_checkpoint
                            else None
                        ),
                        "source_teacher_checkpoint": (
                            initial_actor_metadata or {}
                        ).get("source_teacher_checkpoint"),
                        "source_teacher_global_step": (
                            initial_actor_metadata or {}
                        ).get("source_teacher_global_step"),
                    }
                    if initial_actor_checkpoint
                    else None,
                )

    final_evaluation = evaluate(model, eval_env, eval_episodes)
    _save_checkpoint(
        results_dir / "final_policy.pt",
        model,
        optimizer,
        config,
        global_step,
        final_evaluation,
        env.actor_feature_names,
        {
            "policy_role": "distilled_live_student",
            "policy_architecture": "mlp",
            "observation_contract": (initial_actor_metadata or {}).get(
                "observation_contract"
            ),
            "base_actor_features": (initial_actor_metadata or {}).get(
                "base_actor_features"
            ),
            "history_length": (initial_actor_metadata or {}).get(
                "history_length", 1
            ),
            "source_initial_actor_checkpoint": str(initial_actor_checkpoint),
            "source_teacher_checkpoint": (initial_actor_metadata or {}).get(
                "source_teacher_checkpoint"
            ),
            "source_teacher_global_step": (initial_actor_metadata or {}).get(
                "source_teacher_global_step"
            ),
        }
        if initial_actor_checkpoint
        else None,
    )
    if best_evaluation is None:
        best_evaluation = final_evaluation
        _save_checkpoint(
            results_dir / "best_policy.pt",
            model,
            optimizer,
            config,
            global_step,
            final_evaluation,
            env.actor_feature_names,
            {
                "policy_role": "distilled_live_student",
                "policy_architecture": "mlp",
                "observation_contract": (initial_actor_metadata or {}).get(
                    "observation_contract"
                ),
                "base_actor_features": (initial_actor_metadata or {}).get(
                    "base_actor_features"
                ),
                "history_length": (initial_actor_metadata or {}).get(
                    "history_length", 1
                ),
                "source_initial_actor_checkpoint": str(
                    initial_actor_checkpoint
                ),
                "source_teacher_checkpoint": (
                    initial_actor_metadata or {}
                ).get("source_teacher_checkpoint"),
                "source_teacher_global_step": (
                    initial_actor_metadata or {}
                ).get("source_teacher_global_step"),
            }
            if initial_actor_checkpoint
            else None,
        )

    elapsed = time.time() - wall_start
    metrics = {
        "experiment": config["name"],
        "hypothesis": config["hypothesis"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "global_step": global_step,
        "device": str(device),
        "num_envs": env.num_envs,
        "environment": asdict(env_config),
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "history": history,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n[Saved] {results_dir / 'best_policy.pt'}")
    print(f"[Saved] {results_dir / 'final_policy.pt'}")
    print(f"[Saved] {results_dir / 'metrics.json'}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_ai_gp.py configs/ai_gp_NNN.yaml")
    run(sys.argv[1])


if __name__ == "__main__":
    main()
