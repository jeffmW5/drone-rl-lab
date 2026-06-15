"""Behavior cloning for the Swift-style AI-GP full-course teacher."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.model import ActorCritic
from ai_gp_rl.swift_expert import geometric_gate_teacher_action
from train_ai_gp import (
    _build_env_config,
    _load_initial_actor,
    _save_checkpoint,
    evaluate,
    load_config,
)


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    section = config["ai_gp"]
    bc_section: dict[str, Any] = config.get("swift_bc", {})
    budget_seconds = int(config.get("budget_seconds", 3600))

    env_config = _build_env_config(section)
    device = torch.device(env_config.device)
    allow_cpu = bool(section.get("allow_cpu", False))
    if device.type == "cuda" and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("Swift BC requested CUDA but no CUDA device is available")
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

    env = AIGPVectorEnv(env_config)
    env.set_curriculum(1.0)
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
    if initial_actor_checkpoint:
        checkpoint_path = Path(initial_actor_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        _load_initial_actor(
            model,
            checkpoint_path,
            env.actor_feature_names,
            allow_teacher=bool(section.get("allow_teacher_initialization", False)),
        )
        print(f"  INITIAL:    {checkpoint_path}")

    optimizer = torch.optim.Adam(
        model.actor_mean.parameters(),
        lr=float(bc_section.get("learning_rate", 3e-4)),
        eps=1e-5,
    )
    action_loss_weights = torch.tensor(
        bc_section.get("action_loss_weights", [1.0, 1.0, 1.0, 1.0]),
        device=device,
        dtype=torch.float32,
    )
    if action_loss_weights.shape != (env.action_dim,):
        raise ValueError("action_loss_weights must contain four values")

    total_interactions = int(bc_section.get("total_interactions", 20_000_000))
    total_updates = max((total_interactions + env.num_envs - 1) // env.num_envs, 1)
    eval_interval = int(bc_section.get("eval_interval_updates", 250))
    eval_episodes = int(section.get("eval_episodes", 256))
    student_rollout_start = int(
        bc_section.get("student_rollout_start_interactions", total_interactions + 1)
    )
    student_rollout_end = int(
        bc_section.get("student_rollout_end_interactions", total_interactions + 1)
    )
    max_student_rollout_ratio = float(
        bc_section.get("max_student_rollout_ratio", 0.0)
    )

    print(f"\n{'=' * 68}")
    print(f"  EXPERIMENT: {config['name']}")
    print("  BACKEND:    ai_gp_swift_bc (geometric full-course teacher)")
    print(f"  DEVICE:     {device}")
    print(f"  ENVS:       {env.num_envs:,}")
    print(f"  UPDATES:    {total_updates:,}")
    print(f"  BUDGET:     {budget_seconds}s")
    print(f"  OUTPUT:     {results_dir}")
    print(f"{'=' * 68}\n")

    observation, _ = env.reset()
    wall_start = time.time()
    history: list[dict[str, float]] = []
    best_evaluation: dict[str, float] | None = None
    best_score = (float("-inf"), -1.0, -1.0, float("-inf"))
    metadata_extra = {
        "training_method": "swift_geometric_full_course_bc",
        "expert": "geometric_gate_teacher_v1",
        "policy_architecture": "mlp",
        "source_initial_actor_checkpoint": str(initial_actor_checkpoint)
        if initial_actor_checkpoint
        else None,
    }

    for update in range(1, total_updates + 1):
        elapsed = time.time() - wall_start
        if elapsed >= budget_seconds:
            print(f"[TimeBudget] stopped after {elapsed:.1f}s")
            break
        interactions = update * env.num_envs
        actor_observation = observation[:, : env.actor_observation_dim].detach()
        with torch.no_grad():
            target_action = geometric_gate_teacher_action(env)

        predicted_action = torch.tanh(model.actor_mean(actor_observation))
        squared_error = (predicted_action - target_action).square()
        loss = (squared_error * action_loss_weights).mean() / action_loss_weights.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor_mean.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            rollout_progress = (
                (interactions - student_rollout_start)
                / max(student_rollout_end - student_rollout_start, 1)
            )
            student_rollout_ratio = (
                max(0.0, min(rollout_progress, 1.0)) * max_student_rollout_ratio
            )
            use_student = (
                torch.rand(
                    (env.num_envs, 1),
                    device=device,
                    generator=env.generator,
                )
                < student_rollout_ratio
            )
            executed_action = torch.where(
                use_student,
                predicted_action.detach(),
                target_action,
            )
            raw_action = torch.atanh(executed_action.clamp(-0.999, 0.999))
            observation, _, _, _, _ = env.step(raw_action)

        if update == 1 or update % 25 == 0:
            elapsed = time.time() - wall_start
            record = {
                "update": update,
                "interactions": interactions,
                "imitation_loss": float(loss.detach()),
                "collective_mae": float(
                    (predicted_action[:, 0] - target_action[:, 0]).abs().mean()
                ),
                "roll_mae": float(
                    (predicted_action[:, 1] - target_action[:, 1]).abs().mean()
                ),
                "pitch_mae": float(
                    (predicted_action[:, 2] - target_action[:, 2]).abs().mean()
                ),
                "yaw_mae": float(
                    (predicted_action[:, 3] - target_action[:, 3]).abs().mean()
                ),
                "student_rollout_ratio": student_rollout_ratio,
                "interactions_per_second": interactions / max(elapsed, 1e-6),
            }
            history.append(record)
            print(
                f"update {update:>5}/{total_updates} | "
                f"interactions {interactions:>11,} | "
                f"loss {float(loss.detach()):.6f} | "
                f"act_mae {record['collective_mae']:.3f}/"
                f"{record['roll_mae']:.3f}/"
                f"{record['pitch_mae']:.3f}/"
                f"{record['yaw_mae']:.3f} | "
                f"student {student_rollout_ratio:>5.1%} | "
                f"IPS {record['interactions_per_second']:,.0f}"
            )

        if update % eval_interval == 0 or update == total_updates:
            evaluation = evaluate(model, eval_env, eval_episodes)
            print(
                "  [Eval] "
                f"success={evaluation['success_rate']:.1%} "
                f"gates={evaluation['mean_gates']:.2f} "
                f"collision={evaluation['collision_rate']:.1%} "
                f"missed={evaluation['missed_gate_rate']:.1%} "
                f"vertical={evaluation['vertical_runaway_rate']:.1%}"
            )
            bounded_progress = (
                evaluation["mean_gates"]
                * (1.0 - evaluation["collision_rate"])
                * (1.0 - evaluation["out_of_bounds_rate"])
                * (1.0 - evaluation["missed_gate_rate"])
                * (1.0 - evaluation["vertical_runaway_rate"])
            )
            score = (
                bounded_progress + evaluation["success_rate"],
                evaluation["success_rate"],
                evaluation["mean_gates"],
                evaluation["mean_distance_reduction_m"],
            )
            if score > best_score:
                best_score = score
                best_evaluation = evaluation
                _save_checkpoint(
                    results_dir / "best_policy.pt",
                    model,
                    optimizer,
                    config,
                    interactions,
                    evaluation,
                    env.actor_feature_names,
                    metadata_extra,
                )

    final_evaluation = evaluate(model, eval_env, eval_episodes)
    final_interactions = min(update * env.num_envs, total_interactions)
    _save_checkpoint(
        results_dir / "final_policy.pt",
        model,
        optimizer,
        config,
        final_interactions,
        final_evaluation,
        env.actor_feature_names,
        metadata_extra,
    )
    if best_evaluation is None:
        best_evaluation = final_evaluation
        _save_checkpoint(
            results_dir / "best_policy.pt",
            model,
            optimizer,
            config,
            final_interactions,
            final_evaluation,
            env.actor_feature_names,
            metadata_extra,
        )

    metrics = {
        "experiment": config["name"],
        "hypothesis": config["hypothesis"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - wall_start,
        "interactions": final_interactions,
        "device": str(device),
        "num_envs": env.num_envs,
        "environment": asdict(env_config),
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "history": history,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\n[Saved] {results_dir / 'best_policy.pt'}")
    print(f"[Saved] {results_dir / 'final_policy.pt'}")
    print(f"[Saved] {results_dir / 'metrics.json'}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_ai_gp_swift_bc.py configs/ai_gp_NNN.yaml")
    run(sys.argv[1])


if __name__ == "__main__":
    main()
