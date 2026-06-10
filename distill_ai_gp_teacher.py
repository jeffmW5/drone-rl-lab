"""Distill a privileged AI-GP teacher into the deployable 18D live policy."""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ai_gp_rl.contract import ACTOR_FEATURE_NAMES
from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.model import ActorCritic
from train_ai_gp import _build_env_config, _save_checkpoint, evaluate, load_config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(config_path: Path, teacher_checkpoint_path: Path) -> None:
    config = _load_yaml(config_path)
    distillation = config["distillation"]
    student_section = config["ai_gp"]
    teacher_config_path = config_path.parent.parent / config["teacher_config"]
    teacher_config = load_config(teacher_config_path)
    teacher_section = copy.deepcopy(teacher_config["ai_gp"])

    device = torch.device(distillation.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("distillation requested CUDA but CUDA is unavailable")
    seed = int(distillation.get("seed", 43))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_envs = int(distillation.get("num_envs", 4096))
    teacher_section["device"] = str(device)
    teacher_section["num_envs"] = num_envs
    teacher_section.setdefault("env", {})
    teacher_section["env"]["actor_observation_mode"] = "swift_teacher"
    teacher_section["env"]["num_envs"] = num_envs
    teacher_section["env"]["device"] = str(device)
    teacher_section["env"]["randomization"] = bool(
        distillation.get("randomization", True)
    )
    if distillation.get("race_start_only", True):
        teacher_section["env"]["near_gate_spawn_ratio_start"] = 0.0
        teacher_section["env"]["near_gate_spawn_ratio_end"] = 0.0

    env = AIGPVectorEnv(_build_env_config(teacher_section))
    env.set_curriculum(1.0)
    teacher_checkpoint = torch.load(
        teacher_checkpoint_path, map_location=device, weights_only=False
    )
    teacher_metadata = teacher_checkpoint["metadata"]
    teacher = ActorCritic(
        observation_dim=int(teacher_metadata["observation_dim"]),
        action_dim=int(teacher_metadata["action_dim"]),
        actor_observation_dim=int(teacher_metadata["actor_observation_dim"]),
        hidden_sizes=tuple(teacher_metadata["hidden_sizes"]),
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()

    eval_config = _build_env_config(student_section, evaluation=True)
    eval_config.device = str(device)
    eval_env = AIGPVectorEnv(eval_config)
    eval_env.set_curriculum(1.0)
    hidden_sizes = tuple(distillation.get("hidden_sizes", [128, 128]))
    student = ActorCritic(
        observation_dim=eval_env.observation_dim,
        action_dim=eval_env.action_dim,
        actor_observation_dim=eval_env.actor_observation_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(
        student.actor_mean.parameters(),
        lr=float(distillation.get("learning_rate", 3e-4)),
    )

    root = Path(__file__).resolve().parent
    results_dir = root / "results" / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    total_interactions = int(distillation.get("total_interactions", 10_000_000))
    total_updates = max((total_interactions + num_envs - 1) // num_envs, 1)
    student_rollout_start = int(
        distillation.get("student_rollout_start_interactions", total_interactions + 1)
    )
    student_rollout_end = int(
        distillation.get("student_rollout_end_interactions", total_interactions + 1)
    )
    max_student_rollout_ratio = float(
        distillation.get("max_student_rollout_ratio", 0.0)
    )
    eval_interval = int(distillation.get("eval_interval_updates", 250))
    eval_episodes = int(distillation.get("eval_episodes", 256))
    observation, _ = env.reset()
    history: list[dict[str, float]] = []
    best_evaluation: dict[str, float] | None = None
    best_score = (-1.0, -1.0, float("-inf"))
    wall_start = time.time()

    metadata_extra = {
        "policy_role": "distilled_live_student",
        "deployable_live_contract": True,
        "source_teacher_checkpoint": teacher_checkpoint_path.name,
        "source_teacher_global_step": int(teacher_checkpoint.get("global_step", 0)),
    }

    for update in range(1, total_updates + 1):
        with torch.no_grad():
            teacher_raw_action = teacher.actor_mean(
                observation[:, : teacher.actor_observation_dim]
            )
            target_action = torch.tanh(teacher_raw_action)
            live_observation = env.live_actor_observation()

        predicted_action = torch.tanh(student.actor_mean(live_observation))
        loss = torch.nn.functional.mse_loss(predicted_action, target_action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            interactions = update * num_envs
            rollout_progress = (
                (interactions - student_rollout_start)
                / max(student_rollout_end - student_rollout_start, 1)
            )
            student_rollout_ratio = max(
                0.0, min(rollout_progress, 1.0)
            ) * max_student_rollout_ratio
            use_student = (
                torch.rand(
                    (num_envs, 1),
                    device=device,
                    generator=env.generator,
                )
                < student_rollout_ratio
            )
            student_raw_action = student.actor_mean(live_observation)
            rollout_action = torch.where(
                use_student, student_raw_action, teacher_raw_action
            )
            observation, _, _, _, _ = env.step(rollout_action)

        if update == 1 or update % 25 == 0:
            elapsed = time.time() - wall_start
            record = {
                "update": update,
                "interactions": interactions,
                "imitation_loss": float(loss),
                "interactions_per_second": interactions / max(elapsed, 1e-6),
                "student_rollout_ratio": student_rollout_ratio,
            }
            history.append(record)
            print(
                f"update {update:>5}/{total_updates} | "
                f"interactions {interactions:>11,} | "
                f"loss {float(loss):.6f} | "
                f"student {student_rollout_ratio:>5.1%} | "
                f"IPS {record['interactions_per_second']:,.0f}"
            )

        if update % eval_interval == 0 or update == total_updates:
            evaluation = evaluate(student, eval_env, eval_episodes)
            print(
                "  [StudentEval] "
                f"success={evaluation['success_rate']:.1%} "
                f"gates={evaluation['mean_gates']:.2f} "
                f"collision={evaluation['collision_rate']:.1%}"
            )
            score = (
                evaluation["success_rate"],
                evaluation["mean_gates"],
                evaluation["mean_distance_reduction_m"],
            )
            if score > best_score:
                best_score = score
                best_evaluation = evaluation
                _save_checkpoint(
                    results_dir / "best_student.pt",
                    student,
                    optimizer,
                    config,
                    interactions,
                    evaluation,
                    ACTOR_FEATURE_NAMES,
                    metadata_extra,
                )

    final_evaluation = evaluate(student, eval_env, eval_episodes)
    _save_checkpoint(
        results_dir / "final_student.pt",
        student,
        optimizer,
        config,
        min(total_updates * num_envs, total_interactions),
        final_evaluation,
        ACTOR_FEATURE_NAMES,
        metadata_extra,
    )
    metrics = {
        "experiment": config["name"],
        "hypothesis": config["hypothesis"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - wall_start,
        "interactions": min(total_updates * num_envs, total_interactions),
        "teacher_checkpoint": str(teacher_checkpoint_path),
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "history": history,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python distill_ai_gp_teacher.py "
            "configs/ai_gp_003_distilled_live_student.yaml "
            "results/<teacher>/best_policy.pt"
        )
    run(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
