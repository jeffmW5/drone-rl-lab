"""Distill a privileged AI-GP teacher into a deployable live policy."""

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

from ai_gp_rl.action_governor import ActionGovernorConfig, govern_action
from ai_gp_rl.env import AIGPVectorEnv
from ai_gp_rl.model import ActorCritic, RecurrentStudentPolicy
from train_ai_gp import _build_env_config, _save_checkpoint, load_config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _student_raw_action(
    student: ActorCritic | RecurrentStudentPolicy,
    live_observation: torch.Tensor,
    hidden_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(student, RecurrentStudentPolicy):
        if hidden_state is None:
            raise ValueError("recurrent student requires hidden state")
        return student.forward_step(live_observation, hidden_state)
    return student.actor_mean(live_observation), None


@torch.no_grad()
def _evaluate_student(
    student: ActorCritic | RecurrentStudentPolicy,
    env: AIGPVectorEnv,
    episode_count: int,
    governor_config: ActionGovernorConfig | None,
) -> dict[str, float]:
    observation, _ = env.reset()
    hidden_state = (
        student.initial_state(env.num_envs, device=env.device)
        if isinstance(student, RecurrentStudentPolicy)
        else None
    )
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
    intervention_count = 0
    sample_count = 0
    completed = 0
    while completed < episode_count:
        live_observation = observation[:, : env.actor_observation_dim]
        raw_action, next_hidden = _student_raw_action(
            student, live_observation, hidden_state
        )
        if governor_config is not None:
            requested_action = torch.tanh(raw_action)
            executed_action, intervention = govern_action(
                requested_action,
                env.previous_action,
                env.live_observation_history[:, -1],
                governor_config,
            )
            raw_action = torch.atanh(executed_action.clamp(-0.999, 0.999))
            intervention_count += int((intervention > 1e-5).sum())
            sample_count += env.num_envs
        observation, _, _, _, info = env.step(raw_action)
        if next_hidden is not None:
            hidden_state = next_hidden * (~info["done"]).unsqueeze(1)

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
        accepted = done_ids[: episode_count - completed]
        returns.append(info["episode_return"][accepted].detach().cpu())
        gates.append(info["gates_passed"][accepted].float().detach().cpu())
        successes.append(info["success"][accepted].float().detach().cpu())
        collisions.append(info["collision"][accepted].float().detach().cpu())
        out_of_bounds.append(
            info["out_of_bounds"][accepted].float().detach().cpu()
        )
        vertical_runaways.append(
            (
                (max_altitude[accepted] >= 0.95 * env.config.max_altitude_m)
                | (
                    longest_upward_run[accepted].float() * env.config.dt
                    >= 1.0
                )
            )
            .float()
            .detach()
            .cpu()
        )
        distance_reduction.append(
            info["distance_reduction"][accepted].detach().cpu()
        )
        completed += accepted.numel()
        max_altitude[done_ids] = env.position[done_ids, 2]
        upward_run[done_ids] = 0
        longest_upward_run[done_ids] = 0
    return {
        "mean_return": float(torch.cat(returns).mean()),
        "mean_gates": float(torch.cat(gates).mean()),
        "success_rate": float(torch.cat(successes).mean()),
        "collision_rate": float(torch.cat(collisions).mean()),
        "out_of_bounds_rate": float(torch.cat(out_of_bounds).mean()),
        "vertical_runaway_rate": float(torch.cat(vertical_runaways).mean()),
        "mean_distance_reduction_m": float(torch.cat(distance_reduction).mean()),
        "governor_intervention_rate": (
            intervention_count / sample_count if sample_count else 0.0
        ),
        "episodes": completed,
    }


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
    student_env_section = student_section.get("env", {})
    teacher_section["env"]["live_observation_mode"] = student_env_section.get(
        "actor_observation_mode", "live_features"
    )
    teacher_section["env"]["observation_history_length"] = int(
        student_env_section.get("observation_history_length", 4)
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
    student_architecture = str(distillation.get("student_architecture", "mlp"))
    if student_architecture == "gru":
        student: ActorCritic | RecurrentStudentPolicy = RecurrentStudentPolicy(
            observation_dim=eval_env.observation_dim,
            action_dim=eval_env.action_dim,
            actor_observation_dim=eval_env.actor_observation_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        optimized_parameters = student.parameters()
    elif student_architecture == "mlp":
        student = ActorCritic(
            observation_dim=eval_env.observation_dim,
            action_dim=eval_env.action_dim,
            actor_observation_dim=eval_env.actor_observation_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        optimized_parameters = student.actor_mean.parameters()
    else:
        raise ValueError("student_architecture must be 'mlp' or 'gru'")
    optimizer = torch.optim.Adam(
        optimized_parameters,
        lr=float(distillation.get("learning_rate", 3e-4)),
    )
    action_loss_weights = torch.tensor(
        distillation.get("action_loss_weights", [1.0, 1.0, 1.0, 1.0]),
        device=device,
        dtype=torch.float32,
    )
    if action_loss_weights.shape != (env.action_dim,) or bool(
        (action_loss_weights <= 0.0).any()
    ):
        raise ValueError("action_loss_weights must contain four positive values")

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
    recurrent_sequence_length = int(
        distillation.get("recurrent_sequence_length", 32)
    )
    randomization_curriculum_interactions = int(
        distillation.get("randomization_curriculum_interactions", 0)
    )
    governor_section = distillation.get("action_governor")
    governor_config = (
        ActionGovernorConfig(
            slew_limits=tuple(
                float(value)
                for value in governor_section.get(
                    "slew_limits", [0.08, 0.12, 0.12, 0.15]
                )
            ),
            upward_brake_start_mps=float(
                governor_section.get("upward_brake_start_mps", 1.5)
            ),
            upward_brake_gain=float(
                governor_section.get("upward_brake_gain", 0.30)
            ),
        )
        if governor_section
        else None
    )
    eval_interval = int(distillation.get("eval_interval_updates", 250))
    eval_episodes = int(distillation.get("eval_episodes", 256))
    observation, _ = env.reset()
    hidden_state = (
        student.initial_state(num_envs, device=device)
        if isinstance(student, RecurrentStudentPolicy)
        else None
    )
    history: list[dict[str, float]] = []
    best_evaluation: dict[str, float] | None = None
    best_score = (float("-inf"), -1.0, -1.0)
    wall_start = time.time()

    metadata_extra = {
        "policy_role": "distilled_live_student",
        "deployable_live_contract": True,
        "source_teacher_checkpoint": teacher_checkpoint_path.name,
        "source_teacher_global_step": int(teacher_checkpoint.get("global_step", 0)),
        "observation_contract": eval_env.live_observation_contract,
        "base_actor_features": list(eval_env.live_base_feature_names),
        "history_length": (
            eval_env.config.observation_history_length
            if eval_env.live_observation_mode
            in (
                "live_features_temporal",
                "live_features_corners_temporal",
            )
            else 1
        ),
        "action_loss_weights": action_loss_weights.detach().cpu().tolist(),
        "policy_architecture": student_architecture,
        "recurrent_sequence_length": (
            recurrent_sequence_length
            if isinstance(student, RecurrentStudentPolicy)
            else None
        ),
        "action_governor": governor_section,
    }

    optimizer.zero_grad(set_to_none=True)
    recurrent_loss: torch.Tensor | None = None
    recurrent_loss_steps = 0
    for update in range(1, total_updates + 1):
        interactions = update * num_envs
        if randomization_curriculum_interactions > 0:
            env.set_curriculum(
                min(
                    interactions
                    / max(randomization_curriculum_interactions, 1),
                    1.0,
                )
            )
        with torch.no_grad():
            teacher_raw_action = teacher.actor_mean(
                observation[:, : teacher.actor_observation_dim]
            )
            target_action = torch.tanh(teacher_raw_action)
            # Recurrent TBPTT retains inputs until the sequence backward pass.
            # The environment updates its observation buffers in place.
            live_observation = env.live_actor_observation().clone()

        predicted_raw_action, next_hidden_state = _student_raw_action(
            student, live_observation, hidden_state
        )
        predicted_action = torch.tanh(predicted_raw_action)
        if governor_config is not None:
            target_action, _ = govern_action(
                target_action,
                env.previous_action,
                env.live_observation_history[:, -1],
                governor_config,
            )
        squared_action_error = (predicted_action - target_action).square()
        loss = (
            squared_action_error * action_loss_weights
        ).mean() / action_loss_weights.mean()
        if isinstance(student, RecurrentStudentPolicy):
            recurrent_loss = (
                loss if recurrent_loss is None else recurrent_loss + loss
            )
            recurrent_loss_steps += 1
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
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
            requested_action = torch.where(
                use_student,
                predicted_action.detach(),
                torch.tanh(teacher_raw_action),
            )
            if governor_config is not None:
                requested_action, _ = govern_action(
                    requested_action,
                    env.previous_action,
                    env.live_observation_history[:, -1],
                    governor_config,
                )
            rollout_action = torch.atanh(requested_action.clamp(-0.999, 0.999))
            observation, _, _, _, _ = env.step(rollout_action)
            done = env.steps == 0
        if next_hidden_state is not None:
            hidden_state = next_hidden_state * (~done).unsqueeze(1)
            if (
                update % recurrent_sequence_length == 0
                or update == total_updates
            ):
                if recurrent_loss is None:
                    raise RuntimeError("missing recurrent loss")
                (recurrent_loss / recurrent_loss_steps).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                hidden_state = hidden_state.detach()
                recurrent_loss = None
                recurrent_loss_steps = 0

        if update == 1 or update % 25 == 0:
            elapsed = time.time() - wall_start
            record = {
                "update": update,
                "interactions": interactions,
                "imitation_loss": float(loss),
                "collective_mae": float(
                    (predicted_action[:, 0] - target_action[:, 0]).abs().mean()
                ),
                "roll_mae": float(
                    (predicted_action[:, 1] - target_action[:, 1]).abs().mean()
                ),
                "interactions_per_second": interactions / max(elapsed, 1e-6),
                "student_rollout_ratio": student_rollout_ratio,
            }
            history.append(record)
            print(
                f"update {update:>5}/{total_updates} | "
                f"interactions {interactions:>11,} | "
                f"loss {float(loss):.6f} | "
                f"collective {record['collective_mae']:.4f} | "
                f"roll {record['roll_mae']:.4f} | "
                f"student {student_rollout_ratio:>5.1%} | "
                f"IPS {record['interactions_per_second']:,.0f}"
            )

        if update % eval_interval == 0 or update == total_updates:
            evaluation = _evaluate_student(
                student, eval_env, eval_episodes, governor_config
            )
            print(
                "  [StudentEval] "
                f"success={evaluation['success_rate']:.1%} "
                f"gates={evaluation['mean_gates']:.2f} "
                f"collision={evaluation['collision_rate']:.1%} "
                f"oob={evaluation['out_of_bounds_rate']:.1%} "
                f"vertical={evaluation['vertical_runaway_rate']:.1%}"
            )
            bounded_progress_score = (
                evaluation["mean_gates"]
                * (1.0 - evaluation["collision_rate"])
                * (1.0 - evaluation["out_of_bounds_rate"])
                * (1.0 - evaluation["vertical_runaway_rate"])
            )
            score = (
                bounded_progress_score + evaluation["success_rate"],
                evaluation["success_rate"],
                evaluation["mean_gates"],
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
                    eval_env.actor_feature_names,
                    metadata_extra,
                )

    final_evaluation = _evaluate_student(
        student, eval_env, eval_episodes, governor_config
    )
    _save_checkpoint(
        results_dir / "final_student.pt",
        student,
        optimizer,
        config,
        min(total_updates * num_envs, total_interactions),
        final_evaluation,
        eval_env.actor_feature_names,
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
