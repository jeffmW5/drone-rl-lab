import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai_gp_rl.contract import (
    ACTOR_FEATURE_NAMES,
    TEMPORAL_BASE_FEATURE_NAMES,
    temporal_feature_names,
)
from ai_gp_rl.model import ActorCritic, RecurrentStudentPolicy
from scripts.export_ai_gp_live_policy import export_policy


class LivePolicyExportTests(unittest.TestCase):
    def _write_checkpoint(self, root: Path, *, temporal: bool = False) -> Path:
        actor_features = (
            temporal_feature_names(4) if temporal else ACTOR_FEATURE_NAMES
        )
        actor_observation_dim = len(actor_features)
        model = ActorCritic(
            observation_dim=actor_observation_dim + 14,
            action_dim=4,
            actor_observation_dim=actor_observation_dim,
            hidden_sizes=(8, 8),
        )
        path = root / "student.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "global_step": 100,
                "metadata": {
                    "policy_role": "distilled_live_student",
                    "observation_dim": actor_observation_dim + 14,
                    "actor_observation_dim": actor_observation_dim,
                    "action_dim": 4,
                    "hidden_sizes": [8, 8],
                    "actor_features": list(actor_features),
                    "observation_contract": (
                        "temporal_live_v1" if temporal else "live_features_v1"
                    ),
                    "base_actor_features": list(
                        TEMPORAL_BASE_FEATURE_NAMES
                        if temporal
                        else ACTOR_FEATURE_NAMES
                    ),
                    "history_length": 4 if temporal else 1,
                    "source_teacher_checkpoint": "teacher.pt",
                },
            },
            path,
        )
        return path

    def _write_report(self, root: Path, checkpoint: Path, **overrides: float) -> Path:
        summary = {
            "gate0_passage_rate": 1.0,
            "mean_gates": 4.0,
            "collision_rate": 0.0,
            "out_of_bounds_rate": 0.0,
            "vertical_runaway_rate": 0.0,
        }
        summary.update(overrides)
        path = root / "telemetry.json"
        path.write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint.name,
                    "deterministic_summary": summary,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_exports_only_after_deterministic_telemetry_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root)
            report = self._write_report(root, checkpoint)
            output = root / "student.json"

            export_policy(checkpoint, report, output)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["validation_status"],
                "surrogate_passed_pending_windows_simulator",
            )
            self.assertEqual(len(artifact["test_vectors"]), 4)

    def test_exports_temporal_live_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root, temporal=True)
            report = self._write_report(root, checkpoint)
            output = root / "student.json"

            export_policy(checkpoint, report, output)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["observation_contract"], "temporal_live_v1")
            self.assertEqual(artifact["history_length"], 4)
            self.assertEqual(len(artifact["actor_features"]), 80)

    def test_rejects_failed_deterministic_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root)
            report = self._write_report(
                root,
                checkpoint,
                collision_rate=0.45,
                vertical_runaway_rate=0.40,
            )

            with self.assertRaisesRegex(ValueError, "failed deterministic telemetry"):
                export_policy(checkpoint, report, root / "student.json")

    def test_exports_recurrent_policy_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actor_observation_dim = len(TEMPORAL_BASE_FEATURE_NAMES)
            model = RecurrentStudentPolicy(
                observation_dim=actor_observation_dim + 14,
                action_dim=4,
                actor_observation_dim=actor_observation_dim,
                hidden_sizes=(8, 8),
            )
            checkpoint = root / "recurrent.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "global_step": 100,
                    "metadata": {
                        "policy_role": "distilled_live_student",
                        "policy_architecture": "gru",
                        "observation_dim": actor_observation_dim + 14,
                        "actor_observation_dim": actor_observation_dim,
                        "action_dim": 4,
                        "hidden_sizes": [8, 8],
                        "actor_features": list(TEMPORAL_BASE_FEATURE_NAMES),
                        "observation_contract": "recurrent_live_v1",
                        "base_actor_features": list(
                            TEMPORAL_BASE_FEATURE_NAMES
                        ),
                        "history_length": 1,
                        "source_teacher_checkpoint": "teacher.pt",
                    },
                },
                checkpoint,
            )
            report = self._write_report(root, checkpoint)
            output = root / "recurrent.json"

            export_policy(checkpoint, report, output)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema_version"], 2)
            self.assertEqual(artifact["policy_architecture"], "gru")
            self.assertEqual(len(artifact["test_sequence"]), 6)

    def test_exports_action_governor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root, temporal=True)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            payload["metadata"]["action_governor"] = {
                "slew_limits": [0.15, 0.25, 0.25, 0.35],
                "upward_brake_start_mps": 1.0,
                "upward_brake_gain": 0.1,
            }
            torch.save(payload, checkpoint)
            report = self._write_report(root, checkpoint)
            output = root / "governed.json"

            export_policy(checkpoint, report, output)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["action_governor"]["slew_limits"],
                [0.15, 0.25, 0.25, 0.35],
            )


if __name__ == "__main__":
    unittest.main()
