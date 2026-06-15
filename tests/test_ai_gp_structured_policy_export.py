import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai_gp_rl.contract import ACTION_NAMES, STRUCTURED_TEACHER_FEATURE_NAMES
from ai_gp_rl.model import ActorCritic
from scripts.export_ai_gp_structured_policy import export_policy


class StructuredPolicyExportTests(unittest.TestCase):
    def _write_checkpoint(self, root: Path) -> Path:
        actor_observation_dim = len(STRUCTURED_TEACHER_FEATURE_NAMES)
        model = ActorCritic(
            observation_dim=actor_observation_dim + 14,
            action_dim=len(ACTION_NAMES),
            actor_observation_dim=actor_observation_dim,
            hidden_sizes=(8, 8),
        )
        path = root / "best_policy.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "global_step": 1234,
                "metadata": {
                    "schema_version": 1,
                    "observation_dim": actor_observation_dim + 14,
                    "actor_observation_dim": actor_observation_dim,
                    "action_dim": len(ACTION_NAMES),
                    "hidden_sizes": [8, 8],
                    "actor_features": list(STRUCTURED_TEACHER_FEATURE_NAMES),
                    "action_names": list(ACTION_NAMES),
                    "action_semantics": (
                        "tanh-normalized [physical thrust offset, roll command, "
                        "pitch command, yaw command]; negative pitch is forward"
                    ),
                    "action_calibration": {
                        "contract": "measured_ai_gp_v1",
                        "thrust_command_center": 0.295,
                        "thrust_span_up": 0.105,
                        "thrust_span_down": 0.095,
                        "max_roll_rate_radps": 0.3,
                        "max_pitch_rate_radps": 0.2,
                        "max_yaw_rate_radps": 0.15,
                    },
                    "training_method": "swift_geometric_full_course_bc",
                    "expert": "geometric_gate_teacher_v1",
                    "policy_architecture": "mlp",
                },
            },
            path,
        )
        return path

    def _write_report(
        self,
        root: Path,
        checkpoint: Path,
        *,
        randomized: bool = False,
        **overrides: float,
    ) -> Path:
        summary = {
            "gate0_passage_rate": 1.0,
            "mean_gates": 6.0,
            "success_rate": 1.0,
            "collision_rate": 0.0,
            "out_of_bounds_rate": 0.0,
            "missed_gate_rate": 0.0,
            "vertical_runaway_rate": 0.0,
            "gate_crossing_count": 3072,
            "gate_crossing_min_margin_m": 0.16,
        }
        summary.update(overrides)
        path = root / ("random.json" if randomized else "nominal.json")
        path.write_text(
            json.dumps(
                {
                    "experiment": "ai_gp_030_swift_full_course_bc_50m",
                    "checkpoint": checkpoint.name,
                    "checkpoint_global_step": 1234,
                    "environment_randomization": randomized,
                    "evaluation_seed": 10131,
                    "episodes": 512,
                    "deterministic_summary": summary,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_exports_structured_sim_policy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root)
            nominal = self._write_report(root, checkpoint)
            randomized = self._write_report(
                root,
                checkpoint,
                randomized=True,
                mean_gates=4.7,
                success_rate=0.56,
                missed_gate_rate=0.43,
            )
            output = root / "policy.json"

            export_policy(
                checkpoint,
                nominal,
                output,
                randomized_validation_report_paths=(randomized,),
            )

            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["policy_role"], "structured_state_sim_teacher")
            self.assertEqual(artifact["observation_contract"], "structured_teacher_v2")
            self.assertTrue(artifact["requires_structured_state"])
            self.assertTrue(artifact["not_live_vision_policy"])
            self.assertEqual(len(artifact["actor_features"]), 26)
            self.assertEqual(artifact["track"]["gate_count"], 6)
            self.assertEqual(len(artifact["layers"]), 3)
            self.assertEqual(len(artifact["test_vectors"]), 4)
            self.assertEqual(
                artifact["validation"]["randomized"][0]["summary"]["success_rate"],
                0.56,
            )

    def test_rejects_non_promoted_nominal_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._write_checkpoint(root)
            nominal = self._write_report(
                root,
                checkpoint,
                success_rate=0.5,
                mean_gates=4.0,
            )

            with self.assertRaisesRegex(ValueError, "nominal validation failed"):
                export_policy(checkpoint, nominal, root / "policy.json")


if __name__ == "__main__":
    unittest.main()
