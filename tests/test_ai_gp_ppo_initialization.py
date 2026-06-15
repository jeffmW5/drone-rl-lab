import tempfile
import unittest
from pathlib import Path

import torch

from ai_gp_rl.model import ActorCritic
from train_ai_gp import _load_initial_actor


class AIGPPPOInitializationTests(unittest.TestCase):
    def test_loads_only_matching_distilled_actor_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features = ("feature_0", "feature_1")
            source = ActorCritic(16, 4, 2, (8, 8))
            target = ActorCritic(16, 4, 2, (8, 8))
            original_critic = {
                key: value.clone() for key, value in target.critic.state_dict().items()
            }
            checkpoint = root / "student.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "metadata": {
                        "policy_role": "distilled_live_student",
                        "policy_architecture": "mlp",
                        "actor_features": list(features),
                    },
                },
                checkpoint,
            )

            _load_initial_actor(target, checkpoint, features)

            for key, value in source.actor_mean.state_dict().items():
                torch.testing.assert_close(target.actor_mean.state_dict()[key], value)
            for key, value in original_critic.items():
                torch.testing.assert_close(target.critic.state_dict()[key], value)

    def test_rejects_feature_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "student.pt"
            model = ActorCritic(16, 4, 2, (8, 8))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "policy_role": "distilled_live_student",
                        "policy_architecture": "mlp",
                        "actor_features": ["wrong"],
                    },
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "feature contract"):
                _load_initial_actor(model, checkpoint, ("feature_0", "feature_1"))

    def test_allows_matching_teacher_when_explicitly_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "teacher.pt"
            features = ("feature_0", "feature_1")
            source = ActorCritic(16, 4, 2, (8, 8))
            target = ActorCritic(16, 4, 2, (8, 8))
            torch.save(
                {
                    "global_step": 123,
                    "model_state_dict": source.state_dict(),
                    "metadata": {
                        "actor_observation_dim": 2,
                        "actor_features": list(features),
                    },
                },
                checkpoint,
            )

            metadata = _load_initial_actor(
                target,
                checkpoint,
                features,
                allow_teacher=True,
            )

            self.assertEqual(metadata["_source_policy_role"], "teacher")
            self.assertEqual(metadata["_source_global_step"], 123)
            for key, value in source.actor_mean.state_dict().items():
                torch.testing.assert_close(
                    target.actor_mean.state_dict()[key], value
                )


if __name__ == "__main__":
    unittest.main()
