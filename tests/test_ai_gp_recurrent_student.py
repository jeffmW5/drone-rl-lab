import unittest

import torch

from ai_gp_rl.model import RecurrentStudentPolicy, build_policy_from_metadata


class RecurrentStudentTests(unittest.TestCase):
    def test_state_reset_is_row_selective(self) -> None:
        model = RecurrentStudentPolicy(
            observation_dim=34,
            action_dim=4,
            actor_observation_dim=20,
            hidden_sizes=(16, 8),
        )
        observation = torch.ones((3, 20))
        hidden = model.initial_state(3, device="cpu")
        _, next_hidden = model.actor_step(observation, hidden)
        reset_mask = torch.tensor((False, True, False))
        reset_hidden = next_hidden * (~reset_mask).unsqueeze(1)

        self.assertGreater(float(reset_hidden[0].abs().sum()), 0.0)
        self.assertEqual(float(reset_hidden[1].abs().sum()), 0.0)
        self.assertGreater(float(reset_hidden[2].abs().sum()), 0.0)

    def test_factory_builds_recurrent_policy(self) -> None:
        model = build_policy_from_metadata(
            {
                "policy_architecture": "gru",
                "observation_dim": 34,
                "actor_observation_dim": 20,
                "action_dim": 4,
                "hidden_sizes": [16, 8],
            },
            device="cpu",
        )
        self.assertIsInstance(model, RecurrentStudentPolicy)


if __name__ == "__main__":
    unittest.main()
