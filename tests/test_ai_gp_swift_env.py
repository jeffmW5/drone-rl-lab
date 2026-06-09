import unittest

import torch

from ai_gp_rl.contract import SWIFT_TEACHER_OBS_DIM
from ai_gp_rl.env import AIGPEnvConfig, AIGPVectorEnv


class SwiftTeacherEnvTests(unittest.TestCase):
    def make_env(self) -> AIGPVectorEnv:
        return AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                num_envs=8,
                device="cpu",
                randomization=False,
                align_spawn_heading_to_gate=True,
            )
        )

    def test_swift_observation_shape_and_finite_values(self) -> None:
        env = self.make_env()
        observation, _ = env.reset()
        self.assertEqual(env.actor_observation_dim, SWIFT_TEACHER_OBS_DIM)
        self.assertEqual(observation.shape, (8, SWIFT_TEACHER_OBS_DIM + 14))
        self.assertTrue(torch.isfinite(observation).all())

    def test_gate_corners_are_ordered_and_sized(self) -> None:
        env = self.make_env()
        corners = env._active_gate_corners_world()
        top_edge = torch.linalg.vector_norm(corners[:, 1] - corners[:, 0], dim=1)
        right_edge = torch.linalg.vector_norm(corners[:, 2] - corners[:, 1], dim=1)
        self.assertTrue(torch.allclose(top_edge, torch.full_like(top_edge, 1.6)))
        self.assertTrue(torch.allclose(right_edge, torch.full_like(right_edge, 1.6)))

    def test_step_produces_finite_training_signal(self) -> None:
        env = self.make_env()
        observation, reward, _, _, _ = env.step(torch.zeros((8, 4)))
        self.assertTrue(torch.isfinite(observation).all())
        self.assertTrue(torch.isfinite(reward).all())


if __name__ == "__main__":
    unittest.main()
