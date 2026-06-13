import unittest

import torch

from ai_gp_rl.contract import (
    CORNER_BASE_OBS_DIM,
    MOTION_OBS_DIM,
    SWIFT_TEACHER_OBS_DIM,
    TEMPORAL_BASE_OBS_DIM,
)
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
        live_observation = env.live_actor_observation()
        self.assertEqual(live_observation.shape, (8, 18))
        self.assertTrue(torch.isfinite(live_observation).all())

    def test_gate_crossing_info_contains_time_series_state(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                num_envs=1,
                device="cpu",
                randomization=False,
            )
        )
        env.position[0] = torch.tensor((3.99, 0.0, 1.25))
        env.velocity[0] = torch.tensor((2.0, 0.0, 0.0))
        env.attitude[0] = 0.0
        env.angular_rate[0] = 0.0
        env.applied_action[0] = 0.0
        env.previous_action[0] = 0.0
        env.gate_index[0] = 0
        env.gates_passed[0] = 0

        _, _, _, _, info = env.step(torch.zeros((1, 4)))

        self.assertTrue(bool(info["passed_gate"][0]))
        self.assertEqual(int(info["active_gate_index"][0]), 0)
        self.assertEqual(int(info["gates_passed"][0]), 1)
        self.assertGreaterEqual(float(info["position"][0, 0]), 4.0)
        self.assertLess(abs(float(info["gate_lateral_offset"][0])), 1e-5)
        self.assertLess(abs(float(info["gate_vertical_offset"][0])), 0.01)
        self.assertTrue(torch.isfinite(info["velocity"]).all())
        self.assertTrue(torch.isfinite(info["attitude"]).all())

    def test_teacher_can_emit_temporal_student_observation(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                live_observation_mode="live_features_temporal",
                observation_history_length=4,
                num_envs=2,
                device="cpu",
                randomization=False,
                align_spawn_heading_to_gate=True,
            )
        )
        initial = env.live_actor_observation()
        self.assertEqual(initial.shape, (2, 4 * TEMPORAL_BASE_OBS_DIM))
        frames = initial.reshape(2, 4, TEMPORAL_BASE_OBS_DIM)
        self.assertTrue(torch.allclose(frames[:, 0], frames[:, -1]))

        env.step(torch.zeros((2, 4)))
        advanced = env.live_actor_observation().reshape(
            2, 4, TEMPORAL_BASE_OBS_DIM
        )
        self.assertTrue(torch.allclose(advanced[:, 0], frames[:, 1]))
        self.assertTrue(torch.isfinite(advanced).all())

    def test_temporal_actor_uses_flattened_history(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="live_features_temporal",
                observation_history_length=4,
                num_envs=2,
                device="cpu",
                randomization=False,
                align_spawn_heading_to_gate=True,
            )
        )
        observation, _ = env.reset()
        self.assertEqual(env.actor_observation_dim, 4 * TEMPORAL_BASE_OBS_DIM)
        self.assertEqual(
            observation.shape, (2, 4 * TEMPORAL_BASE_OBS_DIM + 14)
        )
        self.assertTrue(torch.isfinite(observation).all())

    def test_long_history_corner_recurrent_and_motion_contracts(self) -> None:
        cases = (
            ("live_features_temporal", 16, 16 * TEMPORAL_BASE_OBS_DIM),
            ("live_features_recurrent", 4, TEMPORAL_BASE_OBS_DIM),
            (
                "live_features_corners_temporal",
                4,
                4 * CORNER_BASE_OBS_DIM,
            ),
            (
                "live_features_corners_recurrent",
                4,
                CORNER_BASE_OBS_DIM,
            ),
            ("live_features_motion", 4, MOTION_OBS_DIM),
        )
        for mode, history_length, actor_dim in cases:
            with self.subTest(mode=mode):
                env = AIGPVectorEnv(
                    AIGPEnvConfig(
                        actor_observation_mode=mode,
                        observation_history_length=history_length,
                        num_envs=2,
                        device="cpu",
                        randomization=False,
                        align_spawn_heading_to_gate=True,
                    )
                )
                observation, _ = env.reset()
                self.assertEqual(env.actor_observation_dim, actor_dim)
                self.assertEqual(observation.shape, (2, actor_dim + 14))
                self.assertTrue(torch.isfinite(observation).all())


if __name__ == "__main__":
    unittest.main()
