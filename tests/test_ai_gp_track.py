import unittest

import torch

from ai_gp_rl.env import AIGPEnvConfig, AIGPVectorEnv
from ai_gp_rl.track import (
    AI_GP_GATE_SIZE_M,
    AI_GP_TRACK_GATES_NED,
    AI_GP_TRACK_NAME,
    ai_gp_track_altitude_offset_m,
    ai_gp_track_surrogate_positions,
    ned_position_to_surrogate,
    ned_vector_to_surrogate,
)


class AIGPTrackTests(unittest.TestCase):
    def test_ned_to_surrogate_contract_and_gate_order(self) -> None:
        self.assertEqual(
            ned_vector_to_surrogate((1.0, 2.0, 3.0)),
            (-1.0, 2.0, -3.0),
        )
        offset = ai_gp_track_altitude_offset_m()
        start = ned_position_to_surrogate(
            (0.0, 0.0, 0.0),
            altitude_offset_m=offset,
        )
        gates = ai_gp_track_surrogate_positions()

        self.assertEqual(len(gates), 6)
        self.assertAlmostEqual(start[2], offset)
        self.assertTrue(all(
            gates[index][0] < gates[index + 1][0]
            for index in range(len(gates) - 1)
        ))
        self.assertAlmostEqual(gates[0][1], AI_GP_TRACK_GATES_NED[0][1])
        self.assertAlmostEqual(gates[-1][2], 1.5)
        self.assertGreater(gates[0][2], gates[-1][2])

    def test_named_track_geometry_and_finite_signal(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                track_name=AI_GP_TRACK_NAME,
                start_position_ned_m=(0.0, 0.0, 0.0),
                position_observation_scale_m=(180.0, 20.0, 35.0),
                num_envs=4,
                device="cpu",
                randomization=False,
                align_spawn_heading_to_gate=True,
                near_gate_spawn_ratio_start=0.0,
                near_gate_spawn_ratio_end=0.0,
                max_altitude_m=35.0,
                max_lateral_m=15.0,
                max_forward_m=175.0,
            )
        )

        observation, reward, _, _, _ = env.step(torch.zeros((4, 4)))
        corners = env._active_gate_corners_world()
        top_edge = torch.linalg.vector_norm(corners[:, 1] - corners[:, 0], dim=1)
        right_edge = torch.linalg.vector_norm(corners[:, 2] - corners[:, 1], dim=1)

        self.assertEqual(env.gate_count, 6)
        self.assertTrue(torch.allclose(
            top_edge,
            torch.full_like(top_edge, AI_GP_GATE_SIZE_M),
        ))
        self.assertTrue(torch.allclose(
            right_edge,
            torch.full_like(right_edge, AI_GP_GATE_SIZE_M),
        ))
        self.assertTrue(torch.isfinite(observation).all())
        self.assertTrue(torch.isfinite(reward).all())

    def test_inside_aperture_crossing_passes_gate(self) -> None:
        env = self._crossing_env()
        env.position[0] = torch.tensor((3.99, 0.0, 1.25))
        env.velocity[0] = torch.tensor((2.0, 100.0, 0.0))

        _, reward, terminated, _, info = env.step(torch.zeros((1, 4)))

        self.assertGreater(float(info["position"][0, 1]), 0.8)
        self.assertTrue(bool(info["passed_gate"][0]))
        self.assertFalse(bool(info["missed_gate"][0]))
        self.assertLess(abs(float(info["gate_lateral_offset"][0])), 0.8)
        self.assertFalse(bool(terminated[0]))
        self.assertEqual(float(reward[0]), 0.0)

    def test_outside_aperture_crossing_terminates_as_miss(self) -> None:
        env = self._crossing_env()
        env.position[0] = torch.tensor((3.99, 1.0, 1.25))
        env.velocity[0] = torch.tensor((2.0, 20.0, 0.0))

        _, reward, terminated, _, info = env.step(torch.zeros((1, 4)))

        self.assertFalse(bool(info["passed_gate"][0]))
        self.assertTrue(bool(info["missed_gate"][0]))
        self.assertTrue(bool(terminated[0]))
        self.assertAlmostEqual(float(reward[0]), -50.0, places=5)
        self.assertGreater(abs(float(info["gate_lateral_offset"][0])), 0.8)

    @staticmethod
    def _crossing_env() -> AIGPVectorEnv:
        return AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                num_envs=1,
                device="cpu",
                randomization=False,
                near_gate_spawn_ratio_start=0.0,
                near_gate_spawn_ratio_end=0.0,
                progress_reward=0.0,
                gate_reward=0.0,
                finish_reward=0.0,
                forward_speed_reward=0.0,
                visibility_reward=0.0,
                camera_alignment_reward=0.0,
                alive_reward=0.0,
                angular_rate_penalty=0.0,
                action_delta_penalty=0.0,
                vertical_speed_penalty=0.0,
                altitude_penalty=0.0,
                collision_penalty=0.0,
                out_of_bounds_penalty=0.0,
                missed_gate_penalty=50.0,
                terminate_on_missed_gate=True,
            )
        )
