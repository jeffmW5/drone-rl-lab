import unittest
from math import atanh

import torch

from ai_gp_rl.contract import (
    CORNER_BASE_OBS_DIM,
    MOTION_OBS_DIM,
    STRUCTURED_TEACHER_OBS_DIM,
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

    def test_measured_dynamics_maps_real_thrust_and_launch_pitch(self) -> None:
        env = self._measured_env(command_latency_s=0.0)
        initial_x = float(env.position[0, 0])

        _, _, _, _, info = env.step(torch.zeros((1, 4)))

        self.assertAlmostEqual(float(info["thrust_command"][0]), 0.295, places=5)
        self.assertGreater(float(info["velocity"][0, 0]), 0.05)
        self.assertGreater(float(info["position"][0, 0]), initial_x)

    def test_measured_dynamics_delays_and_amplifies_rate_command(self) -> None:
        env = self._measured_env(command_latency_s=0.04)
        raw_action = torch.tensor(((0.0, atanh(0.5), 0.0, 0.0),))

        for _ in range(2):
            _, _, _, _, info = env.step(raw_action)
            self.assertAlmostEqual(
                float(info["applied_action"][0, 1]), 0.0, places=5
            )

        _, _, _, _, info = env.step(raw_action)
        self.assertAlmostEqual(
            float(info["applied_action"][0, 1]), 0.5, places=5
        )
        self.assertAlmostEqual(
            float(info["angular_rate"][0, 0]),
            0.5 * 0.3 * 2.65,
            places=4,
        )

    def test_structured_teacher_includes_active_and_next_gate_state(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="structured_teacher_v2",
                num_envs=2,
                device="cpu",
                randomization=False,
                near_gate_spawn_ratio_start=0.0,
                near_gate_spawn_ratio_end=0.0,
            )
        )

        observation, _ = env.reset()

        self.assertEqual(env.actor_observation_dim, STRUCTURED_TEACHER_OBS_DIM)
        self.assertEqual(
            observation.shape,
            (2, STRUCTURED_TEACHER_OBS_DIM + 14),
        )
        self.assertTrue(torch.isfinite(observation).all())
        actor = observation[:, :STRUCTURED_TEACHER_OBS_DIM]
        self.assertFalse(torch.allclose(actor[:, 0:3], actor[:, 6:9]))

    def test_near_gate_spawn_can_replay_high_speed_transition_state(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="structured_teacher_v2",
                num_envs=16,
                device="cpu",
                randomization=False,
                near_gate_spawn_ratio_start=1.0,
                near_gate_spawn_ratio_end=1.0,
                near_gate_forward_speed_mps=(14.0, 14.0),
                near_gate_lateral_speed_mps=(0.0, 0.0),
                near_gate_vertical_speed_mps=(-3.5, -3.5),
            )
        )
        env.reset()
        normal = env.gate_normals[env.gate_index]
        vertical = env.gate_vertical[env.gate_index]

        forward_speed = (env.velocity * normal).sum(1)
        vertical_speed = (env.velocity * vertical).sum(1)

        torch.testing.assert_close(
            forward_speed, torch.full_like(forward_speed, 14.0)
        )
        torch.testing.assert_close(
            vertical_speed, torch.full_like(vertical_speed, -3.5)
        )

    def test_near_gate_spawn_can_target_recorded_failure_state(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="structured_teacher_v2",
                num_envs=8,
                device="cpu",
                randomization=False,
                align_spawn_heading_to_gate=True,
                near_gate_spawn_ratio_start=1.0,
                near_gate_spawn_ratio_end=1.0,
                near_gate_indices=(1,),
                near_gate_distance_m=(23.0, 23.0),
                near_gate_lateral_offset_m=(0.0, 0.0),
                near_gate_vertical_offset_m=(5.0, 5.0),
                near_gate_forward_speed_mps=(14.0, 14.0),
                near_gate_lateral_speed_mps=(1.5, 1.5),
                near_gate_vertical_speed_mps=(-3.5, -3.5),
                near_gate_roll_rad=(-0.02, -0.02),
                near_gate_pitch_rad=(-0.79, -0.79),
                near_gate_yaw_offset_rad=(0.20, 0.20),
                near_gate_previous_action=(-0.04, -0.10, -0.33, 0.12),
            )
        )
        env.reset()
        gate_offset = env.position - env._active_gate_position()
        vertical = env.gate_vertical[env.gate_index]
        vertical_offset = (gate_offset * vertical).sum(1)

        self.assertTrue(torch.all(env.gate_index == 1))
        self.assertTrue(torch.all(env.gates_passed == 1))
        torch.testing.assert_close(
            vertical_offset, torch.full_like(vertical_offset, 5.0)
        )
        torch.testing.assert_close(
            env.attitude[:, 0], torch.full_like(env.attitude[:, 0], -0.02)
        )
        torch.testing.assert_close(
            env.attitude[:, 1], torch.full_like(env.attitude[:, 1], -0.79)
        )
        expected_action = torch.tensor((-0.04, -0.10, -0.33, 0.12))
        torch.testing.assert_close(
            env.previous_action,
            expected_action.expand_as(env.previous_action),
        )
        self.assertTrue(torch.all(env.angular_rate[:, 1] < -0.15))

    @staticmethod
    def _measured_env(*, command_latency_s: float) -> AIGPVectorEnv:
        return AIGPVectorEnv(
            AIGPEnvConfig(
                actor_observation_mode="swift_teacher",
                num_envs=1,
                device="cpu",
                randomization=False,
                start_position_m=(0.0, 0.0, 10.0),
                start_position_noise_m=(0.0, 0.0, 0.0),
                start_velocity_noise_mps=0.0,
                near_gate_spawn_ratio_start=0.0,
                near_gate_spawn_ratio_end=0.0,
                dynamics_model="measured_ai_gp_v1",
                max_roll_rate_radps=0.3,
                max_pitch_rate_radps=0.2,
                max_yaw_rate_radps=0.15,
                thrust_command_center=0.295,
                thrust_command_span_up=0.105,
                thrust_command_span_down=0.095,
                thrust_acceleration_bias_mps2=-5.61718,
                thrust_acceleration_gain_mps2=57.29141,
                base_pitch_offset_rad=0.31069,
                linear_drag_xyz=(0.14170, 0.14170, 0.39055),
                quadratic_drag_xyz=(0.02230, 0.02230, 0.01175),
                rate_response_gain=(2.65, 2.51, 2.58),
                command_latency_s=command_latency_s,
                command_latency_s_range=(
                    command_latency_s,
                    command_latency_s,
                ),
                rate_time_constant_s=0.005,
                rate_time_constant_s_range=(0.005, 0.005),
                action_response=1.0,
                action_response_range=(1.0, 1.0),
                wind_accel_mps2=0.0,
                max_altitude_m=100.0,
                max_forward_m=100.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
