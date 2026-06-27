import unittest

import torch

from ai_gp_rl.env import AIGPEnvConfig, AIGPVectorEnv
from ai_gp_rl.swift_expert import (
    geometric_gate_teacher_action,
    raw_geometric_gate_teacher_action,
)
from ai_gp_rl.track import AI_GP_TRACK_NAME


class SwiftExpertTests(unittest.TestCase):
    def test_geometric_teacher_actions_are_finite_and_normalized(self) -> None:
        env = AIGPVectorEnv(self._env_config(num_envs=8))

        action = geometric_gate_teacher_action(env)

        self.assertEqual(action.shape, (8, 4))
        self.assertTrue(torch.isfinite(action).all())
        self.assertLessEqual(float(action.abs().max()), 0.98 + 1e-6)

    def test_geometric_teacher_completes_measured_course_from_race_start(self) -> None:
        env = AIGPVectorEnv(self._env_config(num_envs=1))
        env.reset()
        completed_gates = None

        for _ in range(1500):
            _, _, _, _, info = env.step(raw_geometric_gate_teacher_action(env))
            if bool(info["done"][0]):
                completed_gates = int(info["gates_passed"][0])
                self.assertFalse(bool(info["missed_gate"][0]))
                self.assertFalse(bool(info["collision"][0]))
                self.assertFalse(bool(info["out_of_bounds"][0]))
                break

        self.assertEqual(completed_gates, 6)

    def test_teacher_tuning_changes_action(self) -> None:
        default_env = AIGPVectorEnv(self._env_config(num_envs=1))
        tuned_env = AIGPVectorEnv(
            self._env_config(
                num_envs=1,
                swift_teacher_lateral_position_gain=2.4,
                swift_teacher_lateral_speed_gain=2.6,
            )
        )
        for env in (default_env, tuned_env):
            env.reset()
            lateral = env.gate_lateral[env.gate_index[0]]
            env.position[0] = env._active_gate_position()[0] - (
                env.gate_normals[env.gate_index[0]] * 8.0
            ) + lateral * 1.0
        default_action = geometric_gate_teacher_action(default_env)
        tuned_action = geometric_gate_teacher_action(tuned_env)

        self.assertGreater(float((default_action - tuned_action).abs().max()), 1e-4)

    @staticmethod
    def _env_config(*, num_envs: int, **overrides: object) -> AIGPEnvConfig:
        config = {
            "actor_observation_mode": "structured_teacher_v2",
            "track_name": AI_GP_TRACK_NAME,
            "start_position_ned_m": (0.0, 0.0, 0.0),
            "num_envs": num_envs,
            "device": "cpu",
            "randomization": False,
            "near_gate_spawn_ratio_start": 0.0,
            "near_gate_spawn_ratio_end": 0.0,
            "dynamics_model": "measured_ai_gp_v1",
            "max_roll_rate_radps": 0.30,
            "max_pitch_rate_radps": 0.20,
            "max_yaw_rate_radps": 0.15,
            "thrust_acceleration_bias_mps2": -5.6171800889,
            "thrust_acceleration_gain_mps2": 57.2914093237,
            "base_pitch_offset_rad": 0.3106907308,
            "linear_drag_xyz": (0.1417047358, 0.1417047358, 0.3905510548),
            "quadratic_drag_xyz": (0.0222959629, 0.0222959629, 0.0117516223),
            "rate_response_gain": (2.6503460081, 2.5121130023, 2.5812295052),
            "command_latency_s": 0.046,
            "command_latency_s_range": (0.046, 0.046),
            "rate_time_constant_s": 0.005,
            "rate_time_constant_s_range": (0.005, 0.005),
            "action_response": 1.0,
            "action_response_range": (1.0, 1.0),
            "max_altitude_m": 35.0,
            "max_lateral_m": 20.0,
            "max_forward_m": 180.0,
            "soft_collision_fraction": 0.0,
            "terminate_on_missed_gate": True,
            "max_episode_steps": 1500,
        }
        config.update(overrides)
        return AIGPEnvConfig(**config)


if __name__ == "__main__":
    unittest.main()
