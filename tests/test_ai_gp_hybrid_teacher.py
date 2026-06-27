import unittest

import torch

from ai_gp_rl.env import AIGPEnvConfig, AIGPVectorEnv
from ai_gp_rl.hybrid_teacher import parse_gate_indices, teacher_takeover_mask


class HybridTeacherTests(unittest.TestCase):
    def test_parse_gate_indices(self) -> None:
        self.assertIsNone(parse_gate_indices("all"))
        self.assertIsNone(parse_gate_indices("*"))
        self.assertEqual(parse_gate_indices("1, 3,5"), {1, 3, 5})

    def test_teacher_takeover_modes_and_gate_filter(self) -> None:
        env = AIGPVectorEnv(
            AIGPEnvConfig(
                num_envs=4,
                device="cpu",
                actor_observation_mode="structured_teacher_v2",
                randomization=False,
                near_gate_spawn_ratio_start=0.0,
                near_gate_spawn_ratio_end=0.0,
            )
        )
        env.reset()
        env_ids = torch.arange(env.num_envs, device=env.device)
        gate_index = env.gate_index
        gate_position = env.env_gate_positions[env_ids, gate_index]
        normal = env.gate_normals[gate_index]
        lateral = env.gate_lateral[gate_index]
        env.position.copy_(gate_position - normal * 5.0)

        policy_mask, _ = teacher_takeover_mask(
            env, mode="policy", takeover_distance_m=10.0
        )
        teacher_mask, _ = teacher_takeover_mask(
            env, mode="teacher", takeover_distance_m=10.0
        )
        near_mask, frame = teacher_takeover_mask(
            env, mode="near_gate_teacher", takeover_distance_m=10.0
        )
        filtered_mask, _ = teacher_takeover_mask(
            env,
            mode="near_gate_teacher",
            takeover_distance_m=10.0,
            gate_indices={int(gate_index[0]) + 1},
        )

        self.assertFalse(bool(policy_mask.any()))
        self.assertTrue(bool(teacher_mask.all()))
        self.assertTrue(bool(near_mask.all()))
        self.assertFalse(bool(filtered_mask.any()))
        self.assertTrue(bool((frame["distance_to_plane_m"] > 4.9).all()))

        centered_mask, _ = teacher_takeover_mask(
            env,
            mode="off_center_near_gate_teacher",
            takeover_distance_m=10.0,
            lateral_threshold_m=0.4,
            vertical_threshold_m=0.4,
        )
        env.position.copy_(gate_position - normal * 5.0 + lateral * 0.5)
        off_center_mask, _ = teacher_takeover_mask(
            env,
            mode="off_center_near_gate_teacher",
            takeover_distance_m=10.0,
            lateral_threshold_m=0.4,
            vertical_threshold_m=0.4,
        )

        self.assertFalse(bool(centered_mask.any()))
        self.assertTrue(bool(off_center_mask.all()))


if __name__ == "__main__":
    unittest.main()
