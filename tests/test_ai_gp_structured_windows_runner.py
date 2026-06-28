import math
import unittest
from pathlib import Path

from scripts.run_ai_gp_structured_windows import (
    StructuredPolicy,
    build_structured_observation,
    gate_plane_metrics,
)
from adapter.telemetry import AttitudeSample, TelemetrySample, Vector3


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "exports"
    / "ai_gp"
    / "ai_gp_040_near_gate_teacher_structured_policy.json"
)


class StructuredWindowsRunnerTests(unittest.TestCase):
    def test_loads_export_and_verifies_vectors(self) -> None:
        policy = StructuredPolicy.load(POLICY_PATH)
        policy.verify_test_vectors()

        self.assertEqual(policy.artifact["observation_contract"], "structured_teacher_v2")
        self.assertEqual(len(policy.actor_features), 26)
        self.assertEqual(policy.gate_count, 6)
        self.assertEqual(len(policy.layers), 3)

    def test_builds_structured_observation_from_ned_telemetry(self) -> None:
        policy = StructuredPolicy.load(POLICY_PATH)
        sample = TelemetrySample(
            monotonic_time_s=1.0,
            mission_time_s=1.0,
            position_m=Vector3(0.0, 0.0, 0.0),
            velocity_mps=Vector3(-1.0, 0.0, 0.0),
            attitude_rad=AttitudeSample(
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rad=math.pi,
            ),
            angular_rate_radps=Vector3(0.3, -0.2, 0.15),
            battery_voltage_v=None,
            is_armed=False,
            mode=None,
        )

        observation = build_structured_observation(
            policy,
            sample,
            0,
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(len(observation), 26)
        self.assertGreater(observation[0], 0.0)

        action = policy.act(observation)
        command = policy.map_action(action)
        self.assertEqual(set(command), {
            "thrust_normalized",
            "roll_rate_radps",
            "pitch_rate_radps",
            "yaw_rate_radps",
        })
        self.assertGreaterEqual(command["thrust_normalized"], 0.0)
        self.assertLessEqual(command["thrust_normalized"], 1.0)

        metrics = gate_plane_metrics(policy, sample, 0)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertLess(metrics["signed_distance_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
