import math
import unittest

from ai_gp_rl.contract import (
    ACTOR_OBS_DIM,
    CORNER_BASE_OBS_DIM,
    MOTION_OBS_DIM,
    TEMPORAL_BASE_OBS_DIM,
    ActionCalibration,
    LivePolicyFeatures,
    TemporalLivePolicyFeatures,
    build_actor_observation,
    build_corner_base_observation,
    build_motion_observation,
    build_temporal_base_observation,
    temporal_feature_names,
)


class AIGPContractTests(unittest.TestCase):
    def test_actor_observation_has_stable_shape_and_finite_values(self) -> None:
        observation = build_actor_observation(
            LivePolicyFeatures(
                body_velocity_mps=(4.0, -2.0, 1.0),
                gravity_body=(0.0, 0.0, -1.0),
                angular_rate_radps=(0.3, -0.6, 0.2),
                gate_center_normalized=(0.25, -0.5),
                gate_area_normalized=0.2,
                gate_confidence=0.9,
                gate_age_s=0.1,
                previous_action=(0.1, -0.2, 0.3, -0.4),
            )
        )
        self.assertEqual(len(observation), ACTOR_OBS_DIM)
        self.assertTrue(all(math.isfinite(value) for value in observation))

    def test_action_calibration_maps_normalized_action(self) -> None:
        calibration = ActionCalibration(
            hover_thrust=0.4,
            thrust_span_up=0.2,
            thrust_span_down=0.1,
            max_roll_rate_radps=1.0,
            max_pitch_rate_radps=1.5,
            max_yaw_rate_radps=0.5,
        )
        command = calibration.map_action((0.5, 1.0, -1.0, 0.5))
        self.assertAlmostEqual(command["thrust_normalized"], 0.5)
        self.assertAlmostEqual(command["roll_rate_radps"], 1.0)
        self.assertAlmostEqual(command["pitch_rate_radps"], -1.5)
        self.assertAlmostEqual(command["yaw_rate_radps"], 0.25)

    def test_temporal_base_preserves_gate_shape(self) -> None:
        observation = build_temporal_base_observation(
            TemporalLivePolicyFeatures(
                body_velocity_mps=(4.0, -2.0, 1.0),
                gravity_body=(0.0, 0.0, -1.0),
                angular_rate_radps=(0.3, -0.6, 0.2),
                gate_center_normalized=(0.25, -0.5),
                gate_size_normalized=(0.4, 0.2),
                gate_area_normalized=0.08,
                gate_confidence=0.9,
                gate_age_s=0.1,
                previous_action=(0.1, -0.2, 0.3, -0.4),
            )
        )
        self.assertEqual(len(observation), TEMPORAL_BASE_OBS_DIM)
        self.assertEqual(observation[11:14], [0.4, 0.2, 0.08])
        self.assertEqual(len(temporal_feature_names(4)), 4 * TEMPORAL_BASE_OBS_DIM)

    def test_corner_and_motion_observations_have_stable_shapes(self) -> None:
        features = TemporalLivePolicyFeatures(
            body_velocity_mps=(0.0, 0.0, 0.0),
            gravity_body=(0.0, 0.0, -1.0),
            angular_rate_radps=(0.0, 0.0, 0.0),
            gate_center_normalized=(0.0, 0.0),
            gate_size_normalized=(0.4, 0.2),
            gate_area_normalized=0.08,
            gate_confidence=0.9,
            gate_age_s=0.0,
            previous_action=(0.0, 0.0, 0.0, 0.0),
        )
        current = build_temporal_base_observation(features)
        corners = build_corner_base_observation(
            features,
            ((-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)),
        )
        motion = build_motion_observation(current, current)
        self.assertEqual(len(corners), CORNER_BASE_OBS_DIM)
        self.assertEqual(len(motion), MOTION_OBS_DIM)
        self.assertEqual(motion[-4:], [0.0, 0.0, 0.0, 0.0])

    def test_action_calibration_rejects_unknown_hover_thrust(self) -> None:
        with self.assertRaises(ValueError):
            ActionCalibration(
                hover_thrust=0.0,
                thrust_span_up=0.1,
                thrust_span_down=0.1,
            )


if __name__ == "__main__":
    unittest.main()
