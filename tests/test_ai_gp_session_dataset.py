import json
import tempfile
import unittest
from pathlib import Path

from ai_gp_rl.contract import (
    ACTOR_OBS_DIM,
    CORNER_BASE_OBS_DIM,
    MOTION_OBS_DIM,
    TEMPORAL_BASE_OBS_DIM,
)
from ai_gp_rl.session_dataset import export_session_dataset


class AIGPSessionDatasetTests(unittest.TestCase):
    def test_exports_detection_and_prior_telemetry_to_actor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir) / "session-1"
            session.mkdir()
            (session / "frames").mkdir()
            (session / "frames" / "10.jpg").write_bytes(b"jpeg")
            (session / "telemetry.jsonl").write_text(
                json.dumps(
                    {
                        "monotonic_time_s": 1.0,
                        "velocity_mps": {"x": 4.0, "y": 2.0, "z": -1.0},
                        "attitude_rad": {
                            "roll_rad": 0.0,
                            "pitch_rad": 0.0,
                            "yaw_rad": 0.0,
                        },
                        "angular_rate_radps": {"x": 0.3, "y": -0.6, "z": 0.2},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (session / "detections.jsonl").write_text(
                json.dumps(
                    {
                        "monotonic_time_s": 1.05,
                        "frame_id": "10",
                        "detection_count": 1,
                        "detections": [
                            {
                                "confidence": 0.9,
                                "bbox": {
                                    "center": {"x": 0.75, "y": 0.25},
                                    "width": 0.4,
                                    "height": 0.2,
                                },
                                "corners": [
                                    {"x": 0.55, "y": 0.15},
                                    {"x": 0.95, "y": 0.15},
                                    {"x": 0.95, "y": 0.35},
                                    {"x": 0.55, "y": 0.35},
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = export_session_dataset(session)

            self.assertEqual(summary.row_count, 1)
            self.assertEqual(summary.full_detection_rows, 1)
            self.assertEqual(summary.persisted_frame_count, 1)
            row = json.loads((session / "rl_features.jsonl").read_text(encoding="utf-8"))
            observation = row["actor_observation"]
            self.assertEqual(len(observation), ACTOR_OBS_DIM)
            self.assertAlmostEqual(observation[0], 0.5)
            self.assertAlmostEqual(observation[1], -0.25)
            self.assertAlmostEqual(observation[2], 0.125)
            self.assertEqual(observation[3:6], [0.0, -0.0, -1.0])
            self.assertAlmostEqual(observation[9], 0.5)
            self.assertAlmostEqual(observation[10], -0.5)
            self.assertAlmostEqual(observation[11], 0.08)
            self.assertAlmostEqual(observation[12], 0.9)
            self.assertAlmostEqual(observation[13], 0.0)
            temporal = row["temporal_base_observation"]
            self.assertEqual(len(temporal), TEMPORAL_BASE_OBS_DIM)
            self.assertAlmostEqual(temporal[11], 0.4)
            self.assertAlmostEqual(temporal[12], 0.2)
            self.assertAlmostEqual(temporal[13], 0.08)
            self.assertEqual(
                len(row["corner_base_observation"]), CORNER_BASE_OBS_DIM
            )
            self.assertAlmostEqual(row["corner_base_observation"][14], 0.1)
            self.assertEqual(row["corner_base_observation"][22], 1.0)
            self.assertEqual(len(row["motion_observation"]), MOTION_OBS_DIM)


if __name__ == "__main__":
    unittest.main()
