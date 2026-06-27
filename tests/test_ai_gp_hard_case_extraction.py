import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai_gp_rl.track import ai_gp_track_surrogate_positions
from scripts.extract_ai_gp_hard_cases import extract_hard_cases


class HardCaseExtractionTests(unittest.TestCase):
    def test_extracts_pre_failure_window_from_tracked_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "telemetry.json"
            gate_0_z = ai_gp_track_surrogate_positions()[0][2]
            report.write_text(
                json.dumps(
                    {
                        "evaluation_seed": 1001,
                        "environment_randomization": True,
                        "trajectories": [
                            {
                                "env_id": 3,
                                "samples": [
                                    {
                                        "step": 1,
                                        "time_s": 0.0,
                                        "position_m": [20.0, 0.0, gate_0_z],
                                        "velocity_mps": [1.0, 0.0, 0.0],
                                        "active_gate_index": 0,
                                        "gates_passed": 0,
                                        "missed_gate": False,
                                        "collision": False,
                                        "out_of_bounds": False,
                                        "done": False,
                                    },
                                    {
                                        "step": 90,
                                        "time_s": 1.8,
                                        "position_m": [24.0, 0.5, gate_0_z],
                                        "velocity_mps": [2.0, 0.1, 0.0],
                                        "active_gate_index": 0,
                                        "gates_passed": 0,
                                        "missed_gate": False,
                                        "collision": False,
                                        "out_of_bounds": False,
                                        "done": False,
                                    },
                                    {
                                        "step": 100,
                                        "time_s": 2.0,
                                        "position_m": [25.0, 0.8, gate_0_z],
                                        "velocity_mps": [2.0, 0.1, 0.0],
                                        "active_gate_index": 0,
                                        "gates_passed": 0,
                                        "missed_gate": True,
                                        "collision": False,
                                        "out_of_bounds": False,
                                        "done": True,
                                    },
                                ],
                            },
                            {
                                "env_id": 4,
                                "samples": [
                                    {
                                        "step": 100,
                                        "time_s": 2.0,
                                        "position_m": [25.0, 0.0, 2.0],
                                        "velocity_mps": [2.0, 0.0, 0.0],
                                        "active_gate_index": 5,
                                        "gates_passed": 6,
                                        "missed_gate": False,
                                        "collision": False,
                                        "out_of_bounds": False,
                                        "done": True,
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "hard_cases.json"

            with redirect_stdout(io.StringIO()):
                result = extract_hard_cases([report], output, window_s=0.5)

            self.assertEqual(result["tracked_trajectory_count"], 2)
            self.assertEqual(result["failed_tracked_trajectory_count"], 1)
            self.assertEqual(result["hard_case_count"], 2)
            self.assertEqual(
                result["summary"]["by_failure_type"], {"missed_gate": 2}
            )
            self.assertIn("plane_offset_m", result["hard_cases"][0]["gate_frame"])
            self.assertLess(
                abs(result["hard_cases"][0]["gate_frame"]["vertical_offset_m"]),
                1e-6,
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
