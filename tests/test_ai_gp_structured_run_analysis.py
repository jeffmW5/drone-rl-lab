import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_ai_gp_structured_runs import analyze_session, rank_sessions, summarize_group


class StructuredRunAnalysisTests(unittest.TestCase):
    def test_analyzes_collision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "session_01"
            session.mkdir()
            (session / "summary.json").write_text(
                json.dumps(
                    {
                        "abort_reason": "collision_abort",
                        "max_active_gate_index": 2,
                        "gate0_passed": True,
                        "race_finished": False,
                        "collision_count": 1,
                        "policy_steps": 3,
                        "command_count": 3,
                        "max_altitude_rise_m": 1.25,
                        "max_upward_speed_mps": 0.75,
                        "thrust_multiplier": 1.1,
                        "roll_rate_multiplier": 2.0,
                        "pitch_rate_multiplier": 1.0,
                        "yaw_rate_multiplier": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                session / "events.jsonl",
                [
                    {"event_type": "race.status", "monotonic_time_s": 1.0, "fields": {"active_gate_index": 1}},
                    {"event_type": "collision", "monotonic_time_s": 2.0, "fields": {"impact": 3.5}},
                ],
            )
            _write_jsonl(
                session / "policy.jsonl",
                [
                    {
                        "monotonic_time_s": 1.9,
                        "gate_plane": {
                            "signed_distance_m": -0.2,
                            "lateral_offset_m": 0.5,
                            "vertical_offset_m": -0.1,
                            "rectangular_margin_m": 0.8,
                        },
                        "mapped_command": {
                            "thrust_normalized": 0.3,
                            "roll_rate_radps": 0.2,
                            "pitch_rate_radps": -0.1,
                            "yaw_rate_radps": 0.05,
                        },
                        "position_m": {"x": 1, "y": 2, "z": 3},
                        "velocity_mps": {"x": 3, "y": 4, "z": 0},
                    }
                ],
            )

            row = analyze_session(session)

        self.assertEqual(row["max_gate"], 2)
        self.assertEqual(row["active_gate_at_collision"], 1)
        self.assertEqual(row["collision_impact"], 3.5)
        self.assertEqual(row["speed_mps"], 5.0)
        self.assertEqual(row["rectangular_margin_m"], 0.8)

    def test_ranks_and_summarizes(self) -> None:
        rows = [
            {"session": "a", "race_finished": False, "max_gate": 1, "collision_count": 1, "policy_steps": 10, "gate0_passed": True},
            {"session": "b", "race_finished": False, "max_gate": 2, "collision_count": 2, "policy_steps": 20, "gate0_passed": True},
            {"session": "c", "race_finished": False, "max_gate": 2, "collision_count": 1, "policy_steps": 15, "gate0_passed": True},
        ]

        ranked = rank_sessions(rows)
        summary = summarize_group(rows)

        self.assertEqual(ranked[0]["session"], "c")
        self.assertEqual(summary["best_session"], "c")
        self.assertEqual(summary["best_max_gate"], 2)
        self.assertEqual(summary["gate0_pass_count"], 3)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
