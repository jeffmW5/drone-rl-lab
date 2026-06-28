import unittest
from pathlib import Path

from scripts.run_ai_gp_policy_ab_windows import (
    PolicySpec,
    RuntimeConfig,
    build_policy_command,
    parse_policy_specs,
    rank_policy_summaries,
    summarize_policy_rows,
)


class PolicyABWindowsTests(unittest.TestCase):
    def test_parse_policy_specs_resolves_repo_relative_paths(self) -> None:
        policies = parse_policy_specs([
            "040=exports/ai_gp/ai_gp_040_near_gate_teacher_structured_policy.json",
            "041=exports/ai_gp/ai_gp_041_windows_transfer_gate2_hardcase_structured_policy.json",
        ])

        self.assertEqual([policy.label for policy in policies], ["040", "041"])
        self.assertTrue(policies[0].path.is_absolute())
        self.assertEqual(policies[0].path.name, "ai_gp_040_near_gate_teacher_structured_policy.json")

    def test_build_policy_command_passes_policy_and_runtime(self) -> None:
        policy = PolicySpec("041", Path("policy.json"))
        runtime = RuntimeConfig(
            thrust_multiplier=1.12,
            roll_rate_multiplier=2.0,
            pitch_rate_multiplier=1.0,
            yaw_rate_multiplier=2.0,
            duration_s=30.0,
            control_rate_hz=50.0,
            attempts=5,
        )

        command = build_policy_command(
            python="python",
            policy=policy,
            config_id="ab_p041",
            runtime=runtime,
        )

        self.assertIn("--policy", command)
        self.assertEqual(command[command.index("--policy") + 1], "policy.json")
        self.assertEqual(command[command.index("--attempts") + 1], "5")
        self.assertEqual(command[command.index("--target-gates") + 1], "0")
        self.assertIn("--allow-gate-plane-miss", command)
        self.assertEqual(command[command.index("--thrust-multiplier") + 1], "1.12")
        self.assertEqual(command[command.index("--roll-rate-multiplier") + 1], "2.0")
        self.assertEqual(command[command.index("--pitch-rate-multiplier") + 1], "1.0")
        self.assertEqual(command[command.index("--yaw-rate-multiplier") + 1], "2.0")

    def test_policy_summary_and_ranking_match_windows_targets(self) -> None:
        policy_040 = PolicySpec("040", Path("p040.json"))
        policy_041 = PolicySpec("041", Path("p041.json"))
        rows_040 = [
            _row("040_a", max_gate=2, gate0=True, collision=True, policy_steps=100),
            _row("040_b", max_gate=1, gate0=True, collision=True, policy_steps=90),
            _row("040_c", max_gate=2, gate0=True, collision=True, policy_steps=110),
        ]
        rows_041 = [
            _row("041_a", max_gate=3, gate0=True, collision=True, policy_steps=120),
            _row("041_b", max_gate=2, gate0=True, collision=True, policy_steps=100),
            _row("041_c", max_gate=3, gate0=True, collision=False, policy_steps=130),
        ]

        summary_040 = summarize_policy_rows(policy_040, rows_040)
        summary_041 = summarize_policy_rows(policy_041, rows_041)
        ranked = rank_policy_summaries([summary_040, summary_041])

        self.assertFalse(summary_040["passes_windows_retest_target"])
        self.assertTrue(summary_041["passes_windows_retest_target"])
        self.assertEqual(ranked[0]["policy_label"], "041")


def _row(
    session: str,
    *,
    max_gate: int,
    gate0: bool,
    collision: bool,
    policy_steps: int,
) -> dict:
    return {
        "session": session,
        "race_finished": False,
        "max_gate": max_gate,
        "collision_count": 1 if collision else 0,
        "gate0_passed": gate0,
        "policy_steps": policy_steps,
    }


if __name__ == "__main__":
    unittest.main()
