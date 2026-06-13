import tempfile
import unittest
from pathlib import Path

import torch

from scripts.configure_ai_gp_action_governor import configure_governor


class AIGPActionGovernorConfigTests(unittest.TestCase):
    def test_attaches_governor_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pt"
            lineage = root / "lineage.pt"
            output = root / "governed.pt"
            torch.save(
                {
                    "metadata": {
                        "policy_role": "distilled_live_student",
                    }
                },
                source,
            )
            torch.save(
                {
                    "metadata": {
                        "source_teacher_checkpoint": "teacher.pt",
                        "source_teacher_global_step": 123,
                    }
                },
                lineage,
            )

            configure_governor(
                source,
                output,
                slew_limits=(0.15, 0.25, 0.25, 0.35),
                upward_brake_start_mps=1.0,
                upward_brake_gain=0.1,
                lineage_checkpoint_path=lineage,
            )

            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            metadata = checkpoint["metadata"]
            self.assertEqual(metadata["source_teacher_checkpoint"], "teacher.pt")
            self.assertEqual(metadata["source_ungoverned_checkpoint"], "source.pt")
            self.assertEqual(
                metadata["action_governor"]["slew_limits"],
                [0.15, 0.25, 0.25, 0.35],
            )

    def test_rejects_non_student_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pt"
            torch.save({"metadata": {"policy_role": "teacher"}}, source)

            with self.assertRaisesRegex(ValueError, "distilled live student"):
                configure_governor(
                    source,
                    root / "output.pt",
                    slew_limits=(0.1, 0.1, 0.1, 0.1),
                    upward_brake_start_mps=1.0,
                    upward_brake_gain=0.1,
                )


if __name__ == "__main__":
    unittest.main()
