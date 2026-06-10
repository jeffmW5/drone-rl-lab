import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ManagePodTests(unittest.TestCase):
    def _base_environment(self, temp_path: Path) -> dict[str, str]:
        fake_key = temp_path / "id_ed25519"
        fake_key.write_text("test key\n", encoding="ascii")
        env = os.environ.copy()
        env.update(
            {
                "DRONE_RL_DEPLOY_KEY": str(fake_key),
                "RUNPOD_API_KEY": "test-api-key",
                "RUNPOD_POD_ID": "old-pod",
            }
        )
        return env

    def test_create_new_pod_stdout_contains_only_pod_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            fake_ssh_keygen = fake_bin / "ssh-keygen"
            fake_ssh_keygen.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'ssh-ed25519 test-public-key'\n",
                encoding="ascii",
            )
            fake_ssh_keygen.chmod(0o755)

            command = textwrap.dedent(
                """
                source scripts/manage_pod.sh
                runpod_query() {
                    printf '%s\n' '{"data":{"podFindAndDeployOnDemand":{"id":"pod123"}}}'
                }
                create_new_pod "NVIDIA GeForce RTX 3090"
                """
            )
            env = self._base_environment(temp_path)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "pod123")
        self.assertIn("Trying to create new pod", result.stderr)

    def test_check_env_uses_persisted_id_when_environment_id_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pod_file = temp_path / "runpod_pod_id"
            pod_file.write_text("new-pod\n", encoding="ascii")
            command = textwrap.dedent(
                """
                source scripts/manage_pod.sh
                pod_exists() {
                    [[ "$1" == "new-pod" ]]
                }
                check_env
                printf '%s\n' "$RUNPOD_POD_ID"
                """
            )
            env = self._base_environment(temp_path)
            env["DRONE_RL_RUNPOD_POD_ID_FILE"] = str(pod_file)
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "new-pod")
        self.assertIn("Configured pod no longer exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
