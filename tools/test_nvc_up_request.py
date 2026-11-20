"""The NVIDIA measurement pod request: L40S only, one GPU per pod (no network, no rent)."""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402


def up_env(environ):
    fake = types.SimpleNamespace(NVC_MEASURE_GPU=release.Release.NVC_MEASURE_GPU)
    return release.Release.nvc_up_env(fake, environ)


def central(env_extra):
    with tempfile.TemporaryDirectory() as d:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("MOJOLEARN_", "NVC_", "PC_ALLOW"))}
        env.update(MOJOLEARN_ORCHESTRATOR="1", MOJOLEARN_NVC_NO_API="1", MOJOLEARN_NVC_PRINT_REQUEST="1",
                   MOJOLEARN_NVC_CONF=str(Path(d) / "none.env"), MOJOLEARN_NVC_HOME=d,
                   MOJOLEARN_DEVPOD_STATE=d, **env_extra)
        return subprocess.run(["bash", str(ROOT / "tools/nvidia_central.sh"), "up", "1"], env=env,
                              capture_output=True, text=True, timeout=60)


class ReleaseGateEnv(unittest.TestCase):
    def test_default_is_one_l40s(self):
        env = up_env({})
        self.assertEqual(env["MOJOLEARN_NVC_GPUS"], "NVIDIA L40S")
        self.assertEqual(env["MOJOLEARN_DEVPOD_GPUS"], "NVIDIA L40S")
        self.assertEqual(env["NVC_GPU_COUNTS"], "1")
        self.assertEqual(env["MOJOLEARN_ORCHESTRATOR"], "1")
        self.assertNotIn("PC_ALLOW_HW_DIFF", env)

    def test_other_gpu_refused(self):
        with self.assertRaises(release.StepFailed):
            up_env({"MOJOLEARN_DEVPOD_GPUS": "NVIDIA RTX 6000 Ada Generation"})

    def test_other_gpu_with_hw_diff_passes_through(self):
        env = up_env({"MOJOLEARN_NVC_GPUS": "NVIDIA GeForce RTX 4090", "PC_ALLOW_HW_DIFF": "1"})
        self.assertEqual(env["MOJOLEARN_NVC_GPUS"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(env["PC_ALLOW_HW_DIFF"], "1")
        self.assertEqual(env["NVC_GPU_COUNTS"], "1")


class CentralUpRequest(unittest.TestCase):
    def test_default_request(self):
        p = central({})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("request: counts=1 gpus=NVIDIA L40S", p.stdout)

    def test_multi_gpu_refused(self):
        p = central({"NVC_GPU_COUNTS": "4 2 1"})
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("one GPU per pod", p.stdout + p.stderr)

    def test_other_gpu_refused_then_allowed(self):
        p = central({"MOJOLEARN_NVC_GPUS": "NVIDIA RTX 6000 Ada Generation"})
        self.assertNotEqual(p.returncode, 0)
        p = central({"MOJOLEARN_NVC_GPUS": "NVIDIA RTX 6000 Ada Generation", "PC_ALLOW_HW_DIFF": "1"})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
