"""The CPU-pod build route is opt-in (--build-backend cpu-box, Andrew 2026-09-25): one
tools/release_linux_build.sh per set, all three launched together, each
writing the release-build tree pack_wheel.py and proof_ok read; the GPU legs
stay available by name and a CPU leg never takes the NVIDIA GPU walk."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parent / "release.py")
rel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rel)
C = "a" * 40


class CpuBox(unittest.TestCase):
    def setUp(self):
        self.ctx = argparse.Namespace(rel=Path(tempfile.mkdtemp()), commit=C)

    def test_default_backend_is_gpu_legs_cpu_box_opt_in(self):
        self.assertIs(rel.BUILD_BACKENDS["cpu-box"], rel.cpu_legs)
        self.assertIs(rel.BUILD_BACKENDS["gpu-legs"], rel.gpu_legs)
        src = (Path(__file__).resolve().parent / "release.py").read_text()
        self.assertIn('ap.add_argument("--build-backend", default="gpu-legs"', src)

    def test_three_legs_one_set_each(self):
        legs = rel.cpu_legs(self.ctx)
        self.assertEqual([l.name for l in legs], ["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"])
        for l in legs:
            self.assertEqual(l.command[:4], ["bash", "tools/release_linux_build.sh", C, "--rent"])
            self.assertEqual(l.command[l.command.index("--archs") + 1], l.arch)
            out = Path(l.command[l.command.index("--out") + 1])
            self.assertEqual(out, l.out_dir)
            self.assertEqual(l.release_build, out / l.name / "release-build")
            self.assertNotIn("--gpu", l.command)


if __name__ == "__main__":
    unittest.main()
