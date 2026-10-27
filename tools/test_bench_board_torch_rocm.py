import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bench_board_torch_rocm import configure_hipblaslt
from bench_board_evidence import runtime_environment


class TorchRocmEnvironmentTest(unittest.TestCase):
    def test_bundled_assets_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "lib/hipblaslt/library"
            assets.mkdir(parents=True)
            (root / "lib/libhipblaslt.so").touch()
            torch = SimpleNamespace(__file__=str(root / "__init__.py"),
                                    version=SimpleNamespace(hip="7.1"))
            env = {}
            configure_hipblaslt(torch, env)
            self.assertEqual(env["HIPBLASLT_TENSILE_LIBPATH"], str(assets.resolve()))
            self.assertEqual(runtime_environment(env), env)
            for override in ("/explicit/assets", ""):
                env = {"HIPBLASLT_TENSILE_LIBPATH": override}
                configure_hipblaslt(torch, env)
                self.assertEqual(env["HIPBLASLT_TENSILE_LIBPATH"], override)
            (root / "lib/libhipblaslt.so").unlink()
            env = {}
            configure_hipblaslt(torch, env)
            self.assertEqual(env, {})

    def test_non_rocm_untouched(self):
        torch = SimpleNamespace(version=SimpleNamespace(hip=None))
        env = {}
        configure_hipblaslt(torch, env)
        self.assertEqual(env, {})


if __name__ == "__main__":
    unittest.main()
