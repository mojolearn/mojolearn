"""Dynamic twin import regression; no torch, compilation or measurements."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import bench_board_neural as neural


class NeuralTwinLoader(unittest.TestCase):
    def test_twin_is_importable_during_execution_and_by_class_module(self):
        for alias in (None, "custom_neural_twin"):
            name = alias or "bbn_fake_twin"
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "fake_twin.py").write_text(
                    "import importlib\n"
                    "self_import = importlib.import_module(__name__)\n"
                    "class Model: pass\n")
                with patch.object(neural, "HERE", tmp), patch.dict(sys.modules):
                    mod = neural._load("fake_twin", alias=alias)
                    self.assertIs(mod.self_import, mod)
                    self.assertEqual(mod.Model.__module__, name)
                    self.assertIs(importlib.import_module(mod.Model.__module__), mod)

    def test_failed_import_restores_module_registry(self):
        for previous in (None, types.ModuleType("bbn_fake_twin")):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "fake_twin.py").write_text("raise RuntimeError('broken twin')\n")
                with patch.object(neural, "HERE", tmp), patch.dict(sys.modules):
                    sys.modules.pop("bbn_fake_twin", None)
                    if previous is not None:
                        sys.modules["bbn_fake_twin"] = previous
                    with self.assertRaisesRegex(RuntimeError, "broken twin"):
                        neural._load("fake_twin")
                    self.assertIs(sys.modules.get("bbn_fake_twin"), previous)


if __name__ == "__main__":
    unittest.main()
