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


class MambaInputsWithoutTorch(unittest.TestCase):
    """0.8.66 AMD measurement smoke: the measured venv has no torch, and the Mamba lanes' input step imported
    mamba/corpus/gen_corpus.py (torch at module top), so no Mamba recipe reached its parameter probe."""

    def test_mamba_weight_spec_reads_the_corpus_metadata_without_torch(self):
        with patch.dict(sys.modules, {"torch": None, "torch.nn.functional": None}):
            neural._MAMBA_METADATA.clear()
            for model in ("mamba1", "mamba2", "mamba3"):
                spec, (xshape, xrange) = neural.mamba_weight_spec(model, dict(d_model=64, batch=2, length=8))
                self.assertEqual([n for n, _, _ in spec], list(neural.MAMBA_NAMES[model]))
                self.assertEqual(xshape, (2, 8, 64))
                self.assertTrue(all(lo <= hi for _, _, (lo, hi) in spec))
        neural._MAMBA_METADATA.clear()
