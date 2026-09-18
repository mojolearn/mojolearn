"""Public nested APIs must be visible without inheriting unrelated evidence."""
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import verification_matrix as matrix
import wheel_api_audit


class PublicPackageInventoryTests(unittest.TestCase):
    def test_models_exports_resolve_to_their_own_implementation(self):
        surface, modules = matrix.public_surface()
        self.assertIn("models", modules)
        self.assertNotIn("models", surface)
        self.assertEqual(surface["models.CausalLM"],
                         ("class", "python/mojolearn/models/causal_lm.py", "CausalLM"))
        self.assertEqual(surface["models.Tokenizer"],
                         ("class", "python/mojolearn/models/tokenizer.py", "Tokenizer"))
        self.assertIn("models.tokenizer.pretokenize", surface)
        self.assertIn("models.safetensors.Checkpoint", surface)

    def test_exports_are_read_without_executing_package_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "python/mojolearn"
            child = pkg / "example"
            child.mkdir(parents=True)
            (pkg / "__init__.py").write_text("__all__ = ['example']\n")
            (child / "__init__.py").write_text(
                "raise RuntimeError('must not execute')\n"
                "from .impl import Actual as Exported\n__all__ = ['Exported']\n")
            (child / "impl.py").write_text("class Actual: pass\n")
            with patch.object(matrix, "ROOT", tmp), patch.object(matrix, "EXTRA_PUBLIC_MODULES", ()):
                surface, _ = matrix.public_surface()
            self.assertEqual(surface['example.Exported'],
                             ('class', 'python/mojolearn/example/impl.py', 'Actual'))

    def test_alias_cycles_terminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cycle.py'
            path.write_text('A = B\nB = A\n')
            self.assertEqual(matrix.package_export(path, 'A')[0], 'name')

    def test_models_tokenizer_does_not_borrow_bare_tokenizer_evidence(self):
        def ordinary(ml):
            return ml.Tokenizer()
        harness = types.SimpleNamespace(LANES={'ordinary': ordinary})
        row = dict(gpu=['apple'], cpu='training', sabotage='seen(build)', batch='part')
        surface = {'models.Tokenizer': ('class', 'models/tokenizer.py', 'Tokenizer')}
        result = matrix.algorithm_rows(harness, {'ordinary': row}, surface, {'Tokenizer'}, {})
        self.assertEqual(result['models.Tokenizer']['lanes'], [])
        self.assertFalse(result['models.Tokenizer']['routed'])

    def test_state_and_configuration_are_not_counted_as_algorithms(self):
        surface, _ = matrix.public_surface()
        result = matrix.algorithm_rows(types.SimpleNamespace(LANES={}), {}, surface, set(), {})
        for name in ('models.CausalLMState', 'models.config.HFConfig', 'models.FAMILIES'):
            self.assertNotIn(name, result)
        self.assertIn('models.CausalLM', result)

    def test_old_wheel_absence_is_reported_and_inspection_does_not_import(self):
        root = matrix.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / 'old.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('mojolearn/__init__.py',
                                 "raise RuntimeError('do not execute')\n__all__ = []\n")
            report = wheel_api_audit.audit([wheel])
        self.assertEqual(matrix.ROOT, root)
        self.assertEqual(report['wheels'][0]['public_names'], [])
        self.assertIn('models.CausalLM', report['wheels'][0]['source_names_absent_from_wheel'])

    def test_wheel_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / 'invalid.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('mojolearn/../../escape.py', '')
            with self.assertRaisesRegex(ValueError, 'unsafe wheel member'):
                wheel_api_audit.inspect_wheel(wheel)


if __name__ == '__main__':
    unittest.main()
