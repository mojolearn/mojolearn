"""Root-only selector checks; no native module or GPU is loaded."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class IdenticalDefaultTests(unittest.TestCase):
    def setUp(self):
        # _backend.py reads the host manifest with `from . import host_surface`
        # (f74fdfc22), so it needs a parent package. A bare module object with
        # the package directory as its __path__ gives it one WITHOUT running
        # mojolearn/__init__.py, which is what "no native module is loaded"
        # requires; loaded as a standalone file the import failed on every
        # Python (found 2026-09-23 on the 3.10/3.11 pod, 3 tests red).
        package_dir = Path(__file__).resolve().parents[1] / 'python/mojolearn'
        package = types.ModuleType('mojolearn')
        package.__path__ = [str(package_dir)]
        self._modules = patch.dict(sys.modules, {'mojolearn': package})
        self._modules.start()
        self.addCleanup(self._modules.stop)
        spec = importlib.util.spec_from_file_location('mojolearn._backend', package_dir / '_backend.py')
        self.backend = importlib.util.module_from_spec(spec)
        sys.modules['mojolearn._backend'] = self.backend
        spec.loader.exec_module(self.backend)

    def test_unset_uses_identical_for_import_and_estimator_default(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(self.backend.requested_mode(), 'identical')
            self.assertEqual(self.backend.default_mode(), 'identical')
            sentinel = object()
            self.backend._SETS['identical'] = sentinel
            self.assertIs(self.backend.load_set(None), sentinel)

    def test_explicit_modes_still_honored(self):
        for mode in ('fast', 'deterministic', 'identical'):
            with self.subTest(mode=mode), patch.dict('os.environ', {'MOJOLEARN_NUMERIC_MODE': mode}, clear=True):
                self.assertEqual(self.backend.requested_mode(), mode)

    def test_missing_identical_binary_cannot_fall_back_to_fast(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(self.backend, 'tier_dir', return_value='/nonexistent/identical'), patch.object(self.backend.os.path, 'exists', return_value=False):
            with self.assertRaisesRegex(ImportError, 'identical'):
                self.backend.load_set(None)


if __name__ == '__main__':
    unittest.main()
