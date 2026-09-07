"""Root-only selector checks; no native module or GPU is loaded."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


class IdenticalDefaultTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_backend.py'
        spec = importlib.util.spec_from_file_location('default_selector_fixture', path)
        self.backend = importlib.util.module_from_spec(spec)
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
