"""Non-building import-identity/source tests for supported pathlib layouts."""
import importlib.util
from pathlib import Path, PosixPath, PurePosixPath
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PathConnections(unittest.TestCase):
    def setUp(self):
        name = '_mojolearn_test_path_connections'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'python/mojolearn/_native_loaded_modules.py')
        self.connection = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.connection
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(self.connection)

    def test_exact_stdlib_types_and_version_specific_ancestors_are_retained(self):
        self.assertEqual(self.connection.PATH_TYPES, (PosixPath, PurePosixPath))
        for cls in self.connection.PATH_TYPES:
            for base in cls.__mro__:
                self.assertTrue(any(base is known for known in self.connection.PATH_BASES))
        exports = next(row[2] for row in self.connection.CONNECTIONS if row[0] == 'mojolearn._native_loaded_modules')
        self.assertIs(dict(exports)['PATH_BASES'], self.connection.PATH_BASES)

    def test_subclasses_are_not_added_to_admitted_types_or_bases(self):
        class ForeignPath(PosixPath):
            def __fspath__(self):
                raise AssertionError('must never be invoked')
        self.assertFalse(any(ForeignPath is cls for cls in self.connection.PATH_TYPES + self.connection.PATH_BASES))

    def test_both_native_paths_use_shared_identities_and_preserve_descriptor_refusal(self):
        native = (ROOT / 'bindings/native_json.mojo').read_text().split('def native_path(value:')[1].split('def native_path_exists')[0]
        archive = (ROOT / 'bindings/classical_archive.mojo').read_text().split('def _archive_path(')[1].split('def ', 1)[0]
        connection = (ROOT / 'python/mojolearn/_native_classical_bridge.py').read_text()
        self.assertIn('native_loaded_module("mojolearn._native_loaded_modules")', native)
        self.assertIn('if cls is not paths[0] and cls is not paths[1]:', native)
        self.assertIn('if py_type(descriptor) is not member_type:', native)
        self.assertIn('if py_type(slot) is not context["archive_path_member_type"]:', archive)
        self.assertIn('archive_path_types=PATH_TYPES', connection)
        self.assertIn('archive_path_bases=PATH_BASES', connection)
        self.assertNotIn('py_call(value.__fspath__', native)
        self.assertNotIn('Python.import_module("pathlib")', native)


if __name__ == '__main__':
    unittest.main()
