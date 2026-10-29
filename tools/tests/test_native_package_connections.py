import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('connections', ROOT / 'tools/check_native_package_connections.py')
connections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connections)


class ConnectionSourceGuard(unittest.TestCase):
    def fixture(self, declaration):
        return {'bindings/example.mojo': 'var m = native_loaded_module("mojolearn._mode")',
                'python/mojolearn/_mode.py': declaration}

    def test_alias_and_lazy_self_registration_without_import(self):
        files = self.fixture('from ._native_loaded_modules import connect_module as link\n'
                             'def setup():\n    link(__name__, unavailable, ("_MODE_METADATA_CONTEXT",))\n')
        self.assertEqual(connections.inspect(files)['errors'], [])

    def test_missing_owner_refused(self):
        self.assertIn('found 0', connections.inspect(self.fixture(''))['errors'][0])

    def test_missing_export_refused(self):
        files = self.fixture('from ._native_loaded_modules import connect_module\n'
                             'connect_module(__name__, owner, ("OTHER",))')
        self.assertIn('missing connected exports', connections.inspect(files)['errors'][0])

    def test_nonliteral_exports_refused(self):
        files = self.fixture('from ._native_loaded_modules import connect_module\n'
                             'connect_module(__name__, owner, computed_exports)')
        self.assertIn('literal string tuple', connections.inspect(files)['errors'][0])

    def test_duplicate_declaration_refused(self):
        source = 'from ._native_loaded_modules import connect_module\n'
        source += 'connect_module(__name__, owner, ("_MODE_METADATA_CONTEXT",))\n' * 2
        self.assertIn('found 2', connections.inspect(self.fixture(source))['errors'][0])

    def test_comments_and_external_optional_modules_not_owners(self):
        files = {'bindings/x.mojo': '# native_loaded_module("mojolearn.absent")\n'
                 'var m = native_loaded_module("torch")'}
        self.assertEqual(connections.inspect(files)['consumers'], {})

    def test_new_owner_requires_reviewed_exports(self):
        files = {'bindings/x.mojo': 'native_loaded_module("mojolearn.new_owner")'}
        self.assertIn('reviewed export contract', connections.inspect(files)['errors'][0])

    def test_repository_declarations_complete(self):
        self.assertEqual(connections.inspect(connections.sources(ROOT))['errors'], [])


if __name__ == '__main__':
    unittest.main()
