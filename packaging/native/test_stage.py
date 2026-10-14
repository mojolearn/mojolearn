# SPDX-License-Identifier: Apache-2.0
"""Packaging checks: source isolation, complete dependencies, and preserved code."""
from pathlib import Path
import tempfile
import unittest

from stage import ROOT, namespace, sources, stage


class NativePackagingTests(unittest.TestCase):
    def test_namespace_does_not_change_docs_comments_or_external_imports(self):
        original = ('"""Example:\nfrom core.a import b\n"""\n'
                    '# from core.a import b\nfrom std.memory import bitcast\n'
                    'from core.a import (\n    b,\n)\nfrom .child import c\n')
        expected = original.replace('\nfrom core.a import (', '\nfrom ...core.a import (')
        self.assertEqual(namespace(original, Path('family/impl/module.mojo'), {'core'}), expected)

    def test_import_alias_preserves_the_bound_name(self):
        self.assertEqual(namespace('import core.a as A\n', Path('family/a.mojo'), {'core'}),
                         'from ..core import a as A\n')
        self.assertEqual(namespace('import core\n', Path('family/a.mojo'), {'core'}),
                         'from .. import core\n')

    def test_python_dependency_refuses_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'api.mojo').write_text('from python import Python\n')
            with self.assertRaisesRegex(ValueError, 'Python runtime'):
                sources(root, ['api.mojo'])

    def test_missing_dependency_refuses_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'api.mojo').write_text('from missing.module import a\n')
            with self.assertRaisesRegex(ValueError, 'unresolved import'):
                sources(root, ['api.mojo'])
            (root / 'missing').mkdir()
            (root / 'missing/__init__.mojo').write_text('')
            with self.assertRaisesRegex(ValueError, 'unresolved import'):
                sources(root, ['api.mojo'])

    def test_standalone_check_is_preserved_under_a_private_name(self):
        self.assertEqual(namespace('def main() raises:\n    check()\n', Path('family/check.mojo'), set()),
                         'def _standalone_main() raises:\n    check()\n')

    def test_staged_package_is_closed_and_native_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp).resolve() / 'stage'
            stage(ROOT, destination)
            package = destination / 'mojolearn'
            paths = list(package.rglob('*.mojo'))
            reached = sources(destination, [path.relative_to(destination) for path in paths])
            self.assertEqual(set(paths), set(reached))
            self.assertIn('comptime NATIVE_PACKAGE = True',
                          (package / 'core/package_profile.mojo').read_text())
            self.assertIn('comptime NATIVE_PACKAGE = False',
                          (ROOT / 'core/package_profile.mojo').read_text())
            with self.assertRaises(FileExistsError):
                stage(ROOT, destination)


if __name__ == '__main__':
    unittest.main()
