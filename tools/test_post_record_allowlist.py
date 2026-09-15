"""The post-record allowlist (0.8.6): after a release record lands, the final
wheel is packed from the recorded build proofs, and python/mojolearn/host_surface.py
may differ from its built copy only in the named record-list assignments.
File-only; no package import, no build, no GPU."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import verify_linux_surface_qualification as surface

MANIFEST = 'python/mojolearn/host_surface.py'
BUILT = '"""Doc."""\nimport sys\nTRAINING_GPU_COLUMNS = ("old",)\n\ndef f():\n    return 1\n'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Equivalence(unittest.TestCase):
    def test_record_lists_and_comments_may_change(self):
        current = ('"""Doc."""\nimport sys\n# the release record\nTRAINING_GPU_COLUMNS = ("new", "newer")\n'
                   'TRAINING_FIX_COLUMNS: tuple = ()\n\ndef f():\n    return 1\n')
        self.assertTrue(surface.post_record_equivalent(BUILT, current))

    def test_anything_else_may_not(self):
        for label, current in (
                ('docstring', BUILT.replace('Doc.', 'Other.')),
                ('import', BUILT.replace('import sys', 'import os')),
                ('function body', BUILT.replace('return 1', 'return 2')),
                ('unlisted name', BUILT + 'CLASSICAL_RECORDED = ()\n'),
                ('tuple target', BUILT + 'TRAINING_GPU_COLUMNS, EXTRA = (), 1\n')):
            with self.subTest(change=label):
                self.assertFalse(surface.post_record_equivalent(BUILT, current))


class Differences(unittest.TestCase):
    def tree(self, tmp, manifest_text):
        root = Path(tmp)
        (root / MANIFEST).parent.mkdir(parents=True)
        (root / MANIFEST).write_text(manifest_text)
        (root / 'kernel.mojo').write_text('fn main(): pass\n')
        build = [['kernel.mojo', digest('fn main(): pass\n')], [MANIFEST, digest(BUILT)]]
        current = {'kernel.mojo': digest((root / 'kernel.mojo').read_text()), MANIFEST: digest(manifest_text)}
        return root, build, current

    def reader(self, text=BUILT):
        return lambda root, commit, rel: text.encode()

    def test_identical_tree_has_no_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, build, current = self.tree(tmp, BUILT)
            self.assertEqual(surface.post_record_differences(build, current, root, 'a' * 40, self.reader()), [])

    def test_record_list_edit_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, build, current = self.tree(tmp, BUILT.replace('"old"', '"new"'))
            self.assertEqual(surface.post_record_differences(build, current, root, 'a' * 40, self.reader()),
                             [MANIFEST])

    def test_refusals(self):
        cases = ('outside the lists', 'other file', 'added file', 'removed file', 'built copy digest', 'unreadable build')
        for label in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                root, build, current = self.tree(tmp, BUILT.replace('"old"', '"new"'))
                reader = self.reader()
                if label == 'outside the lists':
                    text = BUILT.replace('return 1', 'return 2')
                    (root / MANIFEST).write_text(text)
                    current[MANIFEST] = digest(text)
                elif label == 'other file':
                    current['kernel.mojo'] = digest('changed')
                elif label == 'added file':
                    current['new.mojo'] = digest('new')
                elif label == 'removed file':
                    del current['kernel.mojo']
                elif label == 'built copy digest':
                    reader = self.reader(BUILT + '\n')
                else:
                    def reader(root, commit, rel):
                        raise OSError('no such commit')
                with self.assertRaises(ValueError):
                    surface.post_record_differences(build, current, root, 'a' * 40, reader)

    def test_allowlist_is_exactly_the_manifest(self):
        self.assertEqual(surface.POST_RECORD_FILES, (MANIFEST,))
        self.assertEqual(surface.POST_RECORD_NAMES,
                         frozenset({'TRAINING_GPU_COLUMNS', 'TRAINING_FIX_COLUMNS', 'TRAINING_FIX_LANES'}))


if __name__ == '__main__':
    unittest.main()
