"""Authored file-only fixtures. Root executes; never imports native/package code."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('alpha_overlay', Path(__file__).with_name('alpha_overlay.py'))
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


class AlphaOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = self.root / 'python'
        package = self.python / 'mojolearn'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('from ._version import __version__\n')
        (package / '_backend.py').write_text("_MODULES = ('_mojolearn', '_mojolearn_byte_lm')\n")
        self.dist = 'mojolearn-0.6.0.dist-info/'
        self.wheel = self.root / 'mojolearn-0.6.0-py3-none-manylinux_2_28_x86_64.whl'
        self.files = {
            'mojolearn/__init__.py': b'# old API\n',
            'mojolearn/old_only.py': b'# stale module\n',
            'mojolearn/hip/identical/_mojolearn.so': b'FAKE NATIVE BYTES; NEVER EXECUTE',
            'mojolearn/hip/lib/libamdhip64.so.6': b'FAKE RUNTIME BYTES',
            self.dist + 'METADATA': b'Metadata-Version: 2.1\nName: mojolearn\nVersion: 0.6.0\nClassifier: Development Status :: 5 - Production/Stable\n\nOld description\n',
            self.dist + 'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-manylinux_2_28_x86_64\n',
        }

    def write(self, corrupt=False):
        rows = [(name, overlay.record_hash(hashlib.sha256(raw).digest()), str(len(raw)))
                for name, raw in self.files.items()]
        record = self.dist + 'RECORD'
        rows.append((record, '', ''))
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name, raw in self.files.items():
                archive.writestr(name, raw + (b'corruption' if corrupt and name.endswith('.so') else b''))
            archive.writestr(record, buffer.getvalue())

    def test_native_bytes_tags_version_and_missing_module_provenance(self):
        self.write()
        result = overlay.assemble(self.wheel, self.python, '0.6.0a1', self.root / 'out')
        with zipfile.ZipFile(result) as archive:
            for name, raw in self.files.items():
                if '.so' in name:
                    self.assertEqual(archive.read(name), raw)
            new_dist = 'mojolearn-0.6.0a1.dist-info/'
            self.assertEqual(archive.read(new_dist + 'WHEEL'), self.files[self.dist + 'WHEEL'])
            self.assertNotIn('mojolearn/old_only.py', archive.namelist())
            self.assertIn(b'0.6.0a1', archive.read('mojolearn/_version.py'))
            self.assertIn(b'Development Status :: 3 - Alpha', archive.read(new_dist + 'METADATA'))
            provenance = json.loads(archive.read(new_dist + 'ALPHA_PROVENANCE.json'))
            self.assertEqual(provenance['missing_optional_native_modules_by_present_directory'],
                             {'mojolearn/hip/identical': ['_mojolearn_byte_lm']})
            for name, digest, size in csv.reader(io.StringIO(archive.read(new_dist + 'RECORD').decode())):
                if name.endswith('/RECORD'):
                    self.assertEqual((digest, size), ('', ''))
                else:
                    raw = archive.read(name)
                    self.assertEqual((digest, size), (overlay.record_hash(hashlib.sha256(raw).digest()), str(len(raw))))

    def test_changed_native_record_refused_before_output(self):
        self.write(corrupt=True)
        with self.assertRaisesRegex(ValueError, 'RECORD hash'):
            overlay.assemble(self.wheel, self.python, '0.6.0a1', self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_doc_diagnostics_overlay_and_test_cache_exclusion(self):
        package = self.python / 'mojolearn'
        (package / 'ALPHA_API.md').write_bytes(b'Current alpha feature limits\n')
        (self.python / 'mojolearn_diagnostics.py').write_bytes(b'# current diagnostics\n')
        (package / 'tests').mkdir()
        (package / 'tests' / 'test_hidden.py').write_bytes(b'# should not ship\n')
        (package / '__pycache__').mkdir()
        (package / '__pycache__' / 'hidden.py').write_bytes(b'# should not ship\n')
        self.files['mojolearn/ALPHA_API.md'] = b'Old alpha limits\n'
        self.files['mojolearn/retained.md'] = b'Preserved base documentation\n'
        self.files['mojolearn_diagnostics.py'] = b'# stale diagnostics\n'
        self.files['mojolearn/tests/old_test.py'] = b'# old test\n'
        self.files['mojolearn/__pycache__/old.pyc'] = b'old cache'
        self.write()
        result = overlay.assemble(self.wheel, self.python, '0.6.0a1', self.root / 'out')
        with zipfile.ZipFile(result) as archive:
            self.assertEqual(archive.read('mojolearn/ALPHA_API.md'), b'Current alpha feature limits\n')
            self.assertEqual(archive.read('mojolearn_diagnostics.py'), b'# current diagnostics\n')
            self.assertEqual(archive.read('mojolearn/retained.md'), self.files['mojolearn/retained.md'])
            self.assertFalse(any('/tests/' in name or '/__pycache__/' in name for name in archive.namelist()))
            provenance = json.loads(archive.read('mojolearn-0.6.0a1.dist-info/ALPHA_PROVENANCE.json'))
            self.assertNotIn('mojolearn/ALPHA_API.md', provenance['inherited_non_python_sha256'])
            self.assertIn('mojolearn/ALPHA_API.md', provenance['documentation_source_sha256'])
            self.assertIn('mojolearn_diagnostics.py', provenance['python_source_sha256'])

    def test_stable_version_refused(self):
        self.write()
        with self.assertRaisesRegex(ValueError, 'explicit alpha'):
            overlay.assemble(self.wheel, self.python, '0.6.0', self.root / 'out')

    def test_utf8_metadata_description_preserved(self):
        description = 'GPU naïve Bayes — données, λ and 日本語.\n'.encode('utf-8')
        self.files[self.dist + 'METADATA'] = (
            b'Metadata-Version: 2.1\nName: mojolearn\nVersion: 0.6.0\n'
            b'Description-Content-Type: text/markdown; charset=UTF-8\n'
            b'Classifier: Development Status :: 5 - Production/Stable\n\n'
            + description)
        self.write()
        result = overlay.assemble(self.wheel, self.python, '0.6.0a1', self.root / 'out')
        with zipfile.ZipFile(result) as archive:
            metadata = archive.read('mojolearn-0.6.0a1.dist-info/METADATA')
            self.assertTrue(metadata.endswith(description))
            self.assertIn(b'Version: 0.6.0a1\n', metadata)
            self.assertIn(b'Classifier: Development Status :: 3 - Alpha\n', metadata)
            self.assertNotIn(b'Development Status :: 5 - Production/Stable', metadata)
            self.assertIn(overlay.NOTICE.encode('ascii'), metadata)

    def test_traversal_refused(self):
        self.files['../escape'] = b'bad'
        self.write()
        with self.assertRaisesRegex(ValueError, 'unsafe wheel path'):
            overlay.assemble(self.wheel, self.python, '0.6.0a1', self.root / 'out')


if __name__ == '__main__':
    unittest.main()
