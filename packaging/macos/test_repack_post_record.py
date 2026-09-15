"""packaging/macos/repack_post_record.py on synthetic wheels: the recorded wheel's
binaries and every other member stay byte for byte, only the post-record members
change, and a manifest edit outside the record lists is refused."""
import csv
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location('repack_post_record', HERE / 'repack_post_record.py')
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)

MANIFEST_TEMPLATE = ('"""Manifest."""\n'
                     'TRAINING_GPU_COLUMNS = (\n{cols})\n\n'
                     'def training_gpu_column_record():\n'
                     '    return sorted({{c.rsplit("/", 2)[-2] for c in TRAINING_GPU_COLUMNS}})[0]\n\n'
                     'def wheel_bindings():\n    return ["_mojolearn_core_host"]\n')


def manifest(record):
    return MANIFEST_TEMPLATE.format(cols=f'    "bench/results/identity_break/{record}/apple-m4.json",\n')


class Repack(unittest.TestCase):
    def fixture(self, tmp):
        tmp = Path(tmp)
        recorded_dir = tmp / 'recorded'
        recorded_dir.mkdir()
        members = {
            'mojolearn/__init__.py': b'',
            'mojolearn/host_surface.py': manifest('old').encode(),
            'mojolearn/verify_reference/table.json': b'{"old": 1}',
            'mojolearn/identity_columns/old/apple-m4.json': b'{"column": "old"}',
            'mojolearn/identity_columns/COMMIT': b'a' * 40 + b'\n',
            'mojolearn/identical/_mojolearn.so': b'recorded binary bytes',
            'mojolearn/.dylibs/libKGENCompilerRTShared.dylib': b'recorded runtime bytes',
            'mojolearn-9.9.9.dist-info/METADATA': b'Name: mojolearn\n',
            'mojolearn-9.9.9.dist-info/RECORD': b'',
        }
        wheel = recorded_dir / 'mojolearn-9.9.9-py3-none-macosx_11_0_arm64.whl'
        with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        root = tmp / 'checkout'
        (root / 'tools').mkdir(parents=True)
        shutil.copy(REPO / 'tools' / 'verify_linux_surface_qualification.py', root / 'tools')
        shutil.copy(REPO / 'python' / 'mojolearn' / 'host_surface.py', tmp / 'real_host_surface.py')
        (root / 'python' / 'mojolearn' / 'verify_reference').mkdir(parents=True)
        (root / 'python' / 'mojolearn' / 'host_surface.py').write_text(manifest('new'))
        (root / 'python' / 'mojolearn' / 'verify_reference' / 'table.json').write_bytes(b'{"new": 2}')
        col = root / 'bench' / 'results' / 'identity_break' / 'new' / 'apple-m4.json'
        col.parent.mkdir(parents=True)
        col.write_bytes(b'{"column": "new"}')
        return wheel, root, tmp / 'out', members

    def test_only_post_record_members_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, root, out, members = self.fixture(tmp)
            result = repack.repack(wheel, out, root)
            with zipfile.ZipFile(result['wheel']) as z:
                names = set(z.namelist())
                self.assertEqual(z.read('mojolearn/identical/_mojolearn.so'), members['mojolearn/identical/_mojolearn.so'])
                self.assertEqual(z.read('mojolearn/.dylibs/libKGENCompilerRTShared.dylib'),
                                 members['mojolearn/.dylibs/libKGENCompilerRTShared.dylib'])
                self.assertEqual(z.read('mojolearn/identity_columns/COMMIT'), b'a' * 40 + b'\n')
                self.assertEqual(z.read('mojolearn/verify_reference/table.json'), b'{"new": 2}')
                self.assertIn('mojolearn/identity_columns/new/apple-m4.json', names)
                self.assertNotIn('mojolearn/identity_columns/old/apple-m4.json', names)
                rows = list(csv.reader(io.StringIO(z.read('mojolearn-9.9.9.dist-info/RECORD').decode())))
                listed = {r[0]: r for r in rows}
                for name in names:
                    self.assertIn(name, listed)
                for name in names - {'mojolearn-9.9.9.dist-info/RECORD'}:
                    self.assertEqual(listed[name], repack._record_row(name, z.read(name)))
            self.assertEqual(result['changed'], sorted(['mojolearn/host_surface.py', 'mojolearn/verify_reference/table.json',
                                                        'mojolearn/identity_columns/new/apple-m4.json']))
            self.assertEqual(result['removed'], ['mojolearn/identity_columns/old/apple-m4.json'])
            self.assertEqual(result['binaries_unchanged'], 2)

    def test_manifest_edit_outside_record_lists_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, root, out, _ = self.fixture(tmp)
            path = root / 'python' / 'mojolearn' / 'host_surface.py'
            path.write_text(path.read_text().replace('_mojolearn_core_host', '_mojolearn_other_host'))
            with self.assertRaises(SystemExit):
                repack.repack(wheel, out, root)
            self.assertFalse((out / wheel.name).exists())

    def test_missing_named_column_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, root, out, _ = self.fixture(tmp)
            (root / 'bench' / 'results' / 'identity_break' / 'new' / 'apple-m4.json').unlink()
            with self.assertRaises(SystemExit):
                repack.repack(wheel, out, root)

    def test_recorded_wheel_without_commit_witness_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, root, out, members = self.fixture(tmp)
            with zipfile.ZipFile(wheel, 'w') as archive:
                for name, data in members.items():
                    if name != 'mojolearn/identity_columns/COMMIT':
                        archive.writestr(name, data)
            with self.assertRaises(SystemExit):
                repack.repack(wheel, out, root)


if __name__ == '__main__':
    unittest.main()
