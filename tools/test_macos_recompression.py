import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('macos_shrink', Path(__file__).resolve().parents[1] / 'packaging/macos/shrink_wheel.py')
shrink = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shrink)


class RecompressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wheel = self.root / 'candidate.whl'
        self.tool = self.root / 'advzip'
        self.tool.write_bytes(b'test executable identity')
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name, data in [('package/module.so', b'native bytes' * 1000),
                               ('package.dist-info/RECORD', b'original RECORD')]:
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 1, 2, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        self.original = self.wheel.read_bytes()

    def run_candidate(self, mutate=None, failure=False):
        def recompress(argv, **kwargs):
            self.assertEqual(kwargs['timeout'], 1200)
            if failure:
                raise subprocess.CalledProcessError(1, argv)
            path = Path(argv[-1])
            with zipfile.ZipFile(path) as archive:
                rows = [(info, archive.read(info)) for info in archive.infolist()]
            with zipfile.ZipFile(path, 'w') as archive:
                for info, data in rows:
                    if mutate:
                        info, data = mutate(info, data)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data, compresslevel=9)
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(shrink.shutil, 'which', return_value=str(self.tool)), \
             patch.object(shrink.subprocess, 'check_output', return_value='test advzip'), \
             patch.object(shrink.subprocess, 'run', side_effect=recompress):
            shrink.recompress_deflate(self.wheel)

    def test_smaller_standard_zip_keeps_all_member_bytes_metadata_and_record(self):
        before = shrink.zip_witness(self.wheel)
        self.run_candidate()
        self.assertEqual(shrink.zip_witness(self.wheel), before)
        self.assertLess(self.wheel.stat().st_size, len(self.original))
        receipt = json.loads(self.wheel.with_suffix('.compression.json').read_text())
        self.assertTrue(receipt['complete'] and receipt['selected'])
        self.assertTrue(receipt['members_and_metadata_identical'])
        self.assertEqual(receipt['before_sha256'], hashlib.sha256(self.original).hexdigest())

    def test_changed_member_refused_and_original_retained(self):
        with self.assertRaisesRegex(ValueError, 'changed wheel members'):
            self.run_candidate(lambda info, data: (info, data + b'changed'))
        self.assertEqual(self.wheel.read_bytes(), self.original)

    def test_changed_record_refused(self):
        with self.assertRaisesRegex(ValueError, 'changed wheel members'):
            self.run_candidate(lambda info, data: (info, b'changed' if info.filename.endswith('RECORD') else data))
        self.assertEqual(self.wheel.read_bytes(), self.original)

    def test_changed_install_permissions_refused(self):
        def change(info, data):
            info.external_attr ^= 0o100 << 16
            return info, data
        with self.assertRaisesRegex(ValueError, 'install metadata'):
            self.run_candidate(change)
        self.assertEqual(self.wheel.read_bytes(), self.original)

    def test_failed_compressor_retains_original_and_receipt(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_candidate(failure=True)
        self.assertEqual(self.wheel.read_bytes(), self.original)
        self.assertFalse(json.loads(self.wheel.with_suffix('.compression.json').read_text())['complete'])
        self.assertFalse(self.wheel.with_suffix('.recompressing').exists())

    def test_absent_compressor_does_not_override_size_guard(self):
        with patch.object(shrink.shutil, 'which', return_value=None):
            shrink.recompress_deflate(self.wheel)
        self.assertEqual(self.wheel.read_bytes(), self.original)

    def test_nonstandard_zip_codec_refused(self):
        with zipfile.ZipFile(self.wheel, 'w', compression=zipfile.ZIP_BZIP2) as archive:
            archive.writestr('module', b'data')
        with self.assertRaisesRegex(ValueError, 'standard'):
            shrink.zip_witness(self.wheel)


if __name__ == '__main__':
    unittest.main()
