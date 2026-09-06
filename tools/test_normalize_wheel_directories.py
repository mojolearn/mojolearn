import tempfile
from pathlib import Path
import unittest
import zipfile
from normalize_wheel_directories import normalize


class NormalizationTests(unittest.TestCase):
    def test_preserves_all_payload_and_record_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'original.whl', Path(tmp)/'normalized.whl'
            with zipfile.ZipFile(source, 'w') as z:
                z.writestr('pkg/', b'')
                z.writestr('pkg/native.so', b'\x7fELF\x00test')
                z.writestr('pkg.dist-info/RECORD', b'fixed-record\n')
            result = normalize(source, target)
            self.assertEqual(result['removed_empty_directories'], ['pkg/'])
            with zipfile.ZipFile(source) as a, zipfile.ZipFile(target) as b:
                self.assertEqual(set(b.namelist()), {'pkg/native.so', 'pkg.dist-info/RECORD'})
                for name in b.namelist(): self.assertEqual(a.read(name), b.read(name))
            with self.assertRaises(ValueError): normalize(source, target)

    def test_nonempty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'original.whl'
            with zipfile.ZipFile(source, 'w') as z: z.writestr('pkg/', b'payload')
            with self.assertRaises(ValueError): normalize(source, Path(tmp)/'new.whl')


if __name__ == '__main__': unittest.main()
