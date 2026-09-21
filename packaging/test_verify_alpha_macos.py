"""A fresh native macOS build on the light route: admitted only with its smoke receipt.

Authored stdlib file fixtures; fake native members are never imported or executed.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_alpha_artifacts as gate

SOURCE = 'a' * 40


class FreshMacosTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        version_file = self.root / 'python/mojolearn/_version.py'
        version_file.parent.mkdir(parents=True)
        version_file.write_text('__version__ = "9.9.9"\n')
        self.version = gate.release_version(self.root)
        self.name = 'mojolearn-' + self.version + '-py3-none-macosx_11_0_arm64.whl'
        self.wheel = self.dist / self.name
        self.prefix = 'mojolearn-' + self.version + '.dist-info/'
        self.members = {
            self.prefix + 'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-macosx_11_0_arm64\n',
            self.prefix + 'METADATA': ('Metadata-Version: 2.1\nName: mojolearn\nVersion: ' + self.version
                                       + '\nClassifier: Development Status :: 3 - Alpha\n\nFixture only\n').encode(),
            self.prefix + 'portable-math.json': b'{}',
            'mojolearn/_version.py': ('# header\n__version__ = "' + self.version + '"\n').encode(),
            'mojolearn/identity_columns/COMMIT': (SOURCE + '\n').encode(),
            'mojolearn/identical/_mojolearn.so': b'FAKE BYTES, NEVER EXECUTE',
        }

    def stage(self, smoke=True, source=SOURCE):
        rows = [[name, gate.record_hash(hashlib.sha256(raw).digest()), str(len(raw))]
                for name, raw in self.members.items()]
        record = self.prefix + 'RECORD'
        rows.append([record, '', ''])
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name, raw in self.members.items():
                archive.writestr(name, raw)
            archive.writestr(record, stream.getvalue())
        manifest = dict(schema='mojolearn.alpha-release.v1', version=self.version, release_profile='alpha-api',
                        files={self.name: gate.wheel_digest(self.wheel)})
        if smoke:
            receipt = self.dist / 'light-smoke-macos.json'
            receipt.write_text('{"status": "PASSED"}')
            manifest['light_smoke'] = dict(source_commit=source,
                                           receipts={receipt.name: gate.wheel_digest(receipt)})
        raw = json.dumps(manifest).encode()
        (self.dist / 'alpha-manifest.json').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def refused(self, digest, words):
        with self.assertRaises(ValueError) as caught:
            gate.verify(self.dist, digest, None, self.root)
        self.assertIn(words, str(caught.exception))

    def test_fresh_build_with_its_smoke_receipt_is_admitted(self):
        result = gate.verify(self.dist, self.stage(), None, self.root)
        self.assertTrue(result['passed'])

    def test_without_a_macos_smoke_receipt_it_is_refused(self):
        self.refused(self.stage(smoke=False), 'light-smoke-macos.json')

    def test_a_receipt_from_another_source_is_refused(self):
        self.refused(self.stage(source='b' * 40), 'source witness differs')

    def test_missing_platform_math_record_is_refused(self):
        del self.members[self.prefix + 'portable-math.json']
        self.refused(self.stage(), 'platform-math audit record')

    def test_no_identical_native_binaries_is_refused(self):
        del self.members['mojolearn/identical/_mojolearn.so']
        self.members['mojolearn/_mojolearn.so'] = b'FAKE'
        self.refused(self.stage(), 'no identical-mode native binaries')

    def test_shipped_tests_are_refused(self):
        self.members['mojolearn/tests/test_x.py'] = b'pass\n'
        self.refused(self.stage(), 'test/cache artifacts')


if __name__ == '__main__':
    unittest.main()
