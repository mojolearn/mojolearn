"""Authored stdlib file fixtures; checker mocks exercise wiring, not qualification.

Root alone executes. Fake native members must never be imported or executed.
Actual installed evidence validation belongs to check_linux_release_qualification.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_alpha_artifacts as gate


class CombinedLinuxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        self.name = 'mojolearn-0.6.1-py3-none-manylinux_2_28_x86_64.whl'
        self.wheel = self.dist / self.name
        self.qual = self.root / 'linux-qualification.tar.gz'
        self.prefix = 'mojolearn-0.6.1.dist-info/'
        self.members = {
            self.prefix + 'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-manylinux_2_28_x86_64\n',
            self.prefix + 'METADATA': b'Metadata-Version: 2.1\nName: mojolearn\nVersion: 0.6.1\nClassifier: Development Status :: 3 - Alpha\n\nFixture only\n',
            self.prefix + 'LINUX_PAYLOAD.json': json.dumps(dict(schema='mojolearn.linux-payload.v1',
                version='0.6.1', release_profile='alpha-api', assembly_profile='release-0.6.1')).encode(),
            'mojolearn/cuda/sm_89/identical/_mojolearn.so': b'FAKE BYTES, NEVER EXECUTE',
        }
        with tarfile.open(self.qual, 'w:gz') as archive:
            data = b'fixture: mocked checker admission, no numerical proof'
            info = tarfile.TarInfo('build-proofs/fixture.json')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    def stage(self, qualification=True):
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
        manifest = dict(schema='mojolearn.alpha-release.v1', version='0.6.1', release_profile='alpha-api',
                        files={self.name: gate.wheel_digest(self.wheel)})
        if qualification:
            manifest['linux_qualification'] = dict(file=self.qual.name, sha256=gate.wheel_digest(self.qual), wheel=self.name)
        raw = json.dumps(manifest).encode()
        (self.dist / 'alpha-manifest.json').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def check_with_mock(self, digest, checker):
        module = types.SimpleNamespace(check_release061=checker)
        with patch.dict(sys.modules, {'check_linux_release_qualification': module}):
            return gate.verify(self.dist, digest, self.qual, self.root)

    def test_full_checker_required_and_exact_wheel_passed(self):
        digest = self.stage()
        checker = Mock(return_value=dict(status='PASSED', wheel_sha256=gate.wheel_digest(self.wheel),
                       runtime_coverage={key: 'fixture' for key in ('cuda/sm_89', 'cuda/sm_90', 'hip/gfx942')}))
        result = self.check_with_mock(digest, checker)
        self.assertTrue(result['passed'])
        self.assertEqual(checker.call_count, 1)
        self.assertEqual(checker.call_args.args[0], self.wheel)
        self.assertEqual(checker.call_args.args[2], self.root)

    def test_missing_qualification_cannot_admit_fresh_wheel(self):
        digest = self.stage(qualification=False)
        with self.assertRaises(Exception):
            gate.verify(self.dist, digest)

    def test_checker_refusal_propagates(self):
        digest = self.stage()
        with self.assertRaisesRegex(ValueError, 'runtime missing'):
            self.check_with_mock(digest, Mock(side_effect=ValueError('runtime missing')))

    def test_wrong_final_wheel_digest_refused(self):
        digest = self.stage()
        checker = Mock(return_value=dict(status='PASSED', wheel_sha256='0' * 64,
                       runtime_coverage={key: 'fixture' for key in ('cuda/sm_89', 'cuda/sm_90', 'hip/gfx942')}))
        with self.assertRaises(Exception):
            self.check_with_mock(digest, checker)

    def test_old_overlay_cannot_replace_new_linux_payload(self):
        del self.members[self.prefix + 'LINUX_PAYLOAD.json']
        digest = self.stage(qualification=False)
        with self.assertRaises(Exception):
            gate.verify(self.dist, digest)

    def test_changed_archive_refused_before_checker(self):
        digest = self.stage()
        with self.qual.open('ab') as stream:
            stream.write(b'changed')
        checker = Mock()
        with self.assertRaises(Exception):
            self.check_with_mock(digest, checker)
        checker.assert_not_called()

    def test_archive_symlink_refused(self):
        with tarfile.open(self.qual, 'w:gz') as archive:
            info = tarfile.TarInfo('escape')
            info.type = tarfile.SYMTYPE
            info.linkname = '/tmp'
            archive.addfile(info)
        digest = self.stage()
        with self.assertRaises(Exception):
            self.check_with_mock(digest, Mock())


if __name__ == '__main__':
    unittest.main()
