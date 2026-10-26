"""Published logical bindings retain their actual physical wheel ownership."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_reuse as rr


class PublishedBindings(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, name, files):
        path = self.root / name
        with zipfile.ZipFile(path, 'w') as archive:
            for key, value in files.items():
                archive.writestr(key, value)
        return path

    def record(self, rows, linux, mac=None):
        plan = dict(commit='a' * 40, runtime={}, rows=[dict(
            key=f'{target}/{path}', target=target, archive_path=path,
            decision='REUSE', reason='unchanged closure', identity={}, identity_digest='source')
            for target, path in rows])
        return rr.record_identities(plan, linux, mac, self.root / 'record.json')['rows']

    def test_platform_hosts_do_not_overwrite_each_other(self):
        path = 'mojolearn/host/_host.so'
        linux = self.wheel('linux.whl', {path: b'linux host'})
        mac = self.wheel('mac.whl', {path: b'apple host'})
        rows = self.record([(rr.LINUX, path), (rr.MACOS, path)], linux, mac)
        self.assertEqual([r['shipped_sha256'] for r in rows],
                         [hashlib.sha256(v).hexdigest() for v in (b'linux host', b'apple host')])
        self.assertEqual([r['shipped_wheel'] for r in rows], ['linux.whl', 'mac.whl'])

    def test_split_native_and_ptx_groups_record_physical_owners(self):
        wheels, wanted = [], []
        for directory, arch in [('cuda_native', 'sm_89'), ('cuda_ptx', 'sm_80')]:
            folder = f'mojolearn/{directory}/{arch}/identical'
            index = json.dumps(dict(format=1, files={'group.so': ['_first', '_second']}))
            wheels.append(self.wheel(arch + '.whl', {
                folder + '/group.so': arch.encode(), folder + '/_mojolearn_merged.json': index}))
            wanted.extend((rr.LINUX, f'mojolearn/cuda/{arch}/identical/{name}.so')
                          for name in ['_first', '_second'])
        rows = self.record(wanted, wheels)
        for row, arch in zip(rows, ['sm_89', 'sm_89', 'sm_80', 'sm_80']):
            self.assertEqual(row['shipped_sha256'], hashlib.sha256(arch.encode()).hexdigest())
            self.assertTrue(row['shipped_archive_path'].endswith('/group.so'))
            self.assertTrue(row['shipped_index']['archive_path'].endswith('/_mojolearn_merged.json'))

    def test_missing_ambiguous_and_unsafe_group_owners_are_refused(self):
        folder = 'mojolearn/host'
        cases = [({'absent.so': ['_host']}, {}),
                 ({'one.so': ['_host'], 'two.so': ['_host']}, {'one.so': b'1', 'two.so': b'2'}),
                 ({'../escape.so': ['_host']}, {})]
        for groups, binaries in cases:
            with self.subTest(groups=groups):
                files = {folder + '/' + key: value for key, value in binaries.items()}
                files[folder + '/_mojolearn_merged.json'] = json.dumps(dict(format=1, files=groups))
                wheel = self.wheel('bad.whl', files)
                with self.assertRaises(SystemExit):
                    self.record([(rr.LINUX, folder + '/_host.so')], wheel)

    def test_unpublished_platform_stays_missing(self):
        path = 'mojolearn/host/_host.so'
        wheel = self.wheel('mac.whl', {path: b'apple'})
        row = self.record([(rr.LINUX, path)], [], wheel)[0]
        self.assertIsNone(row['shipped_sha256'])


if __name__ == '__main__':
    unittest.main()
