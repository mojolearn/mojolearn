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
        # DEVIATION 2290: the fixture root declares its own version and every
        # name below is derived through the shared reader; no number is pinned.
        version_file = self.root / 'python/mojolearn/_version.py'
        version_file.parent.mkdir(parents=True)
        version_file.write_text('__version__ = "9.9.9"\n')
        self.version = gate.release_version(self.root)
        self.name = 'mojolearn-' + self.version + '-py3-none-manylinux_2_28_x86_64.whl'
        self.wheel = self.dist / self.name
        self.qual = self.root / 'linux-qualification.tar.gz'
        self.prefix = 'mojolearn-' + self.version + '.dist-info/'
        self.members = {
            self.prefix + 'WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-manylinux_2_28_x86_64\n',
            self.prefix + 'METADATA': ('Metadata-Version: 2.1\nName: mojolearn\nVersion: ' + self.version
                                       + '\nClassifier: Development Status :: 3 - Alpha\n\nFixture only\n').encode(),
            self.prefix + 'LINUX_PAYLOAD.json': self.payload(gate.RELEASE_PROFILE),
            'mojolearn/cuda/sm_89/identical/_mojolearn.so': b'FAKE BYTES, NEVER EXECUTE',
        }
        with tarfile.open(self.qual, 'w:gz') as archive:
            data = b'fixture: mocked checker admission, no numerical proof'
            info = tarfile.TarInfo('build-proofs/fixture.json')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    def payload(self, assembly_profile):
        return json.dumps(dict(schema='mojolearn.linux-payload.v1', version=self.version,
                               release_profile='alpha-api', assembly_profile=assembly_profile)).encode()

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
        manifest = dict(schema='mojolearn.alpha-release.v1', version=self.version, release_profile='alpha-api',
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

    def test_missing_qualification_admits_fresh_wheel_as_unqualified(self):
        # Alpha policy, 2026-09-09: the archive is optional; file checks still run.
        digest = self.stage(qualification=False)
        gate.verify(self.dist, digest, None, self.root)

    def test_deprecated_profile_alias_in_payload_still_admits(self):
        # DEVIATION 2290: a payload written under `release-0.6.1` parses as release-linux3.
        for profile in sorted(gate.RELEASE_PROFILES):
            with self.subTest(profile=profile):
                self.members[self.prefix + 'LINUX_PAYLOAD.json'] = self.payload(profile)
                digest = self.stage()
                checker = Mock(return_value=dict(status='PASSED', wheel_sha256=gate.wheel_digest(self.wheel),
                               runtime_coverage={key: 'fixture' for key in ('cuda/sm_89', 'cuda/sm_90', 'hip/gfx942')}))
                result = self.check_with_mock(digest, checker)
                self.assertTrue(result['passed'])
                self.assertEqual(result['version'], self.version)

    def test_payload_of_another_version_refused(self):
        # DEVIATION 2290: the wheel's version must be the one the source root declares.
        (self.root / 'python/mojolearn/_version.py').write_text('__version__ = "9.9.10"\n')
        digest = self.stage()
        with self.assertRaises(Exception):
            self.check_with_mock(digest, Mock())

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
            gate.verify(self.dist, digest, None, self.root)  # DEVIATION 2290: the root names the version

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



class SplitLinuxTests(unittest.TestCase):
    """THE SPLIT LINUX PACKAGES: mojolearn (core), mojolearn_nvidia, mojolearn_amd,
    one wheel per manifest (each package publishes on its own)."""
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
        self.plugins = gate.GPU_PLUGINS

    def members(self, kind):
        v, P = self.version, self.plugins
        tag = 'py3-none-manylinux_2_35_x86_64'
        if kind == 'core':
            wheel_name, distribution, vendor = 'mojolearn', 'mojolearn', None
        else:
            vendor = P.by_profile(kind)
            wheel_name, distribution = P.PLUGINS[vendor]['wheel_name'], P.PLUGINS[vendor]['distribution']
        prefix = wheel_name + '-' + v + '.dist-info/'
        meta = ['Metadata-Version: 2.4', 'Name: ' + distribution, 'Version: ' + v,
                'Classifier: Development Status :: 3 - Alpha']
        payload = dict(schema='mojolearn.linux-payload.v1', version=v, release_profile='alpha-api',
                       assembly_profile='release-split', source_commit='a' * 40,
                       split=dict(role=P.CORE_PROFILE if vendor is None else kind, distribution=distribution))
        members = {prefix + 'WHEEL': ('Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: ' + tag + '\n').encode(),
                   prefix + 'LINUX_PAYLOAD.json': json.dumps(payload).encode()}
        if vendor is None:
            # the core declares no extras and requires BOTH plugins at its
            # own version exactly (`pip install mojolearn` works for everyone)
            meta.append('Requires-Dist: numpy>=1.24')
            meta.extend('Requires-Dist: ' + r for r in P.core_requirements(v))
            members[prefix + P.CORE_MARKER] = json.dumps(P.core_marker(v)).encode()
            members['mojolearn/__init__.py'] = b'# fixture\n'
            members['mojolearn/identity_columns/COMMIT'] = b'a' * 40 + b'\n'
            members['mojolearn/.libs/libfixture.so'] = b'FAKE BYTES, NEVER EXECUTE'
        else:
            meta.append('Requires-Dist: mojolearn==' + v)
            arch = 'sm_89' if vendor == 'cuda' else 'gfx942'
            members[prefix + P.PLUGIN_MARKER] = json.dumps(P.plugin_marker(vendor, v, [arch])).encode()
            members[f'mojolearn/{vendor}/{arch}/identical/_mojolearn_knn.so'] = b'FAKE BYTES, NEVER EXECUTE'
        members[prefix + 'METADATA'] = ('\n'.join(meta) + '\n\nFixture only\n').encode()
        return wheel_name + '-' + v + '-' + tag + '.whl', prefix, members

    def stage(self, kind, mutate=None):
        for f in self.dist.iterdir():
            f.unlink()
        name, prefix, members = self.members(kind)
        if mutate:
            mutate(prefix, members)
        rows = [[n, gate.record_hash(hashlib.sha256(raw).digest()), str(len(raw))] for n, raw in members.items()]
        rows.append([prefix + 'RECORD', '', ''])
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        with zipfile.ZipFile(self.dist / name, 'w') as archive:
            for n, raw in members.items():
                archive.writestr(n, raw)
            archive.writestr(prefix + 'RECORD', stream.getvalue())
        manifest = dict(schema='mojolearn.alpha-release.v1', version=self.version, release_profile='alpha-api',
                        files={name: gate.wheel_digest(self.dist / name)})
        raw = json.dumps(manifest).encode()
        (self.dist / 'alpha-manifest.json').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_each_split_wheel_admits_alone(self):
        for kind in ('core', 'nvidia', 'amd'):
            with self.subTest(kind=kind):
                result = gate.verify(self.dist, self.stage(kind), None, self.root)
                self.assertTrue(result['passed'])

    def test_plugin_pin_and_ownership_are_checked(self):
        def loose_pin(prefix, members):
            members[prefix + 'METADATA'] = members[prefix + 'METADATA'].replace(b'mojolearn==9.9.9', b'mojolearn>=9.9.9')

        def python_in_plugin(prefix, members):
            members['mojolearn/cuda/helper.py'] = b'# no Python in a plugin\n'

        def other_vendor(prefix, members):
            members['mojolearn/hip/gfx942/_mojolearn_knn.so'] = b'FAKE'

        def wrong_marker(prefix, members):
            members[prefix + gate.GPU_PLUGINS.PLUGIN_MARKER] = json.dumps(
                gate.GPU_PLUGINS.plugin_marker('cuda', self.version, ['sm_90a'])).encode()

        def combined_profile(prefix, members):
            doc = json.loads(members[prefix + 'LINUX_PAYLOAD.json'])
            doc['assembly_profile'] = gate.RELEASE_PROFILE
            members[prefix + 'LINUX_PAYLOAD.json'] = json.dumps(doc).encode()
        for mutate, why in ((loose_pin, 'exactly mojolearn=='), (python_in_plugin, 'no Python'),
                            (other_vendor, 'outside mojolearn/cuda/'), (wrong_marker, 'marker'),
                            (combined_profile, 'profile/role')):
            with self.subTest(mutate=mutate.__name__):
                digest = self.stage('nvidia', mutate)
                with self.assertRaisesRegex(ValueError, why):
                    gate.verify(self.dist, digest, None, self.root)

    def test_core_requirements_and_ownership_are_checked(self):
        def edit(old, new):
            def mutate(prefix, members):
                raw = members[prefix + 'METADATA']
                assert old in raw, old
                members[prefix + 'METADATA'] = raw.replace(old, new, 1)
            return mutate
        amd = b'Requires-Dist: mojolearn-amd==9.9.9\n'
        nvidia = b'Requires-Dist: mojolearn-nvidia==9.9.9\n'
        gpu_extra = edit(b'\n\nFixture only', b'\nProvides-Extra: nvidia\n\nFixture only')
        missing_amd = edit(amd, b'')
        missing_nvidia = edit(nvidia, b'')
        missing_both = edit(nvidia + amd, b'')
        loose_pin = edit(amd, b'Requires-Dist: mojolearn-amd>=9.9.9\n')
        other_version = edit(nvidia, b'Requires-Dist: mojolearn-nvidia==9.9.8\n')
        extra_marker = edit(amd, b'Requires-Dist: mojolearn-amd==9.9.9; extra == "amd"\n')
        doubled = edit(amd, amd + amd)

        def gpu_set_in_core(prefix, members):
            members['mojolearn/cuda/sm_89/_mojolearn_knn.so'] = b'FAKE'
        for mutate, why in ((gpu_extra, 'no extras'), (missing_amd, 'require exactly'),
                            (missing_nvidia, 'require exactly'), (missing_both, 'require exactly'),
                            (loose_pin, 'require exactly'), (other_version, 'require exactly'),
                            (extra_marker, 'require exactly'), (doubled, 'require exactly'),
                            (gpu_set_in_core, 'GPU set member')):
            with self.subTest(mutate=mutate.__name__):
                digest = self.stage('core', mutate)
                with self.assertRaisesRegex(ValueError, why):
                    gate.verify(self.dist, digest, None, self.root)

    def test_a_split_wheel_of_another_version_is_refused(self):
        digest = self.stage('amd')
        (self.root / 'python/mojolearn/_version.py').write_text('__version__ = "9.9.10"\n')
        with self.assertRaises(ValueError):
            gate.verify(self.dist, digest, None, self.root)

    def test_unknown_distribution_prefix_is_refused(self):
        digest = self.stage('nvidia')
        name = next(p for p in self.dist.iterdir() if p.suffix == '.whl')
        name.rename(name.with_name(name.name.replace('mojolearn_nvidia', 'mojolearn_vulkan')))
        manifest = json.loads((self.dist / 'alpha-manifest.json').read_text())
        manifest['files'] = {name.name.replace('mojolearn_nvidia', 'mojolearn_vulkan'): list(manifest['files'].values())[0]}
        raw = json.dumps(manifest).encode()
        (self.dist / 'alpha-manifest.json').write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'filename'):
            gate.verify(self.dist, hashlib.sha256(raw).hexdigest(), None, self.root)
        self.assertTrue(digest)

if __name__ == '__main__':
    unittest.main()
