"""Small artifact controls; no native code or dependency installation."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml

from qualify_umap_wheel import sha
from verify_umap_qualification import verify


class QualificationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qualification artifact ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = self.root / 'candidate.whl'
        self.results = self.root / 'results.json'
        sources = {}
        for relative in ('python/mojolearn/_umap_impl.py',
                         'python/mojolearn/tests/test_umap_surface.py',
                         'python/mojolearn/tests/test_umap_transform.py',
                         'tools/umap_transform_quality_check.py'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture ' + relative)
            sources[relative] = sha(path)
        jobs = [{'name': name, 'exit_code': 0, 'timed_out': False} for name in
                ('create-venv', 'install-wheel', 'check-dependencies', 'installed-packages')]
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            archive.write(self.root / 'python/mojolearn/_umap_impl.py', 'mojolearn/_umap_impl.py')
            for mode, code in {'fast': 0, 'deterministic': 2, 'identical': 1}.items():
                binary = ('fixture binary ' + mode).encode()
                member = ('mojolearn/' + (mode + '/' if mode != 'fast' else '') +
                          '_mojolearn_metrics.so')
                archive.writestr(member, binary)
                for surface in ('fit', 'transform', 'quality'):
                    jobs.append({'name': f'{surface}-{mode}', 'exit_code': 0,
                                 'timed_out': False, 'installed': {
                                     'version': '0.6.0', 'mode': mode, 'binding_mode_code': code,
                                     'wrapper_sha256': sources['python/mojolearn/_umap_impl.py'],
                                     'binding_sha256': hashlib.sha256(binary).hexdigest()}})
        self.record = {'status': 'PASSED', 'wheel_sha256': sha(self.wheel),
                       'expected_version': '0.6.0', 'source_files': sources, 'jobs': jobs}

    def check(self):
        self.results.write_text(json.dumps(self.record))
        return verify(self.wheel, self.results, self.root, '0.6.0')

    def test_exact_candidate_passes_without_installed_venv(self):
        self.assertEqual(self.check(), sha(self.wheel))

    def test_replaced_wheel_fails_even_with_same_filename(self):
        with self.wheel.open('ab') as handle:
            handle.write(b'changed since qualification')
        with self.assertRaisesRegex(ValueError, 'differs from the qualified wheel'):
            self.check()

    def test_nonpassing_statuses_are_refused(self):
        for status in ('FAILED', 'INCOMPLETE', None):
            with self.subTest(status=status):
                self.record['status'] = status
                with self.assertRaisesRegex(ValueError, 'did not pass'):
                    self.check()

    def test_missing_and_duplicate_jobs_are_refused(self):
        self.record['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'jobs are missing'):
            self.check()
        self.record['jobs'].append(self.record['jobs'][-1])
        with self.assertRaisesRegex(ValueError, 'jobs are missing'):
            self.check()

    def test_failed_or_timed_out_job_cannot_hide_under_passed_status(self):
        for field, value in (('exit_code', 1), ('timed_out', True)):
            job = self.record['jobs'][0]
            original = job[field]
            with self.subTest(field=field):
                job[field] = value
                with self.assertRaisesRegex(ValueError, 'job did not succeed'):
                    self.check()
            job[field] = original

    def test_changed_qualification_source_is_refused(self):
        (self.root / 'python/mojolearn/tests/test_umap_surface.py').write_text('different test')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.check()

    def test_stale_installed_binding_wrapper_version_and_mode_are_refused(self):
        installed = self.record['jobs'][-1]['installed']
        for field in ('binding_sha256', 'wrapper_sha256', 'version', 'mode', 'binding_mode_code'):
            original = installed[field]
            with self.subTest(field=field):
                installed[field] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'artifact evidence differs'):
                    self.check()
            installed[field] = original

    def test_wrong_release_version_is_refused(self):
        self.record['expected_version'] = '0.5.0'
        with self.assertRaisesRegex(ValueError, 'release version'):
            self.check()

    def test_workflow_digest_refuses_replacement_before_recording_upload_hash(self):
        repo = Path(__file__).resolve().parents[1]
        steps = yaml.safe_load((repo / '.github/workflows/release-provenance.yml').read_text())['jobs']['build']['steps']
        gate = next(i for i, step in enumerate(steps) if step.get('name') ==
                    'Verify qualified UMAP artifact before recording upload digests')
        self.assertEqual(steps[gate + 1]['id'], 'digest')
        self.assertNotIn('if', steps[gate])
        self.assertFalse(steps[gate].get('continue-on-error', False))
        command = steps[gate]['run'] + '\n' + steps[gate + 1]['run']
        dist = self.root / 'python/dist'
        dist.mkdir()
        self.wheel = self.wheel.rename(dist / 'mojolearn-0.6.0-py3-none-macosx_11_0_arm64.whl')
        output = self.root / 'umap-qualification-123-1'
        output.mkdir()
        self.results = output / 'results.json'
        self.results.write_text(json.dumps(self.record))
        (self.root / 'python/mojolearn/_version.py').write_text('__version__ = "0.6.0"\n')
        for name in ('verify_umap_qualification.py', 'qualify_umap_wheel.py'):
            (self.root / 'tools' / name).symlink_to(repo / 'tools' / name)
        bindir = self.root / 'bin'
        bindir.mkdir()
        python = bindir / 'python3'
        python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python.chmod(0o755)
        github_output = self.root / 'github-output'
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                   GITHUB_WORKSPACE=str(self.root), RUNNER_TEMP=str(self.root),
                   GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1', GITHUB_OUTPUT=str(github_output))
        for changed in (False, True):
            with self.subTest(changed=changed):
                if changed:
                    github_output.unlink()
                    with self.wheel.open('ab') as handle:
                        handle.write(b'replaced after qualification')
                result = subprocess.run(['bash', '-e', '-c', command], cwd=self.root,
                                        env=env, capture_output=True, text=True, timeout=10)
                if changed:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('differs from the qualified wheel', result.stderr)
                    self.assertFalse(github_output.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('wheel_sha256=' + sha(self.wheel), github_output.read_text())


if __name__ == '__main__':
    unittest.main()
