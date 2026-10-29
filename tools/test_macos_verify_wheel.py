#!/usr/bin/env python3
"""Exercise the release shell gate with isolated, synthetic interpreters."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'packaging/macos/verify_wheel.sh'


class WheelMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'packaging/macos/verify_wheel.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, self.script)
        self.dist = self.root / 'python/dist'
        self.dist.mkdir(parents=True)
        (self.dist / 'mojolearn-candidate.whl').touch()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('dirname', 'basename', 'mktemp', 'rm', 'mkdir', 'env'):
            (self.bin / name).symlink_to(shutil.which(name))
        self.calls = self.root / 'calls.log'
        self.env = dict(os.environ, PATH=str(self.bin), MOJOLEARN_RELEASE_MODES='fast identical',
                        WHEEL_TEST_CALLS=str(self.calls))

    def interpreters(self, versions=range(10, 15), *, venv_fail=False, smoke_fail=False,
                     smoke_output='', runtime_fail=False):
        pip_script = '#!/bin/sh\nexit 0\n'
        runtime_guard = 'case "$*" in *--installed*) exit 1;; esac\n' if runtime_fail else ''
        smoke_script = ("#!/bin/sh\n"
                        + 'printf "%s\\n" "$*" >> "$WHEEL_TEST_CALLS"\n'
                        + 'case "$*" in *compatibility_smoke.py*) [ "${WHEEL_TEST_COMPAT_FAIL:-0}" = 0 ] || exit 1;; esac\n'
                        + runtime_guard + "printf '%s\\n' " + shlex.quote(smoke_output)
                        + '\nexit ' + ('1' if smoke_fail else '0') + '\n')
        for version in versions:
            path = self.bin / ('python3.' + str(version))
            path.write_text('#!/bin/sh\n' + ('exit 1\n' if venv_fail else
                'mkdir -p "$3/bin"\n'
                + "printf '%s\\n' " + shlex.quote(pip_script) + ' > "$3/bin/pip"\n'
                + "printf '%s\\n' " + shlex.quote(smoke_script) + ' > "$3/bin/python"\n'
                + '/bin/chmod +x "$3/bin/pip" "$3/bin/python"\n'))
            path.chmod(0o755)

    def run_gate(self, *args):
        return subprocess.run(['/bin/sh', str(self.script), *args], env=self.env,
                              text=True, capture_output=True, timeout=15)

    def test_no_interpreters_cannot_pass(self):
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('0 interpreter(s) passed, 0 failed, 5 skipped', result.stdout)

    def test_partial_matrix_cannot_pass(self):
        self.interpreters([14])
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('1 interpreter(s) passed, 0 failed, 4 skipped', result.stdout)

    def test_failed_venvs_cannot_pass(self):
        self.interpreters(venv_fail=True)
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('5 skipped', result.stdout)

    def test_failed_smoke_cannot_pass(self):
        self.interpreters(smoke_fail=True)
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('5 failed', result.stdout)

    def test_full_matrix_and_no_gpu_label(self):
        self.interpreters()
        for args in ((), ('--no-gpu',)):
            result = self.run_gate(*args)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('PASS python3.'), 10)  # 5 interpreters x {fast, identical}
            self.assertIn('all 5 interpreters passed', result.stdout)
            if args:
                self.assertIn('DEVICE NOT TESTED', result.stdout.splitlines()[-1])

    def test_dependency_free_failure_cannot_be_hidden_by_oracle_success(self):
        self.interpreters(runtime_fail=True)
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count('dependency-free installed runtime'), 5)
        self.assertEqual(result.stdout.count('PASS python3.'), 10)  # 5 interpreters x {fast, identical}

    def test_numerical_suites_only_run_on_latest_minor(self):
        self.interpreters()
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(sum('compatibility_smoke.py' in c for c in calls), 8)
        self.assertEqual(sum('/macos/smoke.py' in c for c in calls), 2)
        self.assertEqual(sum('language_model_smoke.py' in c for c in calls), 1)
        runtime = [c for c in calls if 'check_numpy_free_runtime.py' in c]
        self.assertEqual(len(runtime), 5)
        self.assertEqual(sum('--no-gpu' in c for c in runtime), 4)

    def test_older_minor_compatibility_failure_blocks_release(self):
        self.interpreters()
        self.env['WHEEL_TEST_COMPAT_FAIL'] = '1'
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('1 interpreter(s) passed, 4 failed, 0 skipped', result.stdout)

    def test_ambiguous_wheel_is_refused(self):
        (self.dist / 'mojolearn-other.whl').touch()
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('multiple wheels', result.stderr)

    def test_empty_and_invalid_modes_are_refused(self):
        self.interpreters()
        for modes in ('   ', 'fast bogus', '*'):
            self.env['MOJOLEARN_RELEASE_MODES'] = modes
            result = self.run_gate()
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_unknown_arguments_are_refused(self):
        self.assertEqual(self.run_gate('--no-gp').returncode, 2)

    def test_evidence_backslashes_survive_success_and_failure(self):
        evidence = r'RESULT_JSON {"model":"tree\nleaf","path":"a\\b"}'
        for fails in (False, True):
            self.interpreters(smoke_fail=fails, smoke_output=evidence)
            result = self.run_gate()
            # Four minors: runtime + two compatibility jobs. Latest: runtime,
            # FAST/IDENTICAL ByteLM, language-model smoke and two full mode jobs.
            self.assertEqual(result.stdout.count(evidence), 18, result.stdout)
            self.assertEqual(result.returncode == 0, not fails)


if __name__ == '__main__':
    unittest.main()
