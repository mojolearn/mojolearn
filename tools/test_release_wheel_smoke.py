"""tools/release_wheel_smoke.sh without a rental: the dry run, its refusals,
the transfer deadline, and the whole --ssh path against a local ssh shim whose
`timeout` stands in for the smoke. Nothing here reaches RunPod except the dry
run's free pod listing, which is skipped when no key is configured."""
import json
import os
import pathlib
import subprocess
import tempfile
import time
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'tools/release_wheel_smoke.sh'
COMMIT = 'c' * 40


def make_wheel(directory, commit=COMMIT, name='mojolearn-0.0.0-py3-none-manylinux_2_35_x86_64.whl'):
    path = pathlib.Path(directory) / name
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mojolearn/identity_columns/COMMIT', commit + '\n')
        z.writestr('mojolearn/verify_reference/models/models.json', '{"models": []}')
    return path


SSH_SHIM = '#!/bin/bash\n# every call passes the remote command as its last argument\nexec bash -c "${@: -1}"\n'
SHA_SHIM = '#!/bin/sh\nexec shasum -a 256 "$@"\n'
# The stand-in smoke: `timeout -k 20 N PY qualify.py WHEEL --scope ... --output OUT`
TIMEOUT_SHIM = r'''#!/usr/bin/env python3
import hashlib, json, pathlib, sys
args = sys.argv[4:]
wheel = pathlib.Path(args[2])
out = pathlib.Path(args[args.index('--output') + 1]); out.mkdir(parents=True)
commit = args[args.index('--expected-source-commit') + 1]
status = pathlib.Path(__file__).with_name('STATUS').read_text().strip()
json.dump(dict(wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
               status=status, scope='expanded', source_commit=commit, jobs=[{'name': 'x'}],
               installed=dict(vendor='cuda')), open(out / 'results.json', 'w'))
print('fake smoke', status)
sys.exit(0 if status == 'PASSED' else 1)
'''


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.wheel = make_wheel(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_smoke(self, *args, env=None, timeout=120):
        e = dict(os.environ, MOJOLEARN_RUNPOD_KEY_FILE=str(self.dir / 'no-key'))
        e.pop('RUNPOD_API_KEY', None)
        e.update(env or {})
        return subprocess.run(['bash', str(SMOKE), *args], capture_output=True, text=True,
                              timeout=timeout, env=e)

    def test_dry_run_plans_one_pod_and_two_files(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--out', str(self.dir / 'o'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('DRY RUN', r.stdout)
        self.assertIn('"gpuTypeIds": ["NVIDIA GeForce RTX 4090"]', r.stdout)
        self.assertIn('"allowedCudaVersions": ["13.0"]', r.stdout)
        self.assertIn('qualify_verifier_wheel.py', r.stdout)
        self.assertFalse((self.dir / 'o').exists())

    def test_refuses_wrong_commit_and_unrepaired_wheel(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', 'd' * 40)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('records source commit ' + COMMIT, r.stderr)
        raw = make_wheel(self.dir, name='mojolearn-0.0.0-py3-none-linux_x86_64.whl')
        r = self.run_smoke(str(raw), '--expected-source-commit', COMMIT)
        self.assertIn('not a final manylinux', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--rent', '--ssh', 'x')
        self.assertIn('exclusive', r.stderr)

    def test_with_timeout_kills_a_stalled_transfer(self):
        t = time.time()
        r = subprocess.run(['bash', '-c', 'die(){ exit 1; }; . tools/runpod_pod_lib.sh; with_timeout 2 sleep 30'],
                           cwd=ROOT, capture_output=True, timeout=20)
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(time.time() - t, 10)

    def ssh_path(self, status):
        shims = self.dir / 'bin'
        shims.mkdir()
        for name, body in (('ssh', SSH_SHIM), ('sha256sum', SHA_SHIM), ('timeout', TIMEOUT_SHIM)):
            (shims / name).write_text(body)
            (shims / name).chmod(0o755)
        (shims / 'STATUS').write_text(status)
        out = self.dir / 'out'
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--ssh', 'fake@box',
                           '--out', str(out), env={
                               'PATH': str(shims) + ':' + os.environ['PATH'],
                               'MOJOLEARN_SMOKE_REMOTE_DIR': str(self.dir / 'box' / 'wheel-smoke')})
        return r, out

    def test_ssh_path_brings_home_a_passed_receipt(self):
        r, out = self.ssh_path('PASSED')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        receipt = json.loads((out / 'results.json').read_text())
        self.assertEqual(receipt['status'], 'PASSED')
        self.assertIn('verdict=PASSED', (out / 'smoke.txt').read_text())

    def test_ssh_path_fails_a_failed_smoke(self):
        r, out = self.ssh_path('FAILED')
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('verdict=FAILED', (out / 'smoke.txt').read_text())


if __name__ == '__main__':
    unittest.main()
