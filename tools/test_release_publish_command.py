"""Exercise the publisher's receipt staging and dispatch without external writes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import check_light_release as gate

ROOT = Path(__file__).resolve().parents[1]


class PublishCommandTests(unittest.TestCase):
    def invoke(self, platform, receipt=True, failed=False):
        with tempfile.TemporaryDirectory(prefix='release publish ') as directory:
            root = Path(directory)
            for name in ('bin', 'python/mojolearn', 'tools', 'packaging'):
                (root / name).mkdir(parents=True)
            (root / 'python/mojolearn/_version.py').write_text('__version__ = "0.8.9"\n')
            shutil.copy(ROOT / 'tools/check_light_release.py', root / 'tools')
            # This is a staging/dispatch test, not native artifact admission.
            (root / 'packaging/verify_alpha_artifacts.py').write_text('print("{}")\n')
            source = 'a' * 40
            wheel = root / ('mojolearn-0.8.9-py3-none-' +
                ('manylinux_2_35_x86_64' if platform == 'linux' else 'macosx_11_0_arm64') + '.whl')
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('mojolearn/identity_columns/COMMIT', source)
            report = root / 'results.json'
            report.write_text(json.dumps(dict(status='FAILED' if failed else 'PASSED',
                scope='expanded', release_qualified=False, source_commit=source,
                wheel=str(wheel), wheel_sha256=gate.digest(wheel),
                installed={'vendor': 'cuda' if platform == 'linux' else 'metal'},
                expanded={'scope': 'expanded'},
                jobs=[dict(name=name, exit_code=0) for name in sorted(gate.JOBS)])))
            scripts = {
                'git': '#!/bin/sh\nif [ "$1" = rev-parse ]; then echo ' + source + '; fi\n',
                'gh': '#!/bin/sh\nprintf "%s\\n" "$*" >> "$COMMAND_RECORD"\n'
                      'if [ "$1 $2" = "release view" ]; then exit 1; fi\n'
                      'if [ "$1 $2" = "run list" ]; then echo 42; fi\n',
                'sleep': '#!/bin/sh\nexit 0\n',
            }
            for name, content in scripts.items():
                path = root / 'bin' / name
                path.write_text(content)
                path.chmod(0o755)
            (root / 'bin/python3').symlink_to(sys.executable)
            env = {k: v for k, v in os.environ.items() if not k.startswith('MOJOLEARN_')}
            env.update(PATH=str(root / 'bin') + os.pathsep + env['PATH'],
                       MOJOLEARN_REPO=str(root), COMMAND_RECORD=str(root / 'commands'))
            command = ['bash', str(ROOT / 'tools/release_linux_publish.sh'), str(wheel),
                       'alpha-api-0.8.9-test', 'none', str(root / 'work')]
            if receipt:
                command.extend(['--light-smoke', str(report)])
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            record = root / 'commands'
            calls = record.read_text() if record.exists() else ''
            manifest = json.loads((root / 'work/artifact/alpha-manifest.json').read_text())
            return result, calls, manifest

    def test_both_platforms_stage_receipt_and_select_their_light_batch(self):
        for platform in ('linux', 'macos'):
            with self.subTest(platform=platform):
                result, calls, manifest = self.invoke(platform)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('validation_profile=light', calls)
                self.assertIn('light_platform=' + platform, calls)
                self.assertIn('light-smoke-' + platform + '.json', calls)
                self.assertEqual(manifest['light_smoke']['source_commit'], 'a' * 40)

    def test_failed_smoke_never_reaches_github(self):
        result, calls, _ = self.invoke('linux', failed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('installed smoke did not pass', result.stderr)
        self.assertEqual(calls, '')

    def test_existing_linux_call_keeps_full_profile(self):
        result, calls, manifest = self.invoke('linux', receipt=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('validation_profile=full', calls)
        self.assertNotIn('light_smoke', manifest)


if __name__ == '__main__':
    unittest.main()
