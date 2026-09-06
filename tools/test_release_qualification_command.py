"""Exercise the actual workflow shell with a recording qualification command."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


class QualificationCommandTests(unittest.TestCase):
    def invoke(self, wheelhouse=None, wheels=1):
        workflow = Path(__file__).resolve().parents[1] / '.github/workflows/release-provenance.yml'
        steps = yaml.safe_load(workflow.read_text())['jobs']['build']['steps']
        command = next(s['run'] for s in steps if s.get('name') ==
                       'Qualify installed UMAP fit, transform and held-out quality')
        with tempfile.TemporaryDirectory(prefix='release command ') as directory:
            root = Path(directory)
            (root / 'python/dist').mkdir(parents=True)
            (root / 'python/mojolearn').mkdir()
            (root / 'python/mojolearn/_version.py').write_text('__version__ = "0.6.0"\n')
            for i in range(wheels):
                (root / f'python/dist/mojolearn-0.6.0-{i}.whl').touch()
            (root / 'bin').mkdir()
            recorder = root / 'bin/python3'
            recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENT_RECORD"\n')
            recorder.chmod(0o755)
            env = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'],
                       ARGUMENT_RECORD=str(root / 'args'), GITHUB_WORKSPACE=str(root),
                       RUNNER_TEMP=str(root), GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1')
            env.pop('MOJOLEARN_QUALIFICATION_WHEELHOUSE', None)
            if wheelhouse is not None:
                env['MOJOLEARN_QUALIFICATION_WHEELHOUSE'] = wheelhouse
            result = subprocess.run(['bash', '-e', '-c', command], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10)
            args = (root / 'args').read_text().splitlines() if (root / 'args').exists() else []
            return result.returncode, args

    def test_normal_install_retains_exact_wheel(self):
        status, args = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(args[:2], ['tools/qualify_umap_wheel.py', 'python/dist/mojolearn-0.6.0-0.whl'])
        self.assertNotIn('--wheelhouse', args)

    def test_offline_path_preserves_spaces_and_shell_literals(self):
        wheelhouse = '/tmp/dependency wheels/$(false)'
        status, args = self.invoke(wheelhouse)
        self.assertEqual(status, 0)
        self.assertEqual(args[1:4], ['python/dist/mojolearn-0.6.0-0.whl', '--wheelhouse', wheelhouse])

    def test_ambiguous_or_absent_wheel_never_invokes_qualification(self):
        for count in (0, 2):
            with self.subTest(wheels=count):
                status, args = self.invoke(wheels=count)
                self.assertNotEqual(status, 0)
                self.assertEqual(args, [])


if __name__ == '__main__':
    unittest.main()
