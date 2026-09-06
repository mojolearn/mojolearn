"""Main-only resource controls; stub commands, no compiler or GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class BuildLimitsTests(unittest.TestCase):
    def test_parallel_build_request_refused_before_compiler(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'packaging/linux/build_sets.sh').read_text()
        # Execute the actual pre-build guards; stub downstream tools as sentinels.
        prefix = source.split('# THE TWO LISTS BELOW', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'packaging/linux'
            path.mkdir(parents=True)
            script = path / 'build_sets.sh'
            script.write_text(prefix + '\nexit 99\n')
            for jobs in ('2', '3', '0', '-1', 'garbage'):
                result = subprocess.run(['bash', str(script), tmp + '/out'],
                    env=dict(os.environ, MOJOLEARN_BUILD_JOBS=jobs),
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('must be 1', result.stderr)

    @unittest.skipUnless(os.uname().sysname == 'Darwin', 'Mac host refusal')
    def test_default_refuses_local_native_build(self):
        root = Path(__file__).resolve().parents[1]
        prefix = (root / 'packaging/linux/build_sets.sh').read_text().split('# THE TWO LISTS BELOW', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'packaging/linux'
            path.mkdir(parents=True)
            script = path / 'build_sets.sh'
            script.write_text(prefix + '\nexit 99\n')
            env = dict(os.environ)
            env.pop('MOJOLEARN_BUILD_JOBS', None)
            result = subprocess.run(['bash', str(script), tmp + '/out'], env=env,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn('remote Linux GPU host', result.stderr)


if __name__ == '__main__':
    unittest.main()
