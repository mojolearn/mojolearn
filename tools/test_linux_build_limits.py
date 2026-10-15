"""Main-only resource controls; stub commands, no compiler or GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


#: The guards build_sets.sh runs before its job check: ptx_baseline.py
#: validate-config (the CUDA code format) and the modules it imports.
GUARD_HELPERS = ('ptx_baseline.py', 'ptx_contract.py', 'device_glue.py')


def stage_guards(root, path):
    """Copy the pre-build guards' helper scripts beside the staged prefix."""
    for name in GUARD_HELPERS:
        shutil.copyfile(root / 'packaging/linux' / name, path / name)


class BuildLimitsTests(unittest.TestCase):
    def test_out_of_range_build_jobs_refused_before_compiler(self):
        # DEVIATION 2501: 1..16 parallel builds are legal; anything else is
        # refused before any tool runs.
        root = Path(__file__).resolve().parents[1]
        source = (root / 'packaging/linux/build_sets.sh').read_text()
        # Execute the actual pre-build guards; stub downstream tools as sentinels.
        prefix = source.split('# THE TWO LISTS BELOW', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'packaging/linux'
            path.mkdir(parents=True)
            script = path / 'build_sets.sh'
            script.write_text(prefix + '\nexit 99\n')
            stage_guards(root, path)
            for jobs in ('17', '0', '-1', 'garbage', '1.5'):
                result = subprocess.run(['bash', str(script), tmp + '/out'],
                    env=dict(os.environ, MOJOLEARN_BUILD_JOBS=jobs),
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('must be 1..16', result.stderr)
            for jobs in ('1', '4', '16'):
                result = subprocess.run(['bash', str(script), tmp + '/out'],
                    env=dict(os.environ, MOJOLEARN_BUILD_JOBS=jobs),
                    capture_output=True, text=True, timeout=5)
                self.assertNotIn('must be 1..16', result.stderr, jobs)

    @unittest.skipUnless(os.uname().sysname == 'Darwin', 'Mac host refusal')
    def test_default_refuses_local_native_build(self):
        root = Path(__file__).resolve().parents[1]
        prefix = (root / 'packaging/linux/build_sets.sh').read_text().split('# THE TWO LISTS BELOW', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'packaging/linux'
            path.mkdir(parents=True)
            script = path / 'build_sets.sh'
            script.write_text(prefix + '\nexit 99\n')
            stage_guards(root, path)
            env = dict(os.environ)
            env.pop('MOJOLEARN_BUILD_JOBS', None)
            result = subprocess.run(['bash', str(script), tmp + '/out'], env=env,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn('remote Linux GPU host', result.stderr)


if __name__ == '__main__':
    unittest.main()
