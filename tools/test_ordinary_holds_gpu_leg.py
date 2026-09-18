"""Exercise family ordering/failure retention without builds or GPU execution."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from capture_ordinary_holds import LANES


class LegTests(unittest.TestCase):
    def test_family_capture_precedes_later_build_and_failure_is_retained(self):
        for fail in ('', 'build_gp'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'tools').mkdir()
                (root / 'bindings').mkdir()
                (root / 'bin').mkdir()
                (root / 'commit.txt').write_text('a' * 40)
                source = Path(__file__).with_name('ordinary_holds_gpu_leg.sh')
                shutil.copyfile(source, root / 'tools/ordinary_holds_gpu_leg.sh')
                (root / 'tools/capture_ordinary_holds.py').write_text(
                    'LANES = ' + repr(LANES) + '\n'
                    'if __name__ == "__main__":\n'
                    ' import os,sys\n'
                    ' with open(os.environ["TRACE"], "a") as f: f.write("capture:" + sys.argv[sys.argv.index("--lanes")+1] + "\\n")\n')
                for build in ('build', 'build_kernel_methods', 'build_estimators_host', 'build_gp', 'build_svm'):
                    (root / f'bindings/{build}.sh').write_text(
                        f'echo {build} >> "$TRACE"\n'
                        f'[ "${{FAIL_BUILD:-}}" != "{build}" ]\n')
                for name, body in {'pixi': 'shift; exec "$@"',
                                   'timeout': 'shift 3; exec "$@"',
                                   'realpath': 'shift; printf "%s\\n" "$1"'}.items():
                    path = root / 'bin' / name
                    path.write_text('#!/bin/sh\n' + body + '\n')
                    path.chmod(0o755)
                (root / 'bin/python').symlink_to(sys.executable)
                trace = root / 'trace'
                env = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'],
                           TRACE=str(trace), FAIL_BUILD=fail, MOJOLEARN_ORDINARY_BOUNDED='1',
                           MOJOLEARN_ORDINARY_LANES='kernel-ridge-poly,gpc,svc-poly',
                           MOJOLEARN_GPU_ARCHS='gfx942')
                result = subprocess.run(['bash', str(root / 'tools/ordinary_holds_gpu_leg.sh'),
                                         'hip', 'amd-test', str(root / 'out'), '30'],
                                         env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, int(bool(fail)), result.stderr)
                events = trace.read_text().splitlines()
                self.assertLess(events.index('capture:kernel-ridge-poly'), events.index('build_gp'))
                self.assertIn('capture:svc-poly', events)
                self.assertEqual('capture:gpc' in events, not bool(fail))
                self.assertNotIn('build_training', events)
                self.assertEqual((root / 'out/exit_code').read_text().strip(), str(int(bool(fail))))


if __name__ == '__main__':
    unittest.main()
