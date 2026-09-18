"""No numerical execution: capture scope, wheel identity, and timeout controls."""
import tempfile
from pathlib import Path
import sys
import unittest
import zipfile

import capture_ordinary_holds as capture


class CaptureTests(unittest.TestCase):
    def test_all_retained_properties_and_two_repeats_are_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'column.json'
            cmd = capture.column_command('python', False, 'mamba3', 'cuda', 'nvidia-test', output, True)
            self.assertEqual(cmd[:3], ['python', '-m', 'mojolearn._identity_break'])
            self.assertEqual(cmd[cmd.index('--repeats') + 1], '2')
            for flag in capture.PROPERTIES:
                self.assertIn(flag, cmd)
            self.assertNotIn('--resume', cmd)
            output.write_text('{}')
            self.assertIn('--resume', capture.column_command('python', True, 'gpc', 'cpu', 'cpu-test', output, True))

    def test_partial_or_modified_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'mojolearn'
            package.mkdir()
            wheel = root / 'test.whl'
            with zipfile.ZipFile(wheel, 'w') as z:
                z.writestr('mojolearn/__init__.py', b'correct')
                z.writestr('mojolearn/host/binding.so', b'native')
            (package / '__init__.py').write_bytes(b'correct')
            with self.assertRaisesRegex(ValueError, 'binding.so'):
                capture.verify_wheel(wheel, package)
            (package / 'host').mkdir()
            (package / 'host/binding.so').write_bytes(b'native')
            self.assertEqual(capture.verify_wheel(wheel, package)['package_files_checked'], 2)
            (package / '__init__.py').write_bytes(b'incorrect')
            with self.assertRaisesRegex(ValueError, '__init__'):
                capture.verify_wheel(wheel, package)

    def test_timeout_returns_failure_and_preserves_log(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'log'
            code = capture.run_stage([sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(20)'],
                                      log, 0.2, os.environ.copy(), tmp)
            self.assertEqual(code, 124)
            self.assertIn('started', log.read_text())


if __name__ == '__main__':
    unittest.main()
