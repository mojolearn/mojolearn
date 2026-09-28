import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

DRIVER = Path(__file__).resolve().parents[1] / 'check.py'
WRAPPER = DRIVER.with_name('mac_job.sh')


def load_driver():
    spec = importlib.util.spec_from_file_location('consolidated_test', DRIVER)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'algos_lane_check': types.SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class CheckTests(unittest.TestCase):
    def test_incomplete_and_duplicate_rows_refused(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'rows'
            for text in ['a\tcpu1=AGREE\tdetail\n',
                         'a\tcpu1=AGREE\tdetail\tcpu3=AGREE\tdetail\n' * 2,
                         'other\tcpu1=AGREE\tdetail\tcpu3=AGREE\tdetail\n']:
                file.write_text(text)
                with self.assertRaises(ValueError):
                    driver.read_rows(file, ['a'], ['1', '3'])

    def test_plan_rejects_empty_duplicate_and_unavailable_selection(self):
        driver = load_driver()
        driver.alc = types.SimpleNamespace(
            load_harness=lambda: types.SimpleNamespace(LANES={'a': None}),
            needed_bindings=lambda lanes: {'a': ['binding']})
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, 'source_commit', return_value='commit'):
            for selection in ['a,a', 'a,', ',', 'unknown']:
                with self.assertRaises(ValueError):
                    driver.plan(Path(tmp), selection)
            driver.alc.load_harness = lambda: types.SimpleNamespace(LANES={})
            with self.assertRaises(ValueError):
                driver.plan(Path(tmp))
            driver.alc.load_harness = lambda: types.SimpleNamespace(LANES={'unavailable': None})
            with self.assertRaises(ValueError):
                driver.plan(Path(tmp))

    def test_binding_and_selection_change_fingerprint(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            binding = pkg / 'binding.so'
            binding.write_bytes(b'old binary')
            math = pkg / ('.dylibs/libMojolearnMath.dylib' if driver.sys.platform == 'darwin' else '.libs/libMojolearnMath.so')
            math.parent.mkdir()
            math.write_bytes(b'math')
            driver.alc = types.SimpleNamespace(PKG=pkg, output_for=lambda _: binding, gpu_arch=lambda: 'arch')
            plan = {'needed': {'a': ['binding']}}
            with patch.object(driver, 'source_commit', return_value='commit'):
                first = driver.fingerprint(plan, ['a'], ['1'], 'metal')
                binding.write_bytes(b'new binary')
                second = driver.fingerprint(plan, ['a'], ['1'], 'metal')
                self.assertNotEqual(first, second)
                self.assertNotEqual(second, driver.fingerprint(plan, ['a'], ['3'], 'metal'))

    def test_wrapper_propagates_build_and_clean_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixi = root / 'pixi'
            pixi.write_text('#!/bin/bash\ncase "$*" in\n*"check.py build"*) exit "${BUILD_EXIT:-0}";;\n*"check.py clean"*) exit "${CLEAN_EXIT:-0}";;\nesac\nexit 0\n')
            pixi.chmod(0o755)
            for build, clean, expected in [(7, 0, 7), (0, 9, 9), (0, 0, 0)]:
                env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'], BUILD_EXIT=str(build), CLEAN_EXIT=str(clean))
                result = subprocess.run(['bash', str(WRAPPER), '0/1', str(root/'out')], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual('CONSOLIDATED CHECK PASS' in result.stdout, expected == 0)


if __name__ == '__main__':
    unittest.main()
