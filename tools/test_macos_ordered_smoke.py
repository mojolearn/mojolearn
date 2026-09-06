#!/usr/bin/env python3
"""Synthetic orchestration checks only; the main operator runs them."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('macos_ordered_smoke',
    Path(__file__).resolve().parents[1] / 'packaging/macos/ordered_smoke.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def output(mode='identical', vendor='metal'):
    model = 'trees 3\n'
    record = dict(schema='mojolearn.ordered_python.v1', status='PASS', requested_mode=mode,
                  native_mode_code=gate.MODE_CODES[mode], native_vendor=vendor,
                  rows=32, query_rows=8, trees=3, prediction_bits=[0] * 32,
                  query_prediction_bits=[0] * 8, repeat_prediction_bits=[0] * 32,
                  repeat_exact=True, model_text=model,
                  model_sha256=hashlib.sha256(model.encode()).hexdigest(),
                  refusals=['depth', 'objective', 'categorical', 'duplicate_permutation',
                            'zero_weight_mass', 'eval_set'])
    return 'ORDERED_PYTHON_JSON ' + json.dumps(record) + '\nORDERED PYTHON SURFACE PASS\n'


class OrderedReleaseTests(unittest.TestCase):
    def test_all_modes_admitted(self):
        for mode in gate.MODE_CODES:
            gate.admit_output(output(mode), mode, 'metal')

    def test_wrong_vendor_mode_missing_duplicate_evidence(self):
        for bad in (output(vendor='cuda'), output(mode='fast'), '', output() + output(),
                    output().replace('ORDERED PYTHON SURFACE PASS', '')):
            with self.subTest(output=bad[:50]), self.assertRaises(AssertionError):
                gate.admit_output(bad, 'identical', 'metal')

    def test_corrupted_model_and_repeat(self):
        for bad in (output().replace('trees 3', 'trees 2'),
                    output().replace('"repeat_exact": true', '"repeat_exact": false')):
            with self.assertRaises(AssertionError):
                gate.admit_output(bad, 'identical', 'metal')

    def test_exception_and_early_success_exit_propagate_and_retain_output(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            package_dir = prefix / 'lib/site-packages/mojolearn'
            binary = package_dir / '_mojolearn_gbdt.so'
            binding = SimpleNamespace(__file__=str(binary), gbdt_numeric_mode=lambda: 1,
                                      gbdt_vendor=lambda: 'metal')
            package = SimpleNamespace(__file__=str(package_dir / '__init__.py'),
                OrderedRMSE=lambda **kwargs: SimpleNamespace(_bind=lambda name: binding))
            for failure in (RuntimeError('native failed'), SystemExit(0)):
                def run(*args, **kwargs):
                    print('retained native diagnostic')
                    raise failure
                captured = io.StringIO()
                with mock.patch.object(gate.sys, 'prefix', str(prefix)), \
                        mock.patch.object(gate.runpy, 'run_path', side_effect=run), \
                        contextlib.redirect_stdout(captured), \
                        self.assertRaises((RuntimeError, AssertionError)):
                    gate.run_installed_ordered(prefix, package, 'identical', 'metal')
                self.assertIn('retained native diagnostic', captured.getvalue())

    def test_checkout_package_refused(self):
        package = SimpleNamespace(__file__='/checkout/python/mojolearn/__init__.py')
        with self.assertRaises(AssertionError):
            gate.run_installed_ordered(Path('/checkout'), package, 'identical', 'metal')


if __name__ == '__main__':
    unittest.main()
