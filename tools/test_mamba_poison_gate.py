"""Reject setup failures masquerading as a detected poisoned device read."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PoisonNegativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).with_name('mamba_poison_gate.sh')).read_text()
        self.validator = source.split("<<'PY_POISON_NEGATIVE'\n", 1)[1].split(
            '\nPY_POISON_NEGATIVE', 1)[0]
        fixtures = ('base', 'ties', 'hashed', 'wide', 'denormal',
                    'denormal_ftz', 'dupes', 'odd', 'negative')
        self.clean = {'complete': True, 'repeats': 2, 'cells': {
            lane+'/'+fixture: {'verdict': 'STABLE', 'hashes': ['0123456789abcdef'] * 2}
            for lane in ('mamba1', 'mamba2', 'mamba2-dtlimit', 'mamba3')
            for fixture in fixtures}}
        self.poisoned = copy.deepcopy(self.clean)
        self.poisoned['cells']['mamba2/base'] = {
            'verdict': 'REFUSED', 'hashes': [],
            'error': 'Exception: mamba: NaN in state.buf_xbc at flat index 0 '
                     'REFUSED (row 39: NaN payloads are vendor-shaped; no stage may record one)'}

    def admit(self, capture):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ('capture.json', 'apple.json', 'nvidia.json', 'amd.json')]
            for path, record in zip(paths, [capture, self.clean, self.clean, self.clean]):
                path.write_text(json.dumps(record))
            result = subprocess.run([sys.executable, '-', *map(str, paths)],
                                    input=self.validator, text=True, capture_output=True)
            return result.returncode == 0

    def test_native_nan_refusal_is_numerical_evidence(self):
        self.assertTrue(self.admit(self.poisoned))

    def test_changed_real_hash_is_numerical_evidence(self):
        record = copy.deepcopy(self.clean)
        record['cells']['mamba2/base']['hashes'] = ['fedcba9876543210'] * 2
        self.assertTrue(self.admit(record))

    def test_no_effect_is_not_a_successful_negative_control(self):
        self.assertFalse(self.admit(self.clean))

    def test_unrelated_refusal_is_not_numerical_evidence(self):
        for message in ('ModuleNotFoundError: missing binding',
                        'Exception: mamba: GPU launch failed', ''):
            with self.subTest(error=message):
                record = copy.deepcopy(self.poisoned)
                record['cells']['mamba2/base']['error'] = message
                self.assertFalse(self.admit(record))

    def test_unaffected_controls_must_match(self):
        for fault in ('refused', 'changed'):
            with self.subTest(fault=fault):
                record = copy.deepcopy(self.poisoned)
                if fault == 'refused':
                    record['cells']['mamba1/base'] = copy.deepcopy(record['cells']['mamba2/base'])
                else:
                    record['cells']['mamba1/base']['hashes'] = ['fedcba9876543210'] * 2
                self.assertFalse(self.admit(record))

    def test_missing_or_incomplete_evidence_is_rejected(self):
        for fault in ('cell', 'hashes', 'complete'):
            with self.subTest(fault=fault):
                record = copy.deepcopy(self.poisoned)
                if fault == 'cell':
                    del record['cells']['mamba2/base']
                elif fault == 'hashes':
                    record['cells']['mamba2/base'] = {'verdict': 'STABLE', 'hashes': []}
                else:
                    record['complete'] = False
                self.assertFalse(self.admit(record))


if __name__ == '__main__':
    unittest.main()
