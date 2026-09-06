# SPDX-License-Identifier: Apache-2.0
"""Small stdlib policy tests; retained qualification admission has its own suite."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_ordered_python as gate


def record(vendor='hip'):
    text = '# model\ntrees 3\n'
    return {
        'schema': 'mojolearn.ordered_python.v1', 'status': 'PASS',
        'requested_mode': 'identical', 'native_mode_code': 1, 'native_vendor': vendor,
        'rows': 32, 'query_rows': 8, 'trees': 3, 'repeat_exact': True,
        'refusals': sorted(gate.REFUSALS), 'model_text': text,
        'model_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'prediction_bits': [0x3f800000] * 32,
        'query_prediction_bits': [0xbf800000] * 8,
        'repeat_prediction_bits': [0x3f800000] * 32,
    }


class OrderedComparisonTests(unittest.TestCase):
    def test_valid_record(self):
        self.assertEqual(gate.check_record(record(), 'hip')['rows'], 32)

    def test_record_admission_refusals(self):
        mutations = [
            ('native_mode_code', True), ('requested_mode', 'fast'),
            ('native_vendor', 'cuda'), ('rows', 31), ('repeat_exact', False),
            ('refusals', sorted(gate.REFUSALS)[:-1]),
            ('refusals', ['depth'] * 6), ('model_sha256', 'bad'),
            ('query_prediction_bits', [0] * 7),
            ('prediction_bits', [True] * 32),
            ('prediction_bits', [0x7f800000] * 32),
            ('prediction_bits', [0xff800000] * 32),
            ('prediction_bits', [0x7fc00000] * 32),
            ('prediction_bits', [-1] * 32),
            ('prediction_bits', [2**32] * 32),
            ('repeat_prediction_bits', [0] * 32),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                bad = record()
                bad[field] = value
                with self.assertRaises(ValueError):
                    gate.check_record(bad, 'hip')

    def fixtures(self, base):
        qualifications = {}
        for vendor in ('hip', 'cuda'):
            directory = base / vendor
            directory.mkdir()
            qualification = {
                'status': 'PASSED', 'vendor': vendor, 'source_sha256': 'a' * 64,
                'wheel_sha256': vendor + '-wheel',
                'evidence_sha256': {'qualification-sources.json': 'b' * 64},
            }
            qualifications[directory] = qualification
            (directory / 'qualification.json').write_text(json.dumps(qualification))
            (directory / 'ordered-rmse-identical.log').write_text(
                'ORDERED_PYTHON_NATIVE identical ' + vendor + '\n'
                + gate.PREFIX + json.dumps(record(vendor)) + '\n'
                + 'ORDERED PYTHON SURFACE PASS\n')
            # Original remote paths intentionally do not exist on this machine.
            package = '/remote/vanished/venv/site-packages/mojolearn'
            installed = {
                'mode': 'identical', 'vendor': vendor, 'wheel_sha256': vendor + '-wheel',
                'package': package + '/__init__.py',
                'installed_bindings': {'_mojolearn_gbdt': {
                    'mode_code': 1, 'sha256': 'c' * 64,
                    'path': package + '/identical/_mojolearn_gbdt.so',
                }},
            }
            (directory / 'ordered-rmse-identical.installed.json').write_text(json.dumps(installed))
            (directory / 'wheel-audit.json').write_text(json.dumps({
                'extension_hashes': {'identical/_mojolearn_gbdt.so': 'c' * 64},
            }))
        return qualifications

    def test_compare_is_portable_and_calls_retained_for_both(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qualifications = self.fixtures(root)
            with patch.object(gate, 'retained', side_effect=lambda p: (qualifications[p], {})) as retained:
                result = gate.compare(root / 'hip', root / 'cuda')
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['prediction_cells'], 32)
            self.assertEqual(retained.call_count, 2)

    def test_failed_retained_admission_cannot_be_bypassed(self):
        with patch.object(gate, 'retained', side_effect=ValueError('retained red')):
            with self.assertRaisesRegex(ValueError, 'retained red'):
                gate.load(Path('/unused'))

    def test_duplicate_log_and_cross_vendor_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qualifications = self.fixtures(root)
            cuda_log = root / 'cuda' / 'ordered-rmse-identical.log'
            original = cuda_log.read_text()
            with patch.object(gate, 'retained', side_effect=lambda p: (qualifications[p], {})):
                cuda_log.write_text(original + gate.PREFIX + json.dumps(record('cuda')) + '\n')
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    gate.compare(root / 'hip', root / 'cuda')
                changed = record('cuda')
                changed['query_prediction_bits'][0] = 0
                cuda_log.write_text(original.replace(json.dumps(record('cuda')), json.dumps(changed)))
                with self.assertRaisesRegex(ValueError, 'IDENTICAL mismatch'):
                    gate.compare(root / 'hip', root / 'cuda')
                cuda_log.write_text(original)
                qualifications[root / 'cuda']['source_sha256'] = 'd' * 64
                with self.assertRaisesRegex(ValueError, 'Different native build sources'):
                    gate.compare(root / 'hip', root / 'cuda')
                qualifications[root / 'cuda']['source_sha256'] = 'a' * 64
                qualifications[root / 'cuda']['evidence_sha256']['qualification-sources.json'] = 'e' * 64
                with self.assertRaisesRegex(ValueError, 'Different qualification sources'):
                    gate.compare(root / 'hip', root / 'cuda')


if __name__ == '__main__':
    unittest.main()
