#!/usr/bin/env python3
"""Small synthetic evidence tests. Main lane alone runs this file."""
import copy
import hashlib
import struct
import unittest
from pathlib import Path
from unittest import mock

import verify_linux_surface_qualification as gate


def bits(n, d):
    return {'shape': [n, d], 'uint32': [[0] * d for _ in range(n)],
            'float32_le_sha256': hashlib.sha256(struct.pack('<I', 0) * n * d).hexdigest()}


def quality():
    rows = []
    for profile, (n, q, d) in gate.FIXTURES.items():
        rows.append({'profile': profile, 'passed': True, 'binding_sha256': 'abc',
                     'binding_mode_code': 1, 'fitted_mode': 'identical',
                     'training_input': bits(n, 3), 'query_input': bits(q, 3),
                     'training_embedding': bits(n, d), 'query_embedding': bits(q, d),
                     'quality': {'trustworthiness': 0.9, 'retention': 0.5},
                     'control_margins': {'trustworthiness': 0.4, 'retention': 0.3},
                     'controls': {name: {'trustworthiness': 0.5, 'retention': 0.2} for name in
                                  ('query_embedding_permutation', 'training_embedding_permutation')}})
    return {'status': 'PASS', 'profile': 'expanded', 'mode': 'identical', 'k': 5,
            'thresholds': {'trustworthiness': 0.85, 'retention': 0.35, 'minimum_control_margin': 0.15},
            'results': rows}


class AdmissionTests(unittest.TestCase):
    def test_complete_expanded_quality(self):
        gate.check_quality(quality(), 'identical', 'abc')

    def test_missing_duplicate_and_wrong_mode(self):
        for mutation in ('missing', 'duplicate', 'mode', 'binding'):
            record = quality()
            if mutation == 'missing':
                record['results'].pop()
            elif mutation == 'duplicate':
                record['results'][1] = copy.deepcopy(record['results'][0])
            elif mutation == 'mode':
                record['results'][0]['binding_mode_code'] = 0
            else:
                record['results'][0]['binding_sha256'] = 'different'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.check_quality(record, 'identical', 'abc')

    def test_pass_boolean_cannot_hide_failed_metric(self):
        record = quality()
        record['results'][0]['quality']['trustworthiness'] = 0.8
        with self.assertRaises(ValueError):
            gate.check_quality(record, 'identical', 'abc')

    def test_raw_hash_and_nonfinite(self):
        for value in (1, 0x7f800000, 0x7fc00000, -1):
            record = quality()
            record['results'][0]['query_embedding']['uint32'][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate.check_quality(record, 'identical', 'abc')

    def test_weakened_contract_refused(self):
        record = quality()
        record['thresholds']['trustworthiness'] = 0.1
        with self.assertRaises(ValueError):
            gate.check_quality(record, 'identical', 'abc')


    def test_compare_matches_and_rejects_wrong_source_or_bits(self):
        left = {'vendor': 'hip', 'source_sha256': 'source', 'wheel_sha256': 'hipwheel',
                'evidence_sha256': {'qualification-sources.json': 'tests'}}
        right = dict(left, vendor='cuda', wheel_sha256='cudawheel')
        qa = quality()
        for row in qa['results']:
            row.update(parameters={}, transform_schedule={}, fitted_config=[])
        qb = copy.deepcopy(qa)
        with mock.patch.object(gate, 'retained', side_effect=[(left, qa), (right, qb)]), \
                mock.patch.object(gate, 'sha', return_value='certificate'):
            result = gate.compare(Path('/fetched/amd'), Path('/fetched/nvidia'))
        self.assertEqual(result['compared_arrays'], 24)
        for mismatch in ('source', 'bits', 'vendor'):
            r, q = copy.deepcopy(right), copy.deepcopy(qb)
            if mismatch == 'source':
                r['source_sha256'] = 'stale'
            elif mismatch == 'vendor':
                r['vendor'] = 'hip'
            else:
                q['results'][0]['query_embedding']['uint32'][0][0] = 1
            with self.subTest(mismatch=mismatch), \
                    mock.patch.object(gate, 'retained', side_effect=[(left, qa), (r, q)]), \
                    self.assertRaises(ValueError):
                gate.compare(Path('/fetched/amd'), Path('/fetched/nvidia'))


if __name__ == '__main__':
    unittest.main()
