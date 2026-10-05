"""Pure receipt tests: no Mojo imports, compilation, model fits, or GPU use."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from identical_wave_cnn_sgd_compare import EXPECTED, compare


class GateReceiptTests(unittest.TestCase):
    def setUp(self):
        self.vendors = ['cuda', 'hip', 'metal', 'cpu']
        self.paths = [Path(v+'.json') for v in self.vendors]
        self.reports = [dict(status='PASS', sha='a'*40, arm='on', suite='sgd',
                             harness_sha256='b'*64, vendor=v, timing_samples=0,
                             opponents_executed=0,
                             bindings={'x_linear': dict(numeric_mode=1, vendor=v, sha256='c'*64)},
                             cases={key: dict(status='PASS', digest='d'*64) for key in EXPECTED['sgd']})
                        for v in self.vendors]

    def run_compare(self, reports=None, paths=None):
        raw = {p: json.dumps(r).encode() for p, r in zip(self.paths, reports or self.reports)}
        with patch.object(Path, 'read_bytes', autospec=True, side_effect=lambda path: raw[path]):
            return compare(paths or self.paths, self.vendors), raw

    def test_complete_four_vendor_receipts(self):
        result, raw = self.run_compare()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['receipts'][0]['sha256'], hashlib.sha256(raw[self.paths[0]]).hexdigest())

    def test_source_arm_harness_or_digest_mismatch(self):
        for field, value in [('sha', 'e'*40), ('arm', 'off'), ('harness_sha256', 'e'*64)]:
            reports = copy.deepcopy(self.reports); reports[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.run_compare(reports)
        reports = copy.deepcopy(self.reports)
        reports[1]['cases']['sgd-ovr-sgd']['digest'] = 'e'*64
        with self.assertRaises(ValueError): self.run_compare(reports)

    def test_missing_case_or_vendor(self):
        reports = copy.deepcopy(self.reports); del reports[1]['cases']['sgd-ovr-sgd']
        with self.assertRaises(ValueError): self.run_compare(reports)
        with self.assertRaises(ValueError): self.run_compare(paths=self.paths[:-1])

    def test_failed_gate_or_case(self):
        reports = copy.deepcopy(self.reports); reports[1]['status'] = 'FAIL'
        with self.assertRaises(ValueError): self.run_compare(reports)
        reports = copy.deepcopy(self.reports); reports[1]['cases']['sgd-ovr-sgd']['status'] = 'FAIL'
        with self.assertRaises(ValueError): self.run_compare(reports)

    def test_invalid_or_missing_digests(self):
        for value in [None, '', 'not-a-hash']:
            reports = copy.deepcopy(self.reports)
            for report in reports: report['cases']['sgd-ovr-sgd']['digest'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.run_compare(reports)

    def test_invalid_binding_provenance(self):
        for field, value in [('vendor', 'cpu'), ('numeric_mode', 0), ('sha256', 'bad')]:
            reports = copy.deepcopy(self.reports); reports[1]['bindings']['x_linear'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.run_compare(reports)

    def test_duplicate_vendor_and_timing_rejected(self):
        with self.assertRaises(ValueError): self.run_compare(paths=self.paths+[self.paths[0]])
        reports = copy.deepcopy(self.reports); reports[1]['timing_samples'] = 1
        with self.assertRaises(ValueError): self.run_compare(reports)


if __name__ == '__main__': unittest.main()
