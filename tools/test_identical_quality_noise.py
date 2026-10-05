"""Synthetic retained-artifact tests, no model or GPU execution."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import identical_quality_noise as q


class QualityNoiseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_sha = 'a' * 40
        self.protocol = {
            'schema': 1, 'reviewed': True, 'primary_metrics': ['accuracy', 'r2'],
            'family_alpha': .05, 'per_metric_alpha': .025,
            'resamples': 2000, 'seed': 20261004, 'quantile_method': 'linear',
            'margin_rule': 'baseline_score_minus_lower_95_bound',
            'resampling': {'kind': 'iid', 'justification': 'Synthetic independent Bernoulli test rows.'},
            'cases': {
                'classification': {'prediction_key': 'clf', 'target_key': 'cy', 'probability_key': 'prob'},
                'regression': {'prediction_key': 'reg', 'target_key': 'ry'}}}
        self.targets = self.root / 'targets.npz'
        self.cy = np.arange(64) % 2
        self.ry = np.linspace(.1, 5, 64)
        np.savez(self.targets, cy=self.cy, ry=self.ry)
        evidence = self.root / 'synthetic-source.txt'
        evidence.write_text('Synthetic independent test observations; no real-model evidence.\n')
        self.metadata = {'targets_sha256': q.digest(self.targets), 'sampling_design': 'random_iid',
                         'iid_justified': True,
                         'source_evidence': [{'path': str(evidence), 'sha256': q.digest(evidence)}]}
        self.protocol_path = self.root / 'protocol.json'
        self.metadata_path = self.root / 'metadata.json'
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.metadata_path.write_text(json.dumps(self.metadata))
        clf = self.cy.copy()
        clf[:16] = 1 - clf[:16]
        self.off = self.make_artifact('off', clf, self.ry + .1)

    def make_artifact(self, arm, clf, reg):
        path = self.root / (arm + '.npz')
        prob = np.where(clf == 1, .8, .2)
        np.savez(path, clf=clf, reg=reg, prob=prob)
        binary = str(self.root / arm / '_mojolearn_rf.so')
        build = {'status': 'PASS', 'sha': self.source_sha, 'arm': arm,
                 'modules': {'rf': {'status': 'PASS', 'artifact': binary, 'sha256': 'b' * 64}}}
        build_path = self.root / (arm + '-build.json')
        build_path.write_text(json.dumps(build))
        receipt = {'schema': 1, 'arm': arm, 'numeric_mode': 'identical',
                   'source_sha': self.source_sha, 'predictions_sha256': q.digest(path),
                   'targets_sha256': q.digest(self.targets), 'fixture_id': 'synthetic-test',
                   'parameter_sha256': {'classification': 'c' * 64, 'regression': 'd' * 64},
                   'native_build_receipts': [{'path': str(build_path), 'sha256': q.digest(build_path)}],
                   'loaded_native_artifacts': {binary: 'b' * 64}}
        receipt_path = self.root / (arm + '-receipt.json')
        receipt_path.write_text(json.dumps(receipt))
        return path, receipt_path

    def freeze(self):
        args = argparse.Namespace(protocol=self.protocol_path, target_metadata=self.metadata_path,
                                  baseline=self.off[0], baseline_receipt=self.off[1], targets=self.targets,
                                  out=self.root / 'frozen.json')
        q.calibrate(args)
        return args.out

    def adjudicate(self, frozen, on):
        return q.compare(argparse.Namespace(frozen=frozen, frozen_sha256=q.digest(frozen),
                                            candidate=on[0], candidate_receipt=on[1],
                                            out=self.root / 'decision.json'))

    def test_identical_predictions_pass_without_claiming_training_noise(self):
        frozen = self.freeze()
        with np.load(self.off[0]) as z:
            on = self.make_artifact('on', z['clf'], z['reg'])
        result = self.adjudicate(frozen, on)
        self.assertEqual(result['status'], 'PASS')
        for row in result['metrics'].values():
            self.assertEqual(row['lower_bound'], 0)
            self.assertEqual(row['upper_bound'], 0)
        self.assertTrue(result['root_decision_required'])
        self.assertIn('logloss', result['metrics']['classification']['diagnostics']['candidate'])

    def test_large_degradation_fails_both_primary_metrics(self):
        frozen = self.freeze()
        on = self.make_artifact('on', 1 - self.cy, self.ry + 10)
        result = self.adjudicate(frozen, on)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(all(r['status'] == 'FAIL' for r in result['metrics'].values()))

    def test_uncertain_small_loss_is_inconclusive_not_equivalence(self):
        np.savez(self.targets, cy=self.cy, ry=self.ry, groups=np.arange(64) // 3)
        self.protocol['resampling'] = {
            'kind': 'groups', 'unit': 'synthetic-batch', 'minimum_units': 20,
            'justification': 'Independent synthetic batches; preserve within-batch dependence.'}
        for case in self.protocol['cases'].values():
            case['group_key'] = 'groups'
        self.metadata.update(targets_sha256=q.digest(self.targets),
                             sampling_design='synthetic_clusters', iid_justified=False,
                             resampling_unit='synthetic-batch',
                             dependence_justification='Independent synthetic batches.')
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.metadata_path.write_text(json.dumps(self.metadata))
        self.off = self.make_artifact('off', self.cy, self.ry)
        frozen = self.freeze()
        changed = self.cy.copy()
        changed[:3] = 1 - changed[:3]
        result = self.adjudicate(frozen, self.make_artifact('on', changed, self.ry))
        self.assertEqual(result['status'], 'INCONCLUSIVE')
        self.assertEqual(result['metrics']['classification']['upper_bound'], 0)

    def test_tampered_freeze_commitment_refused(self):
        frozen = self.freeze()
        committed = q.digest(frozen)
        value = q.read_json(frozen)
        value['metrics']['classification']['margin'] = 1
        frozen.write_text(json.dumps(value))
        with self.assertRaisesRegex(q.Insufficient, 'freeze digest'):
            q.compare(argparse.Namespace(frozen=frozen, frozen_sha256=committed))

    def test_changed_baseline_refused(self):
        frozen = self.freeze()
        on = self.make_artifact('on', self.cy, self.ry)
        self.off[0].write_bytes(b'changed')
        with self.assertRaisesRegex(q.Insufficient, 'frozen baseline'):
            self.adjudicate(frozen, on)

    def test_native_and_parameter_provenance_fail_closed(self):
        frozen = self.freeze()
        on = self.make_artifact('on', self.cy, self.ry)
        receipt = q.read_json(on[1])
        receipt['parameter_sha256']['regression'] = 'e' * 64
        on[1].write_text(json.dumps(receipt))
        with self.assertRaisesRegex(q.Insufficient, 'parameter_sha256'):
            self.adjudicate(frozen, on)
        receipt['loaded_native_artifacts'] = {'/store/wheel/_mojolearn_rf.so': 'b' * 64}
        on[1].write_text(json.dumps(receipt))
        with self.assertRaisesRegex(q.Insufficient, 'loaded artifact'):
            q.receipt(on[1], on[0], self.targets, 'on')

    def test_temporal_tail_is_not_iid(self):
        metadata = dict(self.metadata, sampling_design='temporal_tail_stride', iid_justified=False)
        with self.assertRaisesRegex(q.Insufficient, 'IID rows unsupported'):
            q.validate_protocol(self.protocol, metadata)
        protocol = dict(self.protocol, reviewed=False)
        with self.assertRaisesRegex(q.Insufficient, 'reviewed'):
            q.validate_protocol(protocol, self.metadata)

    def test_degenerate_r2_and_missing_off_refused(self):
        with self.assertRaisesRegex(q.Insufficient, 'R2 undefined'):
            q.score('regression', np.ones(32), np.ones(32))
        with self.assertRaisesRegex(q.Insufficient, 'wrong arm'):
            q.receipt(self.off[1], self.off[0], self.targets, 'on')

    def test_group_draws_keep_entire_units_and_are_reproducible(self):
        p = copy.deepcopy(self.protocol)
        p['resampling'] = {'kind': 'groups', 'minimum_units': 20}
        row = {'y': np.arange(60), 'group_key': np.repeat(np.arange(20), 3)}
        left, right = q.resamples(row, p, 0), q.resamples(row, p, 0)
        for _ in range(3):
            a, b = next(left), next(right)
            np.testing.assert_array_equal(a, b)
            counts = np.bincount(a, minlength=60).reshape(20, 3)
            self.assertTrue(np.all(counts == counts[:, :1]))
        with self.assertRaisesRegex(q.Insufficient, 'too few'):
            next(q.resamples({'y': np.arange(10), 'group_key': np.arange(10)}, p, 0))

    def test_blocks_follow_verified_order_and_reject_bad_permutation(self):
        p = copy.deepcopy(self.protocol)
        p['resampling'] = {'kind': 'moving_block', 'minimum_units': 20, 'block_length': 4}
        row = {'y': np.arange(100), 'order_key': np.arange(99, -1, -1)}
        sample = next(q.resamples(row, p, 0)).reshape(-1, 4)
        self.assertTrue(np.all((sample[:, 1:] - sample[:, :-1]) % 100 == 99))
        row['order_key'][:] = 0
        with self.assertRaisesRegex(q.Insufficient, 'permutation'):
            next(q.resamples(row, p, 0))

    def test_freeze_refuses_overwrite(self):
        path = self.root / 'immutable.json'
        q.write_new(path, {'original': True})
        with self.assertRaises(FileExistsError):
            q.write_new(path, {'original': False})
        self.assertTrue(q.read_json(path)['original'])


if __name__ == '__main__':
    unittest.main()
